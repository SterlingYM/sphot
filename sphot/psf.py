import numpy as np
import matplotlib.pyplot as plt
import warnings
import traceback

from astropy.utils.exceptions import AstropyUserWarning
from astropy.table import QTable, vstack
from astropy.stats import SigmaClip, sigma_clipped_stats

from photutils.aperture import EllipticalAperture
from astropy.utils import lazyproperty
from photutils.psf import SourceGrouper, PSFPhotometry, ImagePSF
from photutils.detection import DAOStarFinder
from photutils.background import (MMMBackground, MADStdBackgroundRMS,
                                  LocalBackground, MedianBackground,
                                  Background2D)
from photutils.datasets.images import make_model_image as _make_model_image

from .plotting import astroplot
from .logging import logger
from .config import config

def get_full_traceback(e):
    ''' Get the full traceback of an exception as a string. '''
    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
    tb_text = ''.join(tb_lines)
    return tb_text


class _MapCoordInterpolator:
    """Cubic-spline PSF evaluator backed by `scipy.ndimage.map_coordinates`.

    Drop-in for the `RectBivariateSpline` that `ImagePSF.interpolator`
    builds: same cubic B-spline interpolation, agreeing to ~1e-16
    relative, but 3-7x faster for the access pattern PSF photometry
    actually uses (a small stamp per source, evaluated over and over by
    the LM fitter). `RectBivariateSpline.__call__(grid=False)` goes
    through FITPACK's generic bispeu per point; `map_coordinates` is a
    tight C loop over a pre-filtered array.

    The spline pre-filter is computed ONCE here and reused with
    `prefilter=False`; letting map_coordinates re-filter the full PSF on
    every call would cost more than it saves.
    """

    def __init__(self, data):
        from scipy.ndimage import spline_filter
        self._filtered = spline_filter(
            np.ascontiguousarray(np.asarray(data, dtype=float)),
            order=3, mode='constant')

    def __call__(self, xi, yi, grid=False):
        from scipy.ndimage import map_coordinates
        xi = np.asarray(xi, dtype=float)
        yi = np.asarray(yi, dtype=float)
        coords = np.stack([yi.ravel(), xi.ravel()], axis=0)
        vals = map_coordinates(self._filtered, coords, order=3,
                                mode='constant', cval=0.0, prefilter=False)
        return vals.reshape(xi.shape)


class FastImagePSF(ImagePSF):
    """`ImagePSF` with the faster interpolator. photutils documents
    `interpolator` as the intended subclass override point.

    Only safe when `fill_value` is not None: photutils masks
    out-of-bounds pixels to `fill_value` after evaluation, which hides
    the one behavioural difference (RectBivariateSpline extrapolates
    past the PSF edge, map_coordinates returns 0 there).
    """

    @lazyproperty
    def interpolator(self):
        return _MapCoordInterpolator(self.data)


def make_image_psf(data, oversampling, fill_value=0.0, **kwargs):
    ''' Build the PSF model used for photometry, honouring
    `[psf].fast_psf_interpolator`. '''
    cls = ImagePSF
    if fill_value is not None and bool(
            config['psf'].get('fast_psf_interpolator', True)):
        cls = FastImagePSF
    return cls(np.asarray(data, dtype=float), flux=1.0,
               oversampling=oversampling, fill_value=fill_value, **kwargs)


def _build_center_mask(shape, center_mask_params):
    """Build a 2D boolean mask (True = exclude) covering the galaxy
    centre, so DAO/find_peaks/PSFPhotometry won't detect or fit the
    under-modelled core/disc as a point source.

    `center_mask_params` accepts either:
      * `[x_center, y_center, radius]` — circular mask (back-compat).
      * `[x_center, y_center, a, b, theta]` — elliptical mask with
        semi-major `a`, semi-minor `b`, and position angle `theta`
        (radians, CCW from +x). Matches Sersic2D's convention so the
        calibration mask follows the fitted galaxy shape rather than
        masking only a face-on circle.

    Returns `None` if params are missing or the radius/axes are
    non-positive.
    """
    if center_mask_params is None:
        return None
    if len(center_mask_params) == 3:
        x_c, y_c, r = center_mask_params
        if r is None or float(r) <= 0:
            return None
        yy, xx = np.indices(shape, dtype=float)
        return ((xx - float(x_c)) ** 2
                + (yy - float(y_c)) ** 2) <= float(r) ** 2
    if len(center_mask_params) == 5:
        x_c, y_c, a, b, theta = center_mask_params
        if a is None or b is None or float(a) <= 0 or float(b) <= 0:
            return None
        x_c, y_c = float(x_c), float(y_c)
        a, b, theta = float(a), float(b), float(theta)
        yy, xx = np.indices(shape, dtype=float)
        dx = xx - x_c
        dy = yy - y_c
        ct, st = np.cos(theta), np.sin(theta)
        x_rot = dx * ct + dy * st
        y_rot = -dx * st + dy * ct
        return (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0
    raise ValueError(
        f'center_mask_params must be length 3 (circle) or 5 (ellipse), '
        f'got length {len(center_mask_params)}')

class PSFFitter():
    ''' A class to perform PSF fitting. '''
    def __init__(self,cutoutdata):
        self.cutoutdata = cutoutdata
        self.psf_sigma = cutoutdata.psf_sigma
        self.psf_model = self.psf_img2model(cutoutdata.psf,cutoutdata.psf_oversample)

    def psf_img2model(self,psfimg,psf_oversample):
        psf_model = make_image_psf(
            psfimg, psf_oversample, fill_value=0.0, x_0=0, y_0=0)
        return psf_model

    def fit(self,fit_to='sersic_residual',**kwargs):
        ''' Perform PSF fitting via iterative_psf_fitting (wraps
        do_psf_photometry with a threshold ladder so we don't fit >1000
        sources simultaneously in crowded fields).

        The PSF used is `cutoutdata.psf` as-is. Mainloop-level kernel
        calibration (`_maybe_recalibrate_psf` -> `calibrate_psf_step`)
        updates `cutoutdata.psf` with `library * K` between
        iterations; this fitter just reads the current PSF state and
        runs the ladder.

        Args:
            fit_to (str): attribute name of the image to fit, e.g.
                'sersic_residual', 'residual', 'data'.
            kwargs (dict): extra kwargs forwarded to do_psf_photometry.

        Returns:
            CutoutData: updated in-place.
        '''
        self.data = getattr(self.cutoutdata,fit_to)
        # Re-pick up cutoutdata.psf in case calibrate_psf_step replaced it
        # since this PSFFitter was constructed.
        self.psf_model = self.psf_img2model(
            self.cutoutdata.psf, self.cutoutdata.psf_oversample)
        self.psf_sigma = self.cutoutdata.psf_sigma

        x0 = self.cutoutdata.sersic_params_physical['x_0']
        y0 = self.cutoutdata.sersic_params_physical['y_0']
        # Mask size is in units of the EFFECTIVE PSF FWHM (post calibration),
        # not library psf_sigma. The Sersic-residual blob at the galaxy core
        # scales with the convolved PSF; library σ is sub-pixel on sharp
        # filters and produces a useless mask.
        from .calibrate_psf import effective_fwhm_data_px
        cm_factor = float(config['psf'].get('center_mask_r_in_fwhm', 1.5))
        center_mask_r = effective_fwhm_data_px(self.cutoutdata) * cm_factor
        center_mask_params = [x0, y0, center_mask_r]

        # Threshold ladder.
        th_min = config['psf'].get('th_min',1.5)
        th_max = config['psf'].get('th_max',4.0)
        th_iter = config['psf'].get('th_iter',10)
        threshold_list = np.geomspace(th_min, th_max, num=th_iter)[::-1]

        # Warm start. The ladder's job is to DISCOVER sources by peeling
        # bright ones off so faint neighbours emerge. Between main-loop
        # iterations only the Sersic model changes, so the catalogue found
        # last iteration is still very nearly the right answer — rerunning
        # ~8 full detection passes to rediscover it is the single biggest
        # cost in the fit. Instead, seed the joint refit with the previous
        # catalogue and let its own leftover-detection loop pick up what
        # moved. `ladder_full_every` forces a full ladder every Nth call so
        # any drift is re-anchored (0 = warm start forever after the first).
        init_sources = None
        n_call = int(getattr(self.cutoutdata, '_ipsf_call_count', 0))
        if bool(config['psf'].get('ladder_warm_start', False)):
            prev = getattr(self.cutoutdata, 'psf_table', None)
            full_every = int(config['psf'].get('ladder_full_every', 3))
            due_full = (n_call == 0 or
                        (full_every > 0 and n_call % full_every == 0))
            if prev is not None and len(prev) > 0 and not due_full:
                names = (prev.dtype.names
                         if hasattr(prev, 'dtype') and prev.dtype.names
                         else prev.colnames)
                if all(c in names for c in ('x_fit', 'y_fit', 'flux_fit')):
                    init_sources = QTable()
                    init_sources['x'] = np.asarray(prev['x_fit'], dtype=float)
                    init_sources['y'] = np.asarray(prev['y_fit'], dtype=float)
                    init_sources['flux'] = np.asarray(prev['flux_fit'],
                                                      dtype=float)
        self.cutoutdata._ipsf_call_count = n_call + 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psf_table, resid = iterative_psf_fitting(
                self.data,
                self.psf_model,
                self.psf_sigma,
                threshold_list = threshold_list,
                center_mask_params=center_mask_params,
                init_sources=init_sources,
                **kwargs)

        if resid is None:
            # iterative_psf_fitting bailed out (e.g. subtract_background failed).
            # Leave cutoutdata's PSF attributes untouched so callers can detect
            # the no-op via psf_table being None.
            self.cutoutdata.psf_table = None
            return self.cutoutdata

        # Exclude catastrophic-chi² fits from psf_model_total (psf_table
        # itself is preserved — only the subtracted image is corrected).
        resid = _rebuild_resid_excluding_catastrophic_chi2(
            psf_table, resid, self.psf_model)

        psf_model_total = _rebase_psf_model(self.data - resid)
        # TODO: handle the case where all pixels are filled with PSF

        # generate PSF-subtracted data
        mask, bkg_std = sigma_clip_outside_aperture(
            resid,
            self.cutoutdata.sersic_params_physical,
            clip_sigma         = config['psf']['residual_clip_sigma'], # don't go too low
            aper_size_in_r_eff = config['psf']['mask_aper_size_in_r_eff'],
            )
        psf_subtracted_data = self.cutoutdata._rawdata - psf_model_total
        psf_subtracted_data[mask] = np.nan
        psf_subtracted_data_error = np.ones_like(psf_subtracted_data)*bkg_std
        
        # make the residual image
        sersic_modelimg = getattr(self.cutoutdata,'sersic_modelimg',0)
        residual_img = self.cutoutdata._rawdata - psf_model_total - sersic_modelimg
        residual_masked = residual_img.copy()
        residual_masked[mask] = np.nan
        
        # save data
        self.cutoutdata.residual = residual_img
        self.cutoutdata.residual_masked = residual_masked
        self.cutoutdata.psf_modelimg = psf_model_total
        self.cutoutdata.psf_sub_data = psf_subtracted_data 
        self.cutoutdata.psf_sub_data_error = psf_subtracted_data_error
        self.cutoutdata.psf_table = psf_table
        return self.cutoutdata

def _rebase_psf_model(psf_model_total):
    ''' Optionally re-zero the reconstructed PSF-model image.

    `psf_model_total` is `data - resid`, i.e. a DIFFERENCE OF IMAGES -- not a
    rendered sum of PSFs. It therefore can and does go negative (a negative
    LM flux, or a source added back into `resid` by
    `_rebuild_resid_excluding_catastrophic_chi2`). The legacy behaviour
    subtracted its global minimum, on the assumption that "PSFs are positive
    so the minimum is zero". When that assumption fails, ONE negative pixel
    rebases the entire frame: measured on N5468/g000 F150W, a -1.05 minimum
    put a flat +1.05 pedestal across 75% of the image (sky there is 0.34).
    That pedestal propagates into `psf_sub_data`, forces `remove_sky` to fit
    a NEGATIVE sky to compensate, and leaves a residual floor that the
    positive-only Sersic can only absorb by inflating r_eff.

    Modes (`[psf].psf_model_rebase`):
      'none' -- leave the reconstruction alone (default)
      'min'  -- legacy: subtract the global minimum (for reproducing old runs)
    '''
    mode = str(config['psf'].get('psf_model_rebase', 'none')).lower()
    if mode == 'min':
        return psf_model_total - np.nanmin(psf_model_total)
    lo = float(np.nanmin(psf_model_total))
    if lo < 0:
        logger.debug(f'psf_model_total has min={lo:.4g}; left as-is '
                     f'(psf_model_rebase={mode!r})')
    return psf_model_total


def _rebuild_resid_excluding_catastrophic_chi2(
        psf_table, resid, psf_model,
        chi2_factor=5.0, min_sources=5):
    """Return a new `resid` image with catastrophic-chi² sources'
    PSF-model contributions added back in, so that
    `psf_model_total = data − resid` excludes them. **`psf_table` is
    not modified** — downstream code (mainloop Sersic refit, sky fit,
    calibrator) still sees the full source list as iPSF reported it.

    Motivation: in crowded fields, the leftover-detection loop in
    `_final_joint_refit` can pick up local maxima on noise / Sersic-
    residual substructure that aren't real point sources. The pinned-
    position joint refit then assigns those phantoms whatever flux
    best matches at the wrong location — typically via PSF-wing
    overlap with a real source a few pixels away. Their fitted
    reduced_chi2 is wildly larger than well-behaved sources (e.g.
    chi2 ~ 80–90 vs median ~3 in one observed case), and their
    contribution to psf_model_total over-subtracts the data at the
    wrong centre, leaving deep negative pixels in psf_sub_data.

    The minimal-surface fix is to exclude only the phantoms from the
    subtracted-image reconstruction. Removing them from `psf_table`
    also propagates changes into Sersic / calibration state via
    residual_masked and downstream remove_sky steps, which can bias
    the calibrated kernel σ unintentionally.

    `chi2_factor`: any row with reduced_chi2 > chi2_factor × median is
    treated as catastrophic (capped at a minimum of 1.0 so a small
    median doesn't over-prune).
    `min_sources`: skip the filter if there are fewer than this many
    rows — the median is unreliable on small N.
    """
    if psf_table is None or len(psf_table) < min_sources or resid is None:
        return resid
    if 'reduced_chi2' not in psf_table.colnames:
        return resid
    chi2 = np.asarray(psf_table['reduced_chi2'], dtype=float)
    finite = np.isfinite(chi2)
    if int(finite.sum()) < min_sources:
        return resid
    chi2_med = float(np.nanmedian(chi2[finite]))
    cap = max(chi2_med * float(chi2_factor), 1.0)
    bad = chi2 > cap
    bad_idx = np.where(bad)[0]
    if bad_idx.size == 0:
        return resid
    yy, xx = np.indices(resid.shape, dtype=float)
    new_resid = resid.copy()
    n_added = 0
    for i in bad_idx:
        f_i = float(psf_table['flux_fit'][i])
        x_i = float(psf_table['x_fit'][i])
        y_i = float(psf_table['y_fit'][i])
        if not (np.isfinite(f_i) and np.isfinite(x_i) and np.isfinite(y_i)):
            continue
        try:
            m_i = psf_model.evaluate(xx, yy, flux=f_i, x_0=x_i, y_0=y_i)
        except Exception:
            continue
        m_i = np.where(np.isfinite(m_i), m_i, 0.0)
        new_resid = new_resid + m_i
        n_added += 1
    logger.info(
        f'PSF model: excluded {n_added} catastrophic-chi² source(s) '
        f'(reduced_chi2 > {chi2_factor}× median {chi2_med:.2f}) '
        f'from psf_model_total to avoid over-subtraction; '
        f'psf_table retains them.')
    return new_resid


def filter_psfphot_results(phot_result,
                           center_mask_params=None,
                           full_output=False,
                           bkg_std=None,
                           **kwargs):
    ''' Filter the PSF photometry results. '''

    # load config
    cuts_cfit_sigma_clip = config['psf']['cuts_cfit_sigma_clip']
    cuts_pos_diff_median_factor = config['psf']['cuts_pos_diff_median_factor']
    cuts_flux_SNR_min = config['psf']['cuts_flux_SNR_min']
    cuts_res_cen_sigma_clip = config['psf']['cuts_res_cen_sigma_clip']
    cuts_pos_err_max = config['psf']['cuts_pos_err_max']
    
    # load quantities from phot_result
    res_cen = phot_result['cfit']*phot_result['flux_fit']
    cfit = phot_result['cfit']
    qfit = phot_result['qfit']
    x_diff = phot_result['x_fit'] - phot_result['x_init']
    y_diff = phot_result['y_fit'] - phot_result['y_init']
    npixfit = phot_result['n_pixels_fit']
    xerr = phot_result['x_err']
    yerr = phot_result['y_err']

    # flag cuts
    #----------------------------
    # 1: npixfit smaller than full fit_shape region
    # 2: fitted position outside input image bounds
    # 4: non-positive flux
    # 8: possible non-convergence
    # 16: missing parameter covariance
    # 32: fitted parameter near a bound
    # 64: no overlap with data
    # 128: fully masked source
    # 256: too few pixels for fitting
    #----------------------------
    s_flags = ~(phot_result['flags'].value & (2+4+32+64+128+256)).astype(bool)
    s = s_flags.copy()
    
    # --- quality cuts ---
    bad_fits = np.zeros(s.size, dtype=bool)
    
    # 0. position error cuts
    bad_fits |= (xerr > cuts_pos_err_max) | (yerr > cuts_pos_err_max)

    # 1. absolute residual should be within (0, 2*median)
    # _, _, res_cen_std = sigma_clipped_stats(res_cen[s], sigma=3)
    N = cuts_res_cen_sigma_clip
    # bad_fits |= res_cen > max(res_cen_std * N, bkg_std * N) # if res_cen_std is large, it might just mean the residuals are crowded
    bad_fits |= res_cen < -3*N * bkg_std  # avoid over-subtraction (which is mostly just bad fits)
    bad_fits |= (res_cen < -N* bkg_std) & (qfit/np.sqrt(npixfit) > N * bkg_std) # negative center, large overall offset -> bad fit
    
    # 2. N-sigma cuts for cfit
    N = cuts_cfit_sigma_clip
    _, cfit_median, cfit_std = sigma_clipped_stats(cfit[s], sigma=N)
    cfit_std = max(cfit_std, 0.01)  # avoid too small std
    bad_fits |= (cfit<(cfit_median-N*cfit_std)) # | (cfit>cfit_median+N*cfit_std) 
    
    # 3. position difference should be within (0, N*median)
    N = cuts_pos_diff_median_factor
    pos_diff = np.sqrt(x_diff**2 + y_diff**2)
    bad_fits |= pos_diff > np.nanmedian(pos_diff[s])*N

    # 4. flux SNR cut
    flux_SNR = phot_result['flux_fit']/phot_result['flux_err']
    s_fluxerr = flux_SNR >= cuts_flux_SNR_min
    
    s = s_flags & (~bad_fits) & s_fluxerr
    s_dict = {}
    msg = ''
    
    if center_mask_params is not None:
        x_center,y_center,mask_r = center_mask_params
        xdist = phot_result['x_fit'] - x_center
        ydist = phot_result['y_fit'] - y_center
        s_centermask = (xdist**2 + ydist**2 > mask_r**2)
        s_dict['s_centermask'] = s_centermask
        s = s & s_centermask
        msg += f'    center_mask: {int(s_centermask.sum())}\n'

    msg += f'Sources that passed all of the above cuts: {int(s.sum())}\n'
    if full_output:
        return s, s_dict, msg
    return s, msg


def _dedup_phot_against(new_phot, existing_phot, dedup_radius):
    ''' Drop rows in `new_phot` whose (x_fit, y_fit) is within
    `dedup_radius` (in data px) of any row in `existing_phot`. Returns
    the filtered slice of `new_phot`.

    Used inside `iterative_psf_fitting`'s ladder loop so that sources
    re-detected by multiple ladder passes (because the iterative
    subtraction left a residual bump where a fit had been) are not
    written to psf_table multiple times. Without this the saved
    psf_table can contain dozens of near-coincident duplicate entries
    that poison downstream NNLS refits.
    '''
    if (existing_phot is None or len(existing_phot) == 0
            or new_phot is None or len(new_phot) == 0):
        return new_phot
    nx = np.asarray(new_phot['x_fit'], dtype=float)
    ny = np.asarray(new_phot['y_fit'], dtype=float)
    ex = np.asarray(existing_phot['x_fit'], dtype=float)
    ey = np.asarray(existing_phot['y_fit'], dtype=float)
    r2 = float(dedup_radius) ** 2
    keep = np.ones(len(new_phot), dtype=bool)
    for i in range(len(new_phot)):
        dx = nx[i] - ex
        dy = ny[i] - ey
        if (dx * dx + dy * dy).min() <= r2:
            keep[i] = False
    return new_phot[keep]


def sigma_clip_outside_aperture(data,sersic_params_physical,clip_sigma=4,
                                aper_size_in_r_eff=1):
    
    # subtract large scale variation (likely residual from galaxy)
    data_bksub,bkg_std,_ = subtract_background(data)
    
    # sigma-clip pixels outside r_eff
    mask = np.abs(data_bksub) > clip_sigma*bkg_std
    # mask = sigma_clip(data_bksub,sigma=clip_sigma).mask
    
    # exclude pixels inside the 1-Reff isophot aperture
    a = sersic_params_physical['r_eff'] * aper_size_in_r_eff
    x_0 = sersic_params_physical['x_0']
    y_0 = sersic_params_physical['y_0']
    ellip = sersic_params_physical['ellip']
    theta = sersic_params_physical['theta']
    # Defensive clamps: EllipticalAperture requires a,b > 0. A degenerate
    # Sersic fit (r_eff → 0 or ellip → 1) would otherwise crash the
    # pipeline here. Floor at 1 px so the aperture covers at least the
    # central pixel; downstream this just means a 1-pixel core mask.
    a = max(float(a), 1.0)
    b = max((1 - float(ellip)) * a, 1.0)
    aperture = EllipticalAperture((x_0, y_0), a, b, theta=theta)
    # aperture = CircularAperture((data.shape[0]/2,data.shape[1]/2),
    #                             r_eff*aper_size_in_r_eff)
    aperture_mask = aperture.to_mask(method='center')
    aperture_mask_img = aperture_mask.to_image(data.shape).astype(bool)
    mask[aperture_mask_img] = False
    
    return mask,bkg_std # bad pixels are True

def subtract_background(data):
    ''' Subtract background from data. This removes most of the large-scale background variations.'''
    bkgrms = MADStdBackgroundRMS()
    bkg_estimator = MedianBackground()
    mmm_bkg = MMMBackground()

    sigma = config['psf']['bkg_sigma_clip']
    box_size = config['psf']['bkg_box_size']
    filter_size = config['psf']['bkg_filter_size']

    sigma_clip = SigmaClip(sigma=sigma)
    bkg = Background2D(data, box_size, filter_size=filter_size,
                    sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    data_bksub = data - bkg.background

    # the above is somehow often slightly offset from true zero.
    # this can be corrected by running MMMBackground
    data_bksub -= mmm_bkg(data_bksub)

    bkg_std = bkgrms(data_bksub)
    data_error = np.ones_like(data_bksub) * bkg_std

    return data_bksub,bkg_std,data_error

def _resolve_fit_shape(psf_sigma):
    """PSFPhotometry fit_shape sized to a fraction of FWHM.

    Reads `[psf].PSFPhotometry_fit_shape_in_fwhm` (default 1.5). Returns
    (n, n) with n odd and >= 3 — a hard floor of 3 prevents
    photutils from collapsing to a 1x1 single-pixel fit on extremely
    sharp PSFs. With f=1.5 the default fit captures ~94% of a
    Gaussian's flux; broader PSFs (e.g. F160W FWHM~6.6 px) get a
    proportionally wider window so flux isn't fit on a near-flat 3x3.
    """
    frac = float(config['psf'].get('PSFPhotometry_fit_shape_in_fwhm', 1.5))
    n = int(np.ceil(2.355 * float(psf_sigma) * frac))
    if n % 2 == 0:
        n += 1
    n = max(3, n)
    return (n, n)


def _prepare_psf_fitters(th,psf_model,bkg_std,psf_sigma):
    finder_kwargs = config['psf']['finder_kwargs']
    # DAO matched-filter width. Defaults to 2.33 (sphot's historical
    # value). When proto_19's per-iter PSF calibration is active,
    # `dao_fwhm_factor` is updated each main-loop iteration to the
    # value that maximises quality-passing source count for the
    # currently-effective kernel-modulated PSF.
    fwhm_factor = float(config['psf'].get('dao_fwhm_factor', 2.33))
    daofinder = DAOStarFinder(
        threshold=th*bkg_std,
        fwhm=psf_sigma*fwhm_factor, **finder_kwargs)

    localbkg_estimator = LocalBackground(
        config['psf']['localbkg_bounds_in_psfsigma'][0]*psf_sigma,
        config['psf']['localbkg_bounds_in_psfsigma'][1]*psf_sigma,
        MMMBackground()
        )
    grouper = SourceGrouper(
        min_separation=config['psf']['grouper_separation_in_psfsigma'] * psf_sigma
        )

    psf_single = PSFPhotometry(
        psf_model,
        fit_shape          = _resolve_fit_shape(psf_sigma),
        finder             = daofinder,
        grouper            = grouper,
        local_bkg_estimator= localbkg_estimator,
        aperture_radius    = config['psf']['PSFPhotometry_aperture_radius'],
        fitter_maxiters    = config['psf']['PSFPhotometry_fitter_maxiters'],
        group_warning_threshold = config['psf']['PSFPhotometry_group_warning_threshold'],
    )
    return psf_single

def do_psf_photometry(data,data_error,bkg_std,
                      psf_model,psf_sigma,
                      th=2,
                      plot=True,
                      mask=None,
                      **kwargs):
    """Performs PSF photometry. Main function to run PSF photometry.

    This function does the following:

    1. turn psfimg into a fittable model
    2. estimate the background statistics (mean, std)
    3. subtract background from data
    4. perform IterativePSFPhotometry with the finder threshold at th*std
    5. filter the results based on the fit quality (cfit, qfit, flux_err)
    6. perform PSFPhotometry using the filtered results as input
    7. filter the results again
    8. generate model image and residual image
    9. plot the results if necessary

    Args:
        data (2d array): the data to perform PSF photometry.
        psfimg (2d array): the PSF image.
        psf_sigma (float): the HWHM of the PSF. Use FWHM/2
        psf_oversample (int): the oversampling factor of the PSF.
        th (float): the detection threshold in background STD.
        Niter (int): the number of iterations to repeat the photometry (after cleaning up the data). -- !deprecated!
        fit_shape (2-tuple): the shape of the fit.
        render_shape (2-tuple): the shape of each PSF to be rendered.
        finder_kwargs (dict,optional): the kwargs for DAOStarFinder.
        localbkg_bounds (2-tuple,optional): (inner, outer) radii to LocalBackground object, in the unit of psf_sigma.
        grouper_sep (float,optional): the minimum separation between sources to be used for SourceGrouper.
    Returns:
        phot_result (QTable): the photometry result.
        resid (2d array): the residual image.
    """
    # initialize fitters
    psf_single = _prepare_psf_fitters(th, psf_model, bkg_std, psf_sigma)

    with warnings.catch_warnings():
        # abort if there are too many sources in a group
        warnings.filterwarnings(
            "error",
            message=r"Some groups have more than \d+ sources\.",  # regex match
            category=AstropyUserWarning,
        )
        try:
            # fit all simultaneously
            phot_result = psf_single(data, error=data_error, mask=mask)
        except AstropyUserWarning as e:
            # logger.error(f'Too many blended sources. Aborting th={th}...')
            N_blend = config['psf']['PSFPhotometry_group_warning_threshold']
            raise Exception(f'too many (>{N_blend}) blended sources.')
        except Exception as e:
            logger.info(f'IterativePSFPhotometry failed due to: {str(e)}')
            raise 
        
    if phot_result is None:
        logger.info('No detection.')
        return None,None
    
    # generate model image    
    s,msg = filter_psfphot_results(phot_result,bkg_std=bkg_std,**kwargs)
    params_table = QTable()
    params_table['x_0'] = phot_result['x_fit'].value[s]
    params_table['y_0'] = phot_result['y_fit'].value[s]
    params_table['flux'] = phot_result['flux_fit'].value[s]
    model_img = _make_model_image(
        shape = data.shape, 
        model = psf_model, #psf_iter._psfphot.psf_model, 
        params_table = params_table, 
        model_shape = config['psf']['modelimg_render_shape'],
        x_name='x_0', y_name='y_0')
    
    resid = data - model_img

    # plot the results if necessary
    if plot:
        fig,axes = plt.subplots(1,3,figsize=(15,5))
        norm,offset = astroplot(data,ax=axes[0],percentiles=[0.1,99.9])
        astroplot(model_img,ax=axes[1],norm=norm,offset=offset)
        astroplot(resid,ax=axes[2],norm=norm,offset=offset)

    return phot_result, resid

def iterative_psf_fitting(data,psf_model,psf_sigma,
                          threshold_list,
                          progress=None,
                          progress_text='Running iPSF...',
                          init_sources=None,
                          **kwargs):
    ''' Iteratively run do_psf_photometry() with different threshold levels.
    This function is useful for crowded fields, where a single threshold level may fail.

    The threshold list is typically descending (high -> low). Each successful
    pass subtracts detected sources from the residual; the next pass then
    operates on the cleaner residual. The ladder early-exits when
    `th_max_consec_empty` consecutive thresholds add no new sources.

    Args:
        data (2d array): the data to perform PSF photometry.
        psf_model: the PSF ImagePSF model.
        psf_sigma (float): the HWHM of the PSF. Use FWHM/2
        threshold_list (1d array): the list of threshold levels to try, in background STD.
        center_mask_params (list, optional): ``[x_center, y_center, mask_r]``. If provided, sources within ``mask_r`` from ``(x_center, y_center)`` are excluded from the final results. Useful when the central source is very bright and causes many spurious detections nearby.
        kwargs (dict): additional kwargs to pass to do_psf_photometry.

    Returns:
        phot_result (QTable): the combined photometry result from all iterations.
        resid_all (2d array): the final residual image.
    '''
      
    # prepare background-subtracted data
    # take data stats & prepare background-subtracted data
    try:
        data_bksub, bkg_std, data_error = subtract_background(data)
        kwargs.update({'data_shape':data_bksub.shape})
    except Exception as e:
        logger.warning(f'PSF background stats failed ({str(e)}); '
                       f'skipping iterative_psf_fitting')
        return None, None
          
    # initialize variables
    resid = data_bksub.copy()
    phot_result = None
    last_successful_th = None
    # Pre-detection mask covering the Sersic centre. Excludes those
    # pixels from PSFPhotometry (DAO + LM), so the under-modelled
    # galaxy core can't be detected/fit as a spurious point source.
    # `kwargs['center_mask_params']` is also kept for the post-fit
    # `filter_psfphot_results` filter (defence-in-depth).
    center_mask_params = kwargs.get('center_mask_params', None)
    center_mask = _build_center_mask(data_bksub.shape, center_mask_params)
    # Early-stop control: stop after this many consecutive thresholds add no
    # new sources. The threshold list goes high->low, so once a couple of
    # noise-floor passes return nothing, lower thresholds will only add noise.
    max_consec_empty = config['psf'].get('th_max_consec_empty', 3)
    consec_empty = 0

    # In a crowded field the initial bkg_std is dominated by the unresolved
    # star carpet, so a "1.5 sigma" threshold is really "1.5 sigma above the
    # carpet". As we subtract sources iteratively the residual gets cleaner
    # and the true noise floor drops -- re-estimate after each successful
    # pass so subsequent thresholds catch the dim sources we just exposed.
    refit_bkg = config['psf'].get('bkg_refit_per_iteration', True)
    bkg_floor_factor = float(config['psf'].get('bkg_floor_factor', 0.3))
    initial_bkg_std = float(bkg_std)

    # Warm start: skip discovery entirely and hand the previous catalogue
    # straight to the joint refit (which re-fits every source against the
    # current image and runs its own leftover detection for anything new).
    if init_sources is not None and len(init_sources) > 0:
        try:
            new_phot, new_resid = _final_joint_refit(
                data_bksub, data_error, bkg_std,
                psf_model, psf_sigma, None,
                init_override=init_sources, progress=progress,
                **{k: v for k, v in kwargs.items() if k != 'progress'})
        except Exception as e:
            if config['psf']['raise_error']:
                raise
            logger.warning(f'warm-start refit failed: {e}; '
                           f'falling back to the full ladder')
            new_phot, new_resid = None, None
        if new_phot is not None and new_resid is not None:
            logger.info(f'iPSF warm start: {len(init_sources)} seed sources '
                        f'-> {len(new_phot)} fitted (ladder skipped)')
            psf_modelimg_all = data_bksub - new_resid
            return new_phot, data.copy() - psf_modelimg_all

    # loop -- repeat PSF subtraction
    if progress is not None:
        progress_psf = progress.add_task(progress_text,
                                        total=len(threshold_list))
    for th in threshold_list:
        added_sources = False
        try:
            psf_results = do_psf_photometry(resid, data_error, bkg_std,
                                            psf_model, psf_sigma,
                                            th=th, mask=center_mask,
                                            **kwargs)
            if progress is not None:
                progress.update(progress_psf, advance=1, refresh=True)
            if psf_results[0] is None:
                pass  # no detection -> empty pass
            else:
                _phot_result, _resid = psf_results
                if not np.all(~np.isfinite(_resid)) and len(_phot_result) > 0:
                    # append the results, deduplicating against existing rows.
                    # Without this, sources that re-detect across multiple
                    # ladder passes (because the iterative subtraction is
                    # imperfect and they leave a bump) accumulate into
                    # near-duplicate entries in psf_table. Downstream NNLS
                    # refits then have co-degenerate columns and split flux
                    # arbitrarily between them, producing structured positive
                    # residuals between marked sources in dense regions.
                    resid = _resid
                    if phot_result is None:
                        phot_result = _phot_result
                    else:
                        dedup_psfsigma = float(config['psf'].get(
                            'ladder_dedup_psfsigma',
                            config['psf'].get('final_refit_dedup_psfsigma',
                                              1.5)))
                        new_rows = _dedup_phot_against(
                            _phot_result, phot_result,
                            dedup_psfsigma * psf_sigma)
                        n_dropped = len(_phot_result) - len(new_rows)
                        if n_dropped > 0:
                            logger.debug(
                                f'  ladder dedup: dropped {n_dropped}/'
                                f'{len(_phot_result)} duplicate detections '
                                f'(within {dedup_psfsigma:.1f}*psf_sigma)')
                        if len(new_rows) > 0:
                            phot_result = vstack([phot_result, new_rows])
                    last_successful_th = th
                    added_sources = True
        except Exception as e:
            if config['psf']['raise_error']:
                raise
            logger.error(f'Skipping PSF fitting with th={th:.2f}. {str(e)}')

        if added_sources:
            consec_empty = 0
            # Re-estimate the noise floor from the freshly-cleaned residual.
            # In crowded fields this drops with each pass as we peel off the
            # unresolved-star carpet, so the next th=th*bkg_std threshold
            # corresponds to a lower absolute count and can pick up dim
            # stars that were buried in the apparent "noise" before.
            if refit_bkg:
                try:
                    bkgrms_estimator = MADStdBackgroundRMS()
                    new_std = float(bkgrms_estimator(resid))
                    floor = initial_bkg_std * bkg_floor_factor
                    bkg_std = max(new_std, floor)
                    data_error = np.ones_like(resid) * bkg_std
                except Exception as e:
                    logger.debug(f'bkg re-estimation failed at th={th:.2f}: '
                                 f'{e}')
        else:
            consec_empty += 1
            if (last_successful_th is not None
                    and consec_empty >= max_consec_empty):
                logger.debug(f'No new sources for {consec_empty} thresholds; '
                             f'stopping ladder early at th={th:.2f}.')
                if progress is not None:
                    # clear remaining ticks so the bar finishes cleanly
                    progress.update(progress_psf, completed=len(threshold_list),
                                    refresh=True)
                break
    if progress is not None:
        progress.remove_task(progress_psf)

    # Final refit step. Modes picked by `final_refit_method`:
    #   'iterative'  -- _final_joint_refit: iterative LM refit + leftover
    #                   detection. Honours the legacy `final_joint_refit`
    #                   bool for back-compat (false => skip the refit).
    #   'nnls'       -- _final_nnls_refit: build a global non-negative
    #                   least-squares system using `psf_model` for every
    #                   source, solve once, then run the find_peaks
    #                   leftover loop. Deterministic, no negative-flux
    #                   outputs, runs in seconds.
    #   'none'       -- skip the final refit entirely.
    method = str(config['psf'].get('final_refit_method', 'iterative')).lower()
    refit_func = None
    if phot_result is not None and len(phot_result) > 0:
        if method == 'nnls':
            refit_func = _final_nnls_refit
        elif method == 'iterative':
            if config['psf'].get('final_joint_refit', True):
                refit_func = _final_joint_refit
        elif method == 'none':
            refit_func = None
        else:
            logger.warning(f'unknown final_refit_method={method!r}; '
                           f'falling back to iterative')
            if config['psf'].get('final_joint_refit', True):
                refit_func = _final_joint_refit
    if refit_func is not None:
        try:
            new_phot, new_resid = refit_func(
                data_bksub, data_error, bkg_std,
                psf_model, psf_sigma, phot_result,
                progress=progress,
                **{k: v for k, v in kwargs.items() if k != 'progress'},
            )
            if new_phot is not None and new_resid is not None:
                phot_result = new_phot
                resid = new_resid
        except Exception as e:
            if config['psf']['raise_error']:
                raise
            logger.warning(f'final refit ({method}) failed: {e}')

    # make the residual image without background subtraction
    psf_modelimg_all = data_bksub - resid
    resid_all = data.copy() - psf_modelimg_all.copy()
    return phot_result, resid_all


def forced_psf_photometry(data, psf_model, psf_sigma, x_init, y_init,
                           *, flux_init=None, xy_bound=1e-6,
                           center_mask_params=None,
                           progress=None,
                           progress_text='forced PSF photometry'):
    """Closed-form NNLS forced photometry at FIXED positions.

    Bypasses photutils' `PSFPhotometry` entirely — that path uses a
    `SourceGrouper` + LM that scales poorly when many positions fall
    inside one wide-PSF group (e.g. F160W with ~556 base-filter
    positions can hang for hours in a single mega-group LM).

    Recipe (same as `_final_nnls_refit`'s inner solve): render each
    source's unit-flux PSF on the full image to form a column of the
    design matrix, then solve `data = cols @ flux` jointly via NNLS
    (Gram + Cholesky, with a dense fallback). One call, fast.

    Parameters
    ----------
    data : 2D ndarray
        Image to fit (typically `cd.sersic_residual`).
    psf_model : photutils ImagePSF
        Effective PSF model.
    psf_sigma : float
        Used only for the `xy_bound` legacy arg (no LM here).
    x_init, y_init : 1D float arrays
        Pinned source positions in data px.
    flux_init : ignored (NNLS doesn't need a flux seed).
    xy_bound : ignored (positions are FIXED in NNLS).
    center_mask_params : [x_c, y_c, r] | None
        Pixels inside this radius of the Sersic core are masked.

    Returns
    -------
    phot_table : QTable with x_fit, y_fit, flux_fit, flux_err, flags
    resid_image : data - model_image (full-frame, NaNs preserved)
    """
    from scipy.optimize import nnls
    try:
        data_bksub, bkg_std, data_error = subtract_background(data)
    except Exception as e:
        logger.info(f'forced PSF: background stats failed ({e})')
        return None, None

    x_init = np.asarray(x_init, dtype=float).ravel()
    y_init = np.asarray(y_init, dtype=float).ravel()
    if len(x_init) != len(y_init):
        raise ValueError(f'x_init and y_init must be the same length '
                         f'(got {len(x_init)} vs {len(y_init)})')
    n_total = len(x_init)
    if n_total == 0:
        return None, None

    H, W = data_bksub.shape

    # Mask: center exclusion + non-finite pixels.
    mask = _build_center_mask((H, W), center_mask_params)
    nonfinite = ~np.isfinite(data_bksub)
    if mask is None:
        mask_bool = nonfinite.copy() if nonfinite.any() else np.zeros((H, W), dtype=bool)
    else:
        mask_bool = np.asarray(mask, dtype=bool)
        if nonfinite.any():
            mask_bool = mask_bool | nonfinite

    if progress is not None:
        task = progress.add_task(
            f'{progress_text} (N={n_total})', total=2)

    # 1. design matrix: column i = unit-flux PSF rendered at (x_i, y_i).
    # Masked / non-finite pixels are zeroed in BOTH b and the matrix rows
    # so they contribute nothing to the normal equations.
    render_shape = tuple(config['psf']['modelimg_render_shape'])
    b = data_bksub.ravel().astype(np.float32)
    masked_flat = mask_bool.ravel()
    invalid_flat = masked_flat | ~np.isfinite(b)
    b = np.where(invalid_flat, 0.0, b)
    cols = _build_design_matrix(psf_model, x_init, y_init, (H, W),
                                render_shape, invalid_flat=invalid_flat)
    if progress is not None:
        progress.update(task, advance=1, refresh=True)

    # 2. NNLS solve.
    G = _design_gram(cols)
    rhs = _design_rmatvec(cols, b)
    if not (np.all(np.isfinite(G)) and np.all(np.isfinite(rhs))):
        logger.warning('forced NNLS: Gram/rhs has non-finite entries')
        if progress is not None:
            try: progress.remove_task(task)
            except Exception: pass
        return None, None

    try:
        Lc = np.linalg.cholesky(G + 1e-6 * np.eye(G.shape[0]))
        Linv_rhs = np.linalg.solve(Lc, rhs)
        flux_fit, _ = nnls(Lc.T, Linv_rhs, maxiter=10000)
    except (np.linalg.LinAlgError, RuntimeError):
        try:
            flux_fit, _ = nnls(_design_dense(cols).astype(np.float64),
                                b.astype(np.float64), maxiter=20000)
        except Exception as e:
            logger.warning(f'forced NNLS solve failed: {e}')
            if progress is not None:
                try: progress.remove_task(task)
                except Exception: pass
            return None, None

    # 3. flux uncertainties: σ_i ≈ sqrt(σ²_pix * (G⁻¹)_ii). σ_pix from
    # the residual MAD; fall back to bkg_std on degenerate Gram.
    model_flat = _design_matvec(cols, flux_fit.astype(np.float32))
    resid_flat = b - model_flat
    n_unmasked = int((~masked_flat).sum())
    if n_unmasked > n_total + 1:
        sigma_pix = float(np.sqrt(np.sum(resid_flat[~masked_flat] ** 2)
                                   / max(n_unmasked - n_total, 1)))
    else:
        sigma_pix = float(bkg_std)
    try:
        Ginv_diag = np.diag(np.linalg.pinv(G + 1e-6 * np.eye(G.shape[0])))
        flux_err = np.sqrt(np.maximum(Ginv_diag, 0.0)) * sigma_pix
    except Exception:
        flux_err = np.full(n_total, sigma_pix, dtype=float)

    # 4. assemble output. QTable with photutils-compatible column names
    # so downstream code (filter_psfphot_results, plotting, etc.) can
    # consume it the same way.
    phot_table = QTable()
    phot_table['x_init'] = x_init
    phot_table['y_init'] = y_init
    phot_table['x_fit'] = x_init
    phot_table['y_fit'] = y_init
    phot_table['flux_fit'] = flux_fit.astype(float)
    phot_table['flux_err'] = flux_err
    phot_table['x_err'] = np.zeros(n_total)
    phot_table['y_err'] = np.zeros(n_total)
    phot_table['flags'] = np.zeros(n_total, dtype=int)

    # 5. residual on the original data (preserve NaNs outside mask).
    model_2d = model_flat.reshape(H, W)
    resid_full = data.copy() - model_2d

    if progress is not None:
        progress.update(task, advance=1, refresh=True)
        try: progress.remove_task(task)
        except Exception: pass
    return phot_table, resid_full


def run_forced_photometry_on_cutout(cutoutdata, x_init, y_init,
                                     *, flux_init=None,
                                     fit_to='sersic_residual',
                                     xy_bound=1e-6,
                                     progress=None):
    """Run `forced_psf_photometry` on a CutoutData and populate the
    same downstream attrs `PSFFitter.fit` writes (residual, psf_table,
    psf_modelimg, psf_sub_data, psf_sub_data_error, residual_masked).

    Used as a drop-in replacement for `PSFFitter.fit` inside
    `run_scalefit_forced` so the rest of the pipeline (sersic re-fit,
    plotting, save) sees the same set of attributes regardless of
    whether iPSF or forced photometry produced them.
    """
    data = getattr(cutoutdata, fit_to)
    psf_sigma = float(cutoutdata.psf_sigma)
    psf_model = make_image_psf(
        cutoutdata.psf, int(cutoutdata.psf_oversample), fill_value=0.0)

    # Same centre-mask the iPSF path uses (in EFFECTIVE FWHM units).
    sp = getattr(cutoutdata, 'sersic_params_physical', None)
    center_mask_params = None
    if sp is not None:
        from .calibrate_psf import effective_fwhm_data_px
        cm_factor = float(config['psf'].get('center_mask_r_in_fwhm', 1.5))
        if cm_factor > 0:
            try:
                cm_r = effective_fwhm_data_px(cutoutdata) * cm_factor
                center_mask_params = [float(sp['x_0']), float(sp['y_0']), cm_r]
            except Exception:
                center_mask_params = None

    phot_table, resid = forced_psf_photometry(
        data, psf_model, psf_sigma, x_init, y_init,
        flux_init=flux_init, xy_bound=xy_bound,
        center_mask_params=center_mask_params,
        progress=progress)

    if phot_table is None:
        logger.warning('forced PSF returned no photometry; '
                       'leaving cutoutdata attrs unchanged.')
        return cutoutdata

    # Exclude catastrophic-chi² fits from psf_model_total (phot_table
    # itself is preserved — only the subtracted image is corrected).
    resid = _rebuild_resid_excluding_catastrophic_chi2(
        phot_table, resid, psf_model)

    psf_model_total = _rebase_psf_model(data - resid)

    mask, bkg_std = sigma_clip_outside_aperture(
        resid,
        cutoutdata.sersic_params_physical,
        clip_sigma=config['psf']['residual_clip_sigma'],
        aper_size_in_r_eff=config['psf']['mask_aper_size_in_r_eff'],
    )
    psf_subtracted_data = cutoutdata._rawdata - psf_model_total
    psf_subtracted_data[mask] = np.nan
    psf_subtracted_data_error = np.ones_like(psf_subtracted_data) * bkg_std

    sersic_modelimg = getattr(cutoutdata, 'sersic_modelimg', 0)
    residual_img = cutoutdata._rawdata - psf_model_total - sersic_modelimg
    residual_masked = residual_img.copy()
    residual_masked[mask] = np.nan

    cutoutdata.residual = residual_img
    cutoutdata.residual_masked = residual_masked
    cutoutdata.psf_modelimg = psf_model_total
    cutoutdata.psf_sub_data = psf_subtracted_data
    cutoutdata.psf_sub_data_error = psf_subtracted_data_error
    cutoutdata.psf_table = phot_table
    return cutoutdata


def _psf_support_shape(psf_model):
    """Model footprint in DATA px: the PSF image size divided by its
    oversampling, rounded up to odd. An ImagePSF is exactly zero outside
    this footprint (fill_value=0), so rendering with this `model_shape`
    reproduces photutils' own model image bit-for-bit rather than
    truncating the wings like the smaller `modelimg_render_shape`.
    """
    try:
        d = np.asarray(psf_model.data)
        ovs = np.atleast_1d(np.asarray(psf_model.oversampling, dtype=float))
        ny = int(np.ceil(d.shape[0] / float(ovs[0])))
        nx = int(np.ceil(d.shape[1] / float(ovs[-1])))
    except Exception:
        return tuple(config['psf']['modelimg_render_shape'])
    ny += (ny + 1) % 2
    nx += (nx + 1) % 2
    return (max(ny, 3), max(nx, 3))


def _render_phot_model(psf_model, phot, shape, model_shape):
    ''' Sum of fitted PSF models for every row of `phot`. Matches
    `PSFPhotometry.make_residual_image`'s model to float roundoff when
    `model_shape` is the full PSF support (see `_psf_support_shape`).
    '''
    x = np.asarray(phot['x_fit'], dtype=float)
    y = np.asarray(phot['y_fit'], dtype=float)
    f = np.asarray(phot['flux_fit'], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(f)
    if not ok.any():
        return np.zeros(shape, dtype=float)
    return _render_unit_image(psf_model, x[ok], y[ok], f[ok],
                              shape, model_shape)


def _final_joint_refit(data_bksub, data_error, bkg_std,
                       psf_model, psf_sigma, phot_result,
                       progress=None, init_override=None, **kwargs):
    ''' Iterative joint refit + leftover-source detection.

    Each iteration:
        1. Refit all current sources simultaneously (positions pinned by
           default via xy_bounds=0) on the original background-subtracted
           data. The grouper splits dense regions into independent LM
           subproblems.
        2. Detect leftover sources in the residual at
           ``final_refit_detect_th * MAD(resid)`` -- only POSITIVE peaks,
           so bright stars that have absorbed too much flux (negative
           central dip) cannot be re-added as new sources.
        3. Deduplicate detections against the existing init list (anything
           closer than ``min_separation_psfsigma * psf_sigma`` to an
           existing source is dropped) and append the new ones.
        4. Stop when no new sources are added, when the residual MAD stops
           dropping by more than ``residual_improvement_tol``, or after
           ``final_refit_iterations`` iterations.

    Pinning positions (xy_bounds=0 by default) prevents bright stars from
    drifting toward dim neighbours and stealing their flux (= central
    over-subtraction in the previous design). The iterative re-detection
    then ensures the dim neighbours that the brights *used to* absorb get
    their own model, so the residual doesn't have positive blobs next to
    the brights either.

    Returns (new_phot_result, new_residual_image), or (None, None) if no
    sources remain after quality filtering or the refit fails.
    '''
    center_mask_params = kwargs.get('center_mask_params', None)
    center_mask = _build_center_mask(data_bksub.shape, center_mask_params)
    # The mask handed to photutils must ALSO cover non-finite pixels.
    # Without this, any source whose fit window touches a NaN region kills
    # the ENTIRE grouped fit ("Objective function has encountered a
    # non-finite value"), the refit silently returns (None, None), and the
    # raw ladder result -- unrefit, negative fluxes included -- is what
    # gets saved as psf_table. Measured: 13 of ~16 iPSF calls on
    # N5468/g000 died this way.
    nonfinite = ~np.isfinite(data_bksub)
    if nonfinite.any():
        center_mask = (nonfinite if center_mask is None
                       else (center_mask | nonfinite))
    if init_override is not None:
        # Warm start: the seed catalogue already passed quality cuts when
        # it was built, so it is NOT re-filtered here — re-applying the
        # cuts every iteration would compound attrition and bleed sources
        # out of the catalogue over the main loop.
        init = init_override
    else:
        s_pass, _ = filter_psfphot_results(
            phot_result, center_mask_params=center_mask_params,
            bkg_std=bkg_std)
        if int(s_pass.sum()) == 0:
            return None, None

        init = QTable()
        init['x'] = np.asarray(phot_result['x_fit'][s_pass], dtype=float)
        init['y'] = np.asarray(phot_result['y_fit'][s_pass], dtype=float)
        init['flux'] = np.asarray(phot_result['flux_fit'][s_pass],
                                  dtype=float)
    if len(init) == 0:
        return None, None

    grouper = SourceGrouper(
        min_separation=config['psf']['grouper_separation_in_psfsigma'] * psf_sigma)
    localbkg = LocalBackground(
        config['psf']['localbkg_bounds_in_psfsigma'][0] * psf_sigma,
        config['psf']['localbkg_bounds_in_psfsigma'][1] * psf_sigma,
        MMMBackground())

    xy_bound = float(config['psf'].get('final_refit_xy_bounds', 0.0))
    n_iter = int(config['psf'].get('final_refit_iterations', 5))
    detect_th = float(config['psf'].get('final_refit_detect_th', 3.0))
    dedup_psfsigma = float(config['psf'].get(
        'final_refit_dedup_psfsigma', 1.5))
    resid_tol = float(config['psf'].get(
        'final_refit_residual_tol', 0.01))
    group_warn = int(config['psf'].get(
        'final_refit_group_warning_threshold', 10000))

    # The ladder's fit_shape scales with PSF FWHM; the joint refit can
    # afford a wider window because positions are pinned and the LM only
    # has to solve for fluxes. Falls back to the FWHM-derived ladder
    # shape if `final_refit_fit_shape` is not explicitly set.
    refit_fit_shape = config['psf'].get(
        'final_refit_fit_shape',
        _resolve_fit_shape(psf_sigma))
    # Pinned positions: with xy_bounds=1e-6 (the legacy emulation of
    # "fixed"), photutils still treats every source as a 3-parameter
    # nonlinear problem — the grouped LM computes finite-difference
    # Jacobians over 3N parameters per group, which is what makes
    # mega-groups in crowded fields pathologically slow, and it flags
    # ~every source with bit 32 ("parameter at bound") as a side
    # effect. Genuinely fixing x_0/y_0 on the model reduces each group
    # to an N-parameter problem that is linear in flux (LM converges
    # in ~2 iterations) and leaves flag 32 meaningful. Same optimum,
    # much cheaper.
    fix_positions = (xy_bound <= 0 and bool(
        config['psf'].get('final_refit_fix_positions', True)))
    if fix_positions:
        psf_model_refit = psf_model.copy()
        psf_model_refit.x_0.fixed = True
        psf_model_refit.y_0.fixed = True
        xy_bounds_arg = None
    else:
        # photutils v3 PSFPhotometry requires xy_bounds to be strictly
        # positive; substitute a tiny value to emulate pinning.
        psf_model_refit = psf_model
        xy_bounds_arg = max(xy_bound, 1e-6)
    psfphot = PSFPhotometry(
        psf_model_refit,
        fit_shape       = refit_fit_shape,
        finder          = None,
        grouper         = grouper,
        local_bkg_estimator = localbkg,
        aperture_radius = config['psf']['PSFPhotometry_aperture_radius'],
        fitter_maxiters = config['psf']['PSFPhotometry_fitter_maxiters'],
        xy_bounds       = xy_bounds_arg,
        group_warning_threshold = group_warn,
    )

    finder_kwargs = config['psf']['finder_kwargs']
    bkgrms_estimator = MADStdBackgroundRMS()

    new_phot = None
    new_resid = data_bksub.copy()
    prev_mad = float('inf')
    dedup_r2 = (dedup_psfsigma * psf_sigma) ** 2
    catastrophic_factor = float(config['psf'].get(
        'final_refit_catastrophic_flux_factor', 20.0))

    # Group-level memoisation across refit iterations. The grouper splits
    # the field into independent LM subproblems; a group's fit depends only
    # on its own member rows plus the (constant) image, error and mask. So
    # a group whose member set is unchanged since the last iteration
    # re-derives exactly the rows it produced before, and can be spliced in
    # from the cache instead of re-fit. Iterations after the first usually
    # only add a handful of leftover sources, touching a few groups —
    # everything else is reused. This is exact, not an approximation.
    reuse_groups = bool(config['psf'].get(
        'final_refit_reuse_unchanged_groups', True))
    group_cache = {}
    prev_sig_set = [None]
    support_shape = _psf_support_shape(psf_model)
    n_fit_total = 0

    def _grouped_fit(init_tbl):
        ''' Returns (phot, resid, status, n_fitted). status is 'ok',
        'nochange' (every group cached -> this iteration reproduces the
        previous one) or 'fail'.
        '''
        x = np.asarray(init_tbl['x'], dtype=float)
        y = np.asarray(init_tbl['y'], dtype=float)
        f = np.asarray(init_tbl['flux'], dtype=float)
        if len(x) == 0:
            return None, None, 'fail', 0
        gids = (np.asarray(grouper(x, y)) if len(x) > 1
                else np.ones(1, dtype=int))
        uniq = np.unique(gids)
        sig = {}
        for g in uniq:
            m = gids == g
            sig[int(g)] = tuple(sorted(zip(x[m].tolist(), y[m].tolist(),
                                           f[m].tolist())))
        # "Nothing to do" means the SOURCE SET is unchanged since the last
        # iteration, not merely that every group is already cached: a
        # source dropped by the catastrophic-flux filter can leave every
        # surviving group's signature intact (it was a singleton group),
        # and the output table must still be rebuilt without it.
        sig_set = frozenset(sig.values())
        if sig_set == prev_sig_set[0]:
            return None, None, 'nochange', 0
        prev_sig_set[0] = sig_set

        stale = [int(g) for g in uniq if sig[int(g)] not in group_cache]
        n_fit = 0
        if stale:
            sel = np.isin(gids, stale)
            n_fit = int(sel.sum())
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                phot_fit = psfphot(data_bksub, error=data_error,
                                   init_params=init_tbl[sel],
                                   mask=center_mask)
            if phot_fit is None or len(phot_fit) == 0:
                return None, None, 'fail', n_fit

            fx = np.asarray(phot_fit['x_init'], dtype=float)
            fy = np.asarray(phot_fit['y_init'], dtype=float)
            for g in stale:
                m = gids == g
                rows = np.zeros(len(phot_fit), dtype=bool)
                for xi, yi in zip(x[m], y[m]):
                    rows |= (fx == xi) & (fy == yi)
                group_cache[sig[g]] = phot_fit[rows]

        parts = [group_cache[sig[int(g)]] for g in uniq]
        parts = [p for p in parts if p is not None and len(p) > 0]
        if not parts:
            return None, None, 'fail', int(sel.sum())
        combined = parts[0] if len(parts) == 1 else vstack(parts)

        # restore the caller's source ordering (photutils returns rows in
        # init order for a full fit; per-group splices must match so the
        # output table is identical, not merely equivalent).
        pos_index = {}
        for i, (xi, yi) in enumerate(zip(x, y)):
            pos_index.setdefault((xi, yi), []).append(i)
        order_key = []
        for xi, yi in zip(np.asarray(combined['x_init'], dtype=float),
                          np.asarray(combined['y_init'], dtype=float)):
            lst = pos_index.get((xi, yi))
            order_key.append(lst.pop(0) if lst else len(x))
        combined = combined[np.argsort(np.asarray(order_key, dtype=int),
                                        kind='stable')]
        model_img = _render_phot_model(psf_model, combined,
                                        data_bksub.shape, support_shape)
        return combined, data_bksub - model_img, 'ok', n_fit

    if progress is not None:
        task = progress.add_task(
            f'final joint refit ({len(init)} sources)', total=n_iter)

    for it in range(n_iter):
        try:
            if reuse_groups:
                cand_phot, cand_resid, status, n_fit = _grouped_fit(init)
                n_fit_total += n_fit
                if status == 'nochange':
                    logger.debug(f'joint refit iter {it}: all groups '
                                 f'unchanged; stopping.')
                    break
                if status == 'fail':
                    raise RuntimeError('grouped refit returned no photometry')
                new_phot, new_resid = cand_phot, cand_resid
                logger.debug(f'joint refit iter {it}: fit {n_fit}/'
                             f'{len(init)} sources ('
                             f'{len(init) - n_fit} reused from cache)')
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    new_phot = psfphot(data_bksub, error=data_error,
                                       init_params=init, mask=center_mask)
                new_resid = psfphot.make_residual_image(data_bksub)
        except Exception as e:
            logger.warning(f'joint refit iter {it} failed: {e}')
            break

        if new_phot is None or len(new_phot) == 0:
            break

        # Drop catastrophic-flux fits before the next iteration. When two
        # sources sit close enough that the grouper bundles them, the LM
        # can find degenerate solutions where one has +1e7 flux and the
        # other -1e7 (they cancel locally but pollute the catalogue and
        # destabilise subsequent iterations).
        flux_arr = np.asarray(new_phot['flux_fit'], dtype=float)
        finite = np.isfinite(flux_arr)
        if finite.any():
            scale = float(np.nanmedian(np.abs(flux_arr[finite])))
            scale = max(scale, 1.0)
            keep = (np.abs(flux_arr) <= catastrophic_factor * scale) & finite
            if keep.sum() < len(flux_arr):
                logger.debug(f'joint refit iter {it}: dropping '
                             f'{int((~keep).sum())} catastrophic fits')
                init = init[keep]
                # If we just dropped, redo this iteration's fit on the
                # cleaned init list before trying to detect leftovers.
                continue

        # Detect leftover positive peaks in the residual at the current
        # noise level. Only positive => over-subtracted (negative) regions
        # don't generate spurious additions.
        try:
            resid_mad = float(bkgrms_estimator(new_resid))
        except Exception:
            resid_mad = bkg_std
        logger.debug(f'joint refit iter {it}: N_init={len(init)} '
                     f'N_out={len(new_phot)} resid_MAD={resid_mad:.10g} '
                     f'n_nonfinite_flux='
                     f'{int((~np.isfinite(np.asarray(new_phot["flux_fit"], dtype=float))).sum())}')

        if progress is not None:
            progress.update(task, description=(
                f'final joint refit (it={it+1}, N={len(init)}, '
                f'resid_MAD={resid_mad:.4f})'))
            progress.update(task, advance=1, refresh=True)

        # Stop if the residual stopped improving meaningfully.
        if prev_mad - resid_mad < resid_tol * prev_mad:
            break
        prev_mad = resid_mad

        try:
            finder = DAOStarFinder(
                threshold=detect_th * resid_mad,
                fwhm=psf_sigma * 2.33,
                **finder_kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                leftover = finder(new_resid, mask=center_mask)
        except Exception:
            leftover = None

        if leftover is None or len(leftover) == 0:
            break

        new_x = np.asarray(leftover['x_centroid'], dtype=float)
        new_y = np.asarray(leftover['y_centroid'], dtype=float)
        # `flux` from DAOStarFinder is a rough integrated estimate that's
        # good enough as an LM starting point.
        new_flux = np.asarray(leftover['flux'], dtype=float)

        # Stage 1: dedupe against existing init.
        if len(init) > 0:
            ex_x = np.asarray(init['x'])
            ex_y = np.asarray(init['y'])
            dx = new_x[:, None] - ex_x[None, :]
            dy = new_y[:, None] - ex_y[None, :]
            min_dist2 = (dx * dx + dy * dy).min(axis=1)
            is_new = min_dist2 > dedup_r2
        else:
            is_new = np.ones(len(new_x), dtype=bool)

        # Stage 2: dedupe new detections against EACH OTHER. Without this,
        # a single faint feature can produce 2-3 DAO candidates within a
        # pixel of each other, all pass the existing-init dedup, and end
        # up bundled into one group with degenerate ±huge fluxes.
        keep_idx = []
        for j in np.where(is_new)[0]:
            ok = True
            for k in keep_idx:
                if (new_x[j] - new_x[k])**2 + (new_y[j] - new_y[k])**2 <= dedup_r2:
                    ok = False
                    break
            if ok:
                keep_idx.append(int(j))

        if not keep_idx:
            break

        added = QTable()
        added['x'] = new_x[keep_idx]
        added['y'] = new_y[keep_idx]
        added['flux'] = new_flux[keep_idx]
        init = vstack([init, added])

    if progress is not None:
        progress.remove_task(task)

    if new_phot is None or len(new_phot) == 0:
        return None, None

    # ---- final non-negative flux solve ----
    # Bad-source endgame. Row-dropping approaches all failed the same way:
    # negative fluxes are a symptom of the UNCONSTRAINED flux solve (LM
    # groups can trade +/- flux between near-degenerate columns), so
    # removing the offending rows and re-fitting simply mints new ones
    # (measured: gate v1 turned a -51 into a -123 and a +577/-huge pair).
    # With positions fixed the flux problem is linear, so put the
    # constraint IN the solver: one global sparse NNLS over the final
    # source list. Negative fluxes become impossible, +/- compensating
    # pairs cannot form, and the model/residual pair is consistent by
    # construction (resid = data - A @ f from the same solve). Sources
    # NNLS zeroes out are dropped (flux 0 contributes nothing, so no
    # re-solve is needed); one DAO leftover round then gives any real
    # star that was hiding under a zeroed source its own entry.
    # 'always' = every refit call; 'final' = only when the caller flags the
    # LAST photometry pass of the main loop (kwargs nnls_now=True) -- the
    # pass whose psf_table/psf_sub_data get saved. Intermediate iPSF calls
    # then keep the cheap LM fluxes (they only feed the Sersic and the
    # calibrator), which avoids paying the NNLS + leftover cost ~11x per
    # filter. true/false accepted for back-compat.
    _mode = config['psf'].get('final_refit_nnls_fluxes', 'final')
    if _mode is True:
        _mode = 'always'
    elif _mode is False:
        _mode = 'never'
    _run_nnls = (_mode == 'always'
                 or (_mode == 'final' and bool(kwargs.get('nnls_now', False))))
    if _run_nnls:
        from scipy.optimize import nnls as _nnls

        def _nnls_solve(phot):
            x = np.asarray(phot['x_fit'], dtype=float)
            y = np.asarray(phot['y_fit'], dtype=float)
            b = data_bksub.ravel().astype(np.float32)
            invalid = ~np.isfinite(b)
            if center_mask is not None:
                invalid |= center_mask.ravel()
            b = np.where(invalid, 0.0, b)
            A = _build_design_matrix(psf_model, x, y, data_bksub.shape,
                                     _psf_support_shape(psf_model),
                                     invalid_flat=invalid)
            G = _design_gram(A)
            rhs = _design_rmatvec(A, b)
            L = np.linalg.cholesky(G + 1e-6 * np.eye(G.shape[0]))
            f, _ = _nnls(L.T, np.linalg.solve(L, rhs), maxiter=10000)
            model = _design_matvec(A, f.astype(np.float32))
            resid = data_bksub - model.reshape(data_bksub.shape)
            # flux errors from the Gram diagonal (same recipe as
            # forced_psf_photometry)
            rr = b - model
            nun = int((~invalid).sum())
            sig = (float(np.sqrt(np.sum(rr[~invalid] ** 2)
                                 / max(nun - len(x), 1)))
                   if nun > len(x) + 1 else float(bkg_std))
            try:
                gdiag = np.diag(np.linalg.pinv(G + 1e-6 * np.eye(G.shape[0])))
                ferr = np.sqrt(np.maximum(gdiag, 0.0)) * sig
            except Exception:
                ferr = np.full(len(x), sig)
            return f, ferr, resid

        try:
            for nnls_round in range(2):
                flux, flux_err, resid_n = _nnls_solve(new_phot)
                zero = flux <= 0
                new_phot = new_phot.copy()
                new_phot['flux_fit'] = flux
                new_phot['flux_err'] = flux_err
                if zero.any():
                    logger.info(f'final NNLS fluxes: {int(zero.sum())}/'
                                f'{len(flux)} source(s) zeroed and dropped')
                    new_phot = new_phot[~zero]
                new_resid = resid_n
                if nnls_round == 1:
                    break
                # one leftover round on the NNLS residual
                try:
                    resid_mad = float(bkgrms_estimator(new_resid))
                    finder = DAOStarFinder(threshold=detect_th * resid_mad,
                                           fwhm=psf_sigma * 2.33,
                                           **finder_kwargs)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        leftover = finder(new_resid, mask=center_mask)
                except Exception:
                    leftover = None
                if leftover is None or len(leftover) == 0:
                    break
                lx = np.asarray(leftover['x_centroid'], dtype=float)
                ly = np.asarray(leftover['y_centroid'], dtype=float)
                ex = np.asarray(new_phot['x_fit'], dtype=float)
                ey = np.asarray(new_phot['y_fit'], dtype=float)
                d2 = ((lx[:, None] - ex[None, :]) ** 2
                      + (ly[:, None] - ey[None, :]) ** 2)
                is_new = d2.min(axis=1) > dedup_r2
                keep_idx = []
                for j in np.where(is_new)[0]:
                    if all((lx[j] - lx[k]) ** 2 + (ly[j] - ly[k]) ** 2
                           > dedup_r2 for k in keep_idx):
                        keep_idx.append(int(j))
                if not keep_idx:
                    break
                logger.info(f'final NNLS fluxes: adding {len(keep_idx)} '
                            f'leftover source(s) and re-solving')
                add = new_phot[:1].copy()
                rows = []
                for j in keep_idx:
                    r = new_phot[:1].copy()
                    r['x_fit'] = lx[j]; r['y_fit'] = ly[j]
                    if 'x_init' in r.colnames:
                        r['x_init'] = lx[j]; r['y_init'] = ly[j]
                    r['flux_fit'] = float(leftover['flux'][j])
                    if 'flags' in r.colnames:
                        r['flags'] = 0
                    rows.append(r)
                new_phot = vstack([new_phot] + rows)
        except Exception as e:
            logger.warning(f'final NNLS flux solve failed: {e}; keeping '
                           f'the LM fluxes')

    return new_phot, new_resid


# ---------------------------------------------------------------------------
# Per-bin NNLS final refit
# ---------------------------------------------------------------------------

def _render_unit_image(psf_model, x, y, flux, shape, render_shape):
    ''' Render a sum of point-source images at the given (x, y) positions
    with the given fluxes, using a single ImagePSF model.
    '''
    if len(x) == 0:
        return np.zeros(shape)
    params = QTable()
    params['x_0'] = np.asarray(x, dtype=float)
    params['y_0'] = np.asarray(y, dtype=float)
    params['flux'] = np.asarray(flux, dtype=float)
    return _make_model_image(
        shape=shape, model=psf_model,
        params_table=params,
        model_shape=tuple(render_shape),
        x_name='x_0', y_name='y_0',
    )


def _build_design_matrix(psf_model, x, y, shape, render_shape,
                          invalid_flat=None):
    ''' Design matrix for the NNLS solves: column `i` is a unit-flux PSF
    rendered at `(x[i], y[i])`, flattened.

    Each column is rendered with `modelimg_render_shape`, so at most
    `prod(render_shape)` of its `H*W` entries are nonzero — ~1% of the
    matrix on a typical crowded cutout. Storing it densely costs
    `H*W*N*4` bytes (141 MB for 576 sources on a 247x247 cutout) and,
    far worse, makes the Gram product `A.T@A` cost `N^2 * H*W`
    multiply-adds (~2e10) almost all of which multiply zeros. `G_ij` is
    just the overlap integral of stamps i and j and is structurally zero
    unless the two sources are within a stamp of each other, so the
    sparse product does ~4e6 real operations instead.

    Values come from the same `_render_unit_image` call the dense path
    used, so the two agree exactly (up to summation order in the Gram).

    Returns a `scipy.sparse.csc_matrix` when `[psf].sparse_nnls` is on,
    otherwise the dense ndarray.
    '''
    H, W = shape
    n = int(len(x))
    use_sparse = bool(config['psf'].get('sparse_nnls', True))
    if not use_sparse:
        cols = np.zeros((H * W, n), dtype=np.float32)
        for i in range(n):
            single = _render_unit_image(psf_model, [x[i]], [y[i]], [1.0],
                                        (H, W), render_shape)
            cols[:, i] = single.ravel().astype(np.float32)
        if invalid_flat is not None and invalid_flat.any():
            cols[invalid_flat, :] = 0.0
        return cols

    from scipy import sparse
    # Render each column directly on its own stamp instead of rendering a
    # full H*W image and then hunting for the nonzeros: same values
    # (verified bit-identical against `_render_unit_image`), ~12x faster,
    # and it avoids allocating an H*W array per source.
    ny, nx = int(render_shape[0]), int(render_shape[1])
    hy, hx = ny // 2, nx // 2
    has_invalid = invalid_flat is not None and invalid_flat.any()
    rows_list, vals_list = [], []
    indptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        xi = float(x[i]); yi = float(y[i])
        iy = int(round(yi)); ix = int(round(xi))
        y0 = max(iy - hy, 0); y1 = min(iy - hy + ny, H)
        x0 = max(ix - hx, 0); x1 = min(ix - hx + nx, W)
        if y0 >= y1 or x0 >= x1 or not (np.isfinite(xi) and np.isfinite(yi)):
            indptr[i + 1] = indptr[i]
            continue
        gy, gx = np.mgrid[y0:y1, x0:x1]
        vals = psf_model.evaluate(gx.astype(float), gy.astype(float),
                                  flux=1.0, x_0=xi, y_0=yi)
        vals = np.where(np.isfinite(vals), vals, 0.0).ravel()
        idx = (gy.ravel() * W + gx.ravel()).astype(np.int64)
        if has_invalid:
            vals = np.where(invalid_flat[idx], 0.0, vals)
        keep = vals != 0.0
        rows_list.append(idx[keep].astype(np.int32))
        vals_list.append(vals[keep].astype(np.float32))
        indptr[i + 1] = indptr[i] + int(keep.sum())
    if n == 0:
        return sparse.csc_matrix((H * W, 0), dtype=np.float32)
    rows = (np.concatenate(rows_list) if rows_list
            else np.zeros(0, dtype=np.int32))
    vals = (np.concatenate(vals_list) if vals_list
            else np.zeros(0, dtype=np.float32))
    return sparse.csc_matrix((vals, rows, indptr),
                              shape=(H * W, n), dtype=np.float32)


def _design_gram(A):
    ''' `A.T @ A` as a small dense float64 array (N x N).

    Accumulated in float64 for the sparse path: the Gram is where the
    dense float32 version loses precision (it sums H*W float32 products
    per entry), and with sparsity the float64 product is cheap enough
    that there is no reason to keep the float32 accumulation.
    '''
    from scipy import sparse
    if sparse.issparse(A):
        A64 = A.astype(np.float64)
        return np.asarray((A64.T @ A64).toarray(), dtype=np.float64)
    G = A.T @ A
    return np.asarray(G, dtype=np.float64)


def _design_rmatvec(A, b):
    ''' `A.T @ b` as a dense float64 vector. '''
    return np.asarray(A.T @ b).ravel().astype(np.float64)


def _design_matvec(A, f):
    ''' `A @ f` as a dense vector. '''
    return np.asarray(A @ f).ravel()


def _design_hstack(A, B):
    ''' Append columns, preserving sparsity. '''
    from scipy import sparse
    if sparse.issparse(A) or sparse.issparse(B):
        return sparse.hstack([A, B], format='csc')
    return np.concatenate([A, B], axis=1)


def _design_drop_last(A, k):
    ''' Drop the last `k` columns. '''
    return A[:, :A.shape[1] - k]


def _design_dense(A):
    ''' Densify for the rare direct-NNLS fallback path. '''
    from scipy import sparse
    return A.toarray() if sparse.issparse(A) else A


def _local_residual_mad(residual, x, y, half=8):
    ''' Median over the listed sources of the MAD of a (2*half+1)^2 cutout
    centred on each. Used as the per-bin blur calibration score.
    '''
    vals = []
    for xi, yi in zip(x, y):
        ix = int(round(float(xi))); iy = int(round(float(yi)))
        sy = slice(max(0, iy - half), iy + half + 1)
        sx = slice(max(0, ix - half), ix + half + 1)
        cut = residual[sy, sx]
        cut = cut[np.isfinite(cut)]
        if cut.size == 0:
            continue
        vals.append(float(np.median(np.abs(cut - np.median(cut)))))
    return float(np.median(vals)) if vals else float('inf')


def _final_nnls_refit(data_bksub, data_error, bkg_std,
                      psf_model, psf_sigma, phot_result,
                      progress=None, **kwargs):
    ''' Final refit by global non-negative least squares.

    Takes the ladder's quality-passing sources (positions pinned),
    builds a single design matrix where each column is a unit-flux
    rendering of `psf_model` at one source's position, and solves
    for fluxes via NNLS in the Cholesky-Gram form. The solver
    inherently produces flux >= 0, so catastrophic compensating-pair
    LM solutions cannot occur.

    The PSF used is exactly `psf_model` — sphot's per-call blur
    calibration in PSFFitter has already picked the best blur for the
    field, and downstream `calibrate_psf_step` (when enabled) further
    refines the kernel. There is no additional flux-dependent
    broadening here.

    After the initial solve, an iterative leftover-detection loop
    (`leftover_max_iterations`) runs `find_peaks` on the residual to
    catch sources DAOStarFinder missed inside the threshold ladder.
    Detected leftovers get rendered with the same `psf_model` and
    appended to the design matrix; NNLS re-solves over the augmented
    set.

    Returns (new_phot_result, new_residual_image) on success, or
    (None, None) if no quality-passing sources exist or the NNLS
    solve fails.
    '''
    from scipy.optimize import nnls

    if phot_result is None or len(phot_result) == 0:
        return None, None

    # 1. quality filter
    center_mask_params = kwargs.get('center_mask_params', None)
    s_pass, _ = filter_psfphot_results(
        phot_result, center_mask_params=center_mask_params,
        bkg_std=bkg_std)
    if int(s_pass.sum()) == 0:
        return None, None

    x_all = np.asarray(phot_result['x_fit'][s_pass], dtype=float)
    y_all = np.asarray(phot_result['y_fit'][s_pass], dtype=float)
    n_total = len(x_all)

    render_shape = tuple(config['psf']['modelimg_render_shape'])

    if progress is not None:
        task = progress.add_task(
            f'final NNLS refit (N={n_total})', total=2)

    # 2. build NNLS design matrix (one column per source, all rendered
    #    with the same psf_model)
    H, W = data_bksub.shape
    b = data_bksub.ravel().astype(np.float32)
    finite = np.isfinite(b)
    if (~finite).any():
        b = np.where(finite, b, 0.0)
    cols = _build_design_matrix(psf_model, x_all, y_all, (H, W),
                                render_shape, invalid_flat=~finite)
    if progress is not None:
        progress.update(task, advance=1, refresh=True)

    # NNLS via Gram + Cholesky (much faster than the tall M x N system)
    G = _design_gram(cols)
    rhs = _design_rmatvec(cols, b)
    if not (np.all(np.isfinite(G)) and np.all(np.isfinite(rhs))):
        logger.warning('NNLS refit: Gram/rhs has non-finite entries; '
                       'aborting refit')
        if progress is not None:
            progress.remove_task(task)
        return None, None
    try:
        L = np.linalg.cholesky(G + 1e-6 * np.eye(G.shape[0]))
        Linv_rhs = np.linalg.solve(L, rhs)
        flux_fit, _ = nnls(L.T, Linv_rhs, maxiter=10000)
    except (np.linalg.LinAlgError, RuntimeError):
        try:
            flux_fit, _ = nnls(_design_dense(cols).astype(np.float64),
                               b.astype(np.float64), maxiter=20000)
        except Exception as e:
            logger.warning(f'NNLS solve failed: {e}')
            if progress is not None:
                progress.remove_task(task)
            return None, None

    # 3. build model + residual
    model = _design_matvec(cols, flux_fit.astype(np.float32)).reshape(H, W)
    new_resid = data_bksub - model

    # 4. iterative leftover detection (find_peaks on the residual +
    #    NNLS re-solve over the augmented source list). DAO's smoothing
    #    kernel inside the ladder washes compact <FWHM blobs below
    #    threshold on a residual image; find_peaks is kernel-free and
    #    catches them. See tests/psf_experiments/proto_12_detection.
    leftover_max_iter = int(config['psf'].get('leftover_max_iterations', 0))
    leftover_x_added = np.empty(0, dtype=float)
    leftover_y_added = np.empty(0, dtype=float)
    n_leftover_added = 0
    if leftover_max_iter > 0:
        from photutils.detection import find_peaks
        from photutils.centroids import centroid_com
        from photutils.background import MADStdBackgroundRMS
        k_sigma = float(config['psf'].get(
            'leftover_detect_k_sigma', 5.0))
        dedup_factor = float(config['psf'].get(
            'leftover_dedup_psfsigma', 1.5))
        box_size = int(config['psf'].get('leftover_box_size', 3))
        center_mask_leftover = _build_center_mask(
            new_resid.shape, kwargs.get('center_mask_params'))
        x_acc = x_all.copy()
        y_acc = y_all.copy()
        for it in range(leftover_max_iter):
            resid_for_det = np.nan_to_num(new_resid, nan=0.0)
            try:
                thr = k_sigma * float(MADStdBackgroundRMS()(resid_for_det))
                peaks = find_peaks(resid_for_det, threshold=thr,
                                   box_size=box_size,
                                   mask=center_mask_leftover,
                                   centroid_func=centroid_com)
            except Exception as e:
                logger.warning(f'leftover detection iter {it}: '
                               f'find_peaks failed: {e}')
                break
            if peaks is None or len(peaks) == 0:
                break
            lx = np.asarray(peaks['x_centroid'], dtype=float)
            ly = np.asarray(peaks['y_centroid'], dtype=float)
            bad = ~np.isfinite(lx) | ~np.isfinite(ly)
            if bad.any():
                lx[bad] = np.asarray(peaks['x_peak'], dtype=float)[bad]
                ly[bad] = np.asarray(peaks['y_peak'], dtype=float)[bad]
            r2 = (dedup_factor * psf_sigma) ** 2
            keep = np.ones(len(lx), dtype=bool)
            for i in range(len(lx)):
                d2 = (x_acc - lx[i]) ** 2 + (y_acc - ly[i]) ** 2
                if d2.size and d2.min() <= r2:
                    keep[i] = False
            lx, ly = lx[keep], ly[keep]
            if len(lx) == 0:
                break
            n_new = len(lx)
            new_cols = _build_design_matrix(
                psf_model, lx, ly, (H, W), render_shape,
                invalid_flat=~finite)
            cols = _design_hstack(cols, new_cols)
            G = _design_gram(cols)
            rhs = _design_rmatvec(cols, b)
            if not (np.all(np.isfinite(G)) and np.all(np.isfinite(rhs))):
                logger.warning(f'leftover NNLS iter {it}: non-finite '
                               f'Gram/rhs; stopping')
                cols = _design_drop_last(cols, n_new)
                break
            try:
                L = np.linalg.cholesky(G + 1e-6 * np.eye(G.shape[0]))
                Linv_rhs = np.linalg.solve(L, rhs)
                flux_fit, _ = nnls(L.T, Linv_rhs, maxiter=10000)
            except (np.linalg.LinAlgError, RuntimeError):
                try:
                    flux_fit, _ = nnls(_design_dense(cols).astype(np.float64),
                                       b.astype(np.float64), maxiter=20000)
                except Exception as e:
                    logger.warning(f'leftover NNLS iter {it} solve '
                                   f'failed: {e}')
                    cols = _design_drop_last(cols, n_new)
                    break
            x_acc = np.concatenate([x_acc, lx])
            y_acc = np.concatenate([y_acc, ly])
            leftover_x_added = np.concatenate([leftover_x_added, lx])
            leftover_y_added = np.concatenate([leftover_y_added, ly])
            n_leftover_added += n_new
            model = _design_matvec(
                cols, flux_fit.astype(np.float32)).reshape(H, W)
            new_resid = data_bksub - model

    if progress is not None:
        progress.update(task, advance=1, refresh=True)
        progress.remove_task(task)

    # 5. assemble phot_result with refit fluxes (existing rows + leftovers)
    new_phot = phot_result[s_pass].copy()
    new_phot['flux_fit'] = flux_fit[:n_total]

    if n_leftover_added > 0 and len(new_phot) > 0:
        templates = []
        for i in range(n_leftover_added):
            row = new_phot[:1].copy()
            try:
                row['x_fit'] = leftover_x_added[i]
                row['y_fit'] = leftover_y_added[i]
                if 'x_init' in row.colnames:
                    row['x_init'] = leftover_x_added[i]
                if 'y_init' in row.colnames:
                    row['y_init'] = leftover_y_added[i]
                row['flux_fit'] = flux_fit[n_total + i]
                if 'flags' in row.colnames:
                    row['flags'] = 0
            except Exception as e:
                logger.warning(f'failed to construct leftover row {i}: {e}')
                continue
            templates.append(row)
        if templates:
            new_phot = vstack([new_phot] + templates)

    msg = f'NNLS refit: {n_total} sources'
    if n_leftover_added > 0:
        msg += f'; +{n_leftover_added} leftovers added by find_peaks'
    logger.info(msg)
    return new_phot, new_resid