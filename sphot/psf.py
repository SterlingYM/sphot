import numpy as np
import matplotlib.pyplot as plt
import warnings
import traceback

from astropy.utils.exceptions import AstropyUserWarning
from astropy.table import QTable, vstack
from astropy.stats import SigmaClip, sigma_clipped_stats

from photutils.aperture import EllipticalAperture
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

class PSFFitter():
    ''' A class to perform PSF fitting. '''
    def __init__(self,cutoutdata):
        self.cutoutdata = cutoutdata
        self.psf_sigma = cutoutdata.psf_sigma

        # Blur-calibration short-circuit: count consecutive fit() calls where
        # the calibration picked the current_blur unchanged. After this
        # reaches `psf_blur_stable_after`, subsequent fit() calls skip the
        # calibration step entirely until the cutoutdata.psf_blurring is
        # externally changed.
        self._blur_stable_count = 0

        # Adaptive calibration step (in pixels). Lazily initialised on first
        # calibration call from config['psf']['psf_blur_initial_delta'].
        # Grows when the minimum lands at an edge of the bracket (we were
        # too narrow); shrinks when we converge to the interior.
        self._blur_delta = None

        # Trajectory of blur values across this fitter's calibrations;
        # rendered live in the rich progress display (one task per fitter).
        self._blur_trajectory = []
        self._blur_task_id = None

            # PSF image to model
        self.psf_model = self.psf_img2model(cutoutdata.psf,cutoutdata.psf_oversample)
        
    def psf_img2model(self,psfimg,psf_oversample):
        psf_model = ImagePSF(
            psfimg, flux=1.0,
            x_0=0, y_0=0, 
            oversampling=psf_oversample, 
            fill_value=0.0
            )
        return psf_model
    
    def update_psf_blur(self,psf_blur):
        ''' Update the PSF model by convolving the PSF image with a Gaussian kernel.
        
        Args:
            psf_blur (float): the sigma of the Gaussian kernel in pixel units.
        '''
        if psf_blur:
            logger.debug(f'Convolving (blurring) PSF with sigma={psf_blur} pix')
            psf_blurred, psf_sigma = self.cutoutdata.blur_psf(psf_blur)
            self.psf_model = self.psf_img2model(psf_blurred,self.cutoutdata.psf_oversample)
            self.psf_sigma = psf_sigma
        else:
            logger.debug(f'Using original PSF without blurring')
            self.psf_model = self.psf_img2model(self.cutoutdata.psf,self.cutoutdata.psf_oversample)
            self.psf_sigma = self.cutoutdata.psf_sigma

        return self.psf_model,self.psf_sigma

    def _current_blur_estimate(self):
        ''' Return the best blur estimate so far.

        Prefers the live cutoutdata.psf_blurring (updated by prior fits) over
        the config default so the calibration iteratively refines across
        main-loop iterations. Falls back to the config value on the first
        call, then to False (= no blurring).
        '''
        curr = getattr(self.cutoutdata, 'psf_blurring', None)
        if curr:
            return curr
        blur_psf_dict = config['prep'].get('blur_psf', False)
        if isinstance(blur_psf_dict, dict):
            return blur_psf_dict.get(self.cutoutdata.filtername, False)
        return blur_psf_dict

    def _render_blur_trajectory(self):
        ''' Render the recorded blur trajectory as a rich-markup string.

        The longest run of equal trailing values (length >= 2) is rendered
        in green to indicate convergence; earlier values are plain.
        '''
        history = self._blur_trajectory
        if not history:
            return ''
        last = history[-1]
        # Count the trailing run of values equal to `last`.
        run = 1
        for v in reversed(history[:-1]):
            if abs(v - last) < 1e-6:
                run += 1
            else:
                break
        converged = run >= 2
        parts = []
        for i, v in enumerate(history):
            in_tail = (len(history) - i) <= run
            if converged and in_tail:
                parts.append(f'[green]{v:.2f}[/green]')
            else:
                parts.append(f'{v:.2f}')
        return ' [dim]→[/dim] '.join(parts)

    def _update_blur_display(self, progress):
        ''' Push the current trajectory into the live display.

        Preferred path: write to `progress.blur_panel`, a persistent
        renderable that core.showprogress places BELOW the progress block
        so calibration summaries always sit at the bottom of the live
        view, regardless of which transient progress tasks are active.

        Fallbacks (in order): a display-only progress task (when no
        BlurPanel is attached -- e.g. caller passed their own Progress);
        plain logger.info (when no progress object exists at all).
        '''
        rendered = self._render_blur_trajectory()
        desc = f'[cyan]blur[/cyan] [{self.cutoutdata.filtername}]: {rendered}'
        blur_panel = getattr(progress, 'blur_panel', None) if progress else None
        if blur_panel is not None:
            blur_panel.update(self.cutoutdata.filtername, desc)
        elif progress is not None:
            if self._blur_task_id is None:
                self._blur_task_id = progress.add_task(
                    desc, total=1, completed=1)
            else:
                progress.update(self._blur_task_id,
                                description=desc, completed=1)
        else:
            plain = (rendered
                     .replace('[green]', '').replace('[/green]', '')
                     .replace('[dim]', '').replace('[/dim]', ''))
            logger.info(f'blur calibration [{self.cutoutdata.filtername}]: '
                        f'{plain}')

    def _posterior_calibrate_blur(self, current_blur, phot_result, progress=None):
        ''' Posterior blur calibration with adaptive additive step and
        parabolic interpolation.

        Evaluates the (cfit+qfit) metric at a 3-point bracket
        [current - Δ, current, current + Δ], where Δ is an adaptive step
        tracked across fit() calls. The winner drives the blur for the
        NEXT iteration's ladder.

        Winner selection:
          * If the grid minimum is a significant improvement over the
            current score (> improvement_tol), try parabolic interpolation
            through the 3 grid points and take the vertex; else use the
            grid minimum. If the parabola is not convex or the vertex
            lands outside the bracket, fall back to the grid minimum.
          * If the grid improvement is below the tolerance, stay at
            current (prevents noise-driven oscillation).

        Step adaptation (user invariant: "keep Δ larger than the jump you
        just took, so the next bracket doesn't need to extrapolate"):
          * jump = |new_blur - current|
          * Δ_next = max(2 * jump, 0.75 * Δ)   (safety factor 2 for
            extrapolation avoidance; floor at 75% so Δ doesn't collapse
            when jump=0)
          * Δ_next is clipped to [psf_blur_min_delta, psf_blur_max_delta].
          * When the grid minimum lands at an edge (jump = Δ), Δ grows;
            when it lands interior (|jump| < Δ), Δ shrinks.
        '''
        if phot_result is None or len(phot_result) == 0:
            return current_blur

        # Seed the trajectory with the starting blur on the first call so
        # the rich display reads "3.80 -> 3.30 -> ..." rather than starting
        # at the first jump.
        if not self._blur_trajectory:
            self._blur_trajectory.append(float(current_blur))

        # --- source selection (top K passing sources) --------------------
        x0 = self.cutoutdata.sersic_params_physical['x_0']
        y0 = self.cutoutdata.sersic_params_physical['y_0']
        center_mask_params = [x0, y0, self.psf_sigma * 2]
        try:
            _, bkg_std, _ = subtract_background(self.data)
        except Exception:
            bkg_std = float(np.nanstd(self.data))
        s_pass, _ = filter_psfphot_results(
            phot_result,
            center_mask_params=center_mask_params,
            bkg_std=bkg_std)
        passing = phot_result[s_pass]
        min_sources = config['psf'].get('psf_blur_calib_min_sources', 3)
        if len(passing) < min_sources:
            logger.debug(f'blur calibration [{self.cutoutdata.filtername}]: '
                         f'only {len(passing)} passing sources '
                         f'(< {min_sources}); skipping')
            return current_blur

        K = min(config['psf'].get('psf_blur_calib_K', 30), len(passing))
        order = np.argsort(passing['flux_fit'])[::-1][:K]
        top = passing[order]
        init_params = QTable()
        init_params['x'] = top['x_fit']
        init_params['y'] = top['y_fit']
        init_params['flux'] = top['flux_fit']

        # --- build additive 4-point symmetric bracket --------------------
        # Points: [current - Δ, current - Δ/3, current + Δ/3, current + Δ].
        # Four samples instead of three give an over-determined quadratic
        # fit (np.polyfit deg=2) that averages out per-pixel noise in the
        # cfit/qfit metric.
        delta_min = float(config['psf'].get('psf_blur_min_delta', 0.05))
        delta_max = float(config['psf'].get('psf_blur_max_delta', 2.0))
        if self._blur_delta is None:
            self._blur_delta = float(
                config['psf'].get('psf_blur_initial_delta', 0.3))
        delta = float(np.clip(self._blur_delta, delta_min, delta_max))

        step = (2.0 * delta) / 3.0
        bracket = [current_blur - delta + i * step for i in range(4)]
        # Clip the low end (current_blur near zero) and dedupe.
        bracket = [max(delta_min, x) for x in bracket]
        bracket = sorted(set(round(x, 6) for x in bracket))

        # --- score each bracket point ------------------------------------
        scores = {}
        for candidate in bracket:
            psf_model, _psf_sigma = self.update_psf_blur(float(candidate))
            try:
                psfphot = PSFPhotometry(
                    psf_model,
                    fit_shape = config['psf']['PSFPhotometry_fit_shape'],
                    finder = None,
                    aperture_radius = config['psf']['PSFPhotometry_aperture_radius'],
                    fitter_maxiters = config['psf']['PSFPhotometry_fitter_maxiters'],
                )
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    new_result = psfphot(self.data, init_params=init_params)
            except Exception as e:
                logger.debug(f'blur calibration [{self.cutoutdata.filtername}]: '
                             f'candidate {candidate:.2f} failed: {e}')
                scores[float(candidate)] = np.inf
                continue
            if new_result is None or len(new_result) == 0:
                scores[float(candidate)] = np.inf
                continue
            cfit_med = float(np.nanmedian(np.abs(new_result['cfit'])))
            qfit_med = float(np.nanmedian(new_result['qfit']))
            scores[float(candidate)] = cfit_med + qfit_med

        # --- fit a quadratic + decide the new blur ----------------------
        improvement_tol = config['psf'].get('psf_blur_improvement_tol', 0.05)
        items = sorted(scores.items())
        xs = np.array([x for x, _ in items], dtype=float)
        ys = np.array([y for _, y in items], dtype=float)
        finite = np.isfinite(ys)
        xs, ys = xs[finite], ys[finite]

        coeffs, vertex = self._quadratic_fit(xs, ys)
        if coeffs is not None:
            score_at_current = float(np.polyval(coeffs, current_blur))
        elif len(ys) > 0:
            # Fall back to closest grid point's score if quadratic fit failed.
            closest = int(np.argmin(np.abs(xs - current_blur)))
            score_at_current = float(ys[closest])
        else:
            score_at_current = np.inf

        if len(ys) > 0:
            grid_best_score = float(np.min(ys))
            grid_best_blur = float(xs[int(np.argmin(ys))])
        else:
            grid_best_score = np.inf
            grid_best_blur = float(current_blur)

        if grid_best_score < score_at_current * (1 - improvement_tol):
            # The measured grid minimum beats current by the tolerance --
            # commit to the parabolic vertex if available, else grid min.
            new_blur = vertex if vertex is not None else grid_best_blur
        else:
            new_blur = float(current_blur)

        # --- adapt delta for the NEXT call -------------------------------
        jump = abs(new_blur - current_blur)
        if jump > 0:
            # Ensure next bracket still contains the new point without
            # extrapolation: delta_next must exceed jump. Factor of 2 is
            # a reasonable safety margin.
            next_delta = max(2.0 * jump, 0.75 * delta)
        else:
            # No movement: gently shrink so we can detect a smaller optimum
            # next time, but never collapse to delta_min immediately.
            next_delta = 0.75 * delta
        self._blur_delta = float(np.clip(next_delta, delta_min, delta_max))

        # Restore PSF state to the chosen blur for downstream code.
        self.update_psf_blur(float(new_blur))

        # Record the trajectory and refresh the live display.
        self._blur_trajectory.append(float(new_blur))
        self._update_blur_display(progress)
        return new_blur

    @staticmethod
    def _quadratic_fit(xs, ys):
        ''' Fit y = a x^2 + b x + c through `(xs, ys)` and return
        (coeffs, vertex_x) where coeffs is the np.polyfit result
        ``[a, b, c]`` and vertex_x = -b/(2a) clipped to within the
        bracket. Either may be None:

        * coeffs is None when fewer than 3 finite points are available.
        * vertex_x is None when the quadratic isn't strictly convex
          (a <= 0) or the vertex falls outside [xs.min(), xs.max()].

        With 3 points polyfit returns the unique exact parabola; with
        4+ points it returns the least-squares fit, averaging out noise
        in the score metric.
        '''
        if len(xs) < 3:
            return None, None
        coeffs = np.polyfit(xs, ys, 2)
        a, b, _c = coeffs
        if a <= 0:
            return coeffs, None  # fit is fine, but no convex minimum
        vertex = float(-b / (2.0 * a))
        if vertex < xs.min() or vertex > xs.max():
            return coeffs, None
        return coeffs, vertex

    def fit(self,fit_to='sersic_residual',**kwargs):
        ''' Perform PSF fitting.
        This function calls iterative_psf_fitting, which wraps our main
        function do_psf_photometry. The role of iterative_psf_fitting is to
        change the detection threshold level so that the PSF fitter does
        not end up fitting >1000 sources at the same time in a highly
        crowded field.

        After the main threshold ladder completes, a cheap posterior
        calibration step uses the brightest quality-passing sources to
        evaluate candidate blurs (factor * current_blur) by median cfit
        and qfit of a refit with finder=None. The winner updates
        cutoutdata.psf_blurring and drives the NEXT iteration's blur.

        Args:
            fit_to (str): the data to fit the PSF to. An attribute of this name needs to exist. A few examples:
                - 'sersic_residual': the residual image after sersic fitting (default)
                - 'residual': the residual image after PSF fitting.
                - 'data': the original data.
            kwargs (dict): additional kwargs to pass to do_psf_photometry.

        Returns:
            cutoutdata (CutoutData): the updated cutoutdata object. Updates are applied in-place, so users don't need to grab this for typical use cases.
        '''
        self.data = getattr(self.cutoutdata,fit_to)

        # Apply the current best blur estimate (from a prior fit() or from
        # the config default). The ladder runs at this blur; the posterior
        # step refines the estimate for the NEXT iteration.
        current_blur = self._current_blur_estimate()
        if current_blur:
            self.update_psf_blur(current_blur)
            self.cutoutdata.psf_blurring = current_blur
        else:
            logger.debug('Using original PSF without blurring')

        x0 = self.cutoutdata.sersic_params_physical['x_0']
        y0 = self.cutoutdata.sersic_params_physical['y_0']
        center_mask_params = [x0,y0,self.psf_sigma*2]

        # Threshold ladder at the current blur.
        th_min = config['psf'].get('th_min',1.5)
        th_max = config['psf'].get('th_max',4.0)
        th_iter = config['psf'].get('th_iter',10)
        threshold_list = np.geomspace(th_min, th_max, num=th_iter)[::-1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psf_table, resid = iterative_psf_fitting(
                self.data,
                self.psf_model,
                self.psf_sigma,
                threshold_list = threshold_list,
                center_mask_params=center_mask_params,
                **kwargs)

        # Posterior blur calibration: drives the blur for the NEXT iteration.
        if current_blur and psf_table is not None:
            stable_after = config['psf'].get('psf_blur_stable_after', 2)
            if self._blur_stable_count >= stable_after:
                logger.debug(f'blur calibration [{self.cutoutdata.filtername}]: '
                             f'stable at {current_blur:.2f}; skipping')
            else:
                new_blur = self._posterior_calibrate_blur(
                    current_blur, psf_table, progress=kwargs.get('progress'))
                if np.isclose(new_blur, current_blur, rtol=1e-6):
                    self._blur_stable_count += 1
                else:
                    self._blur_stable_count = 0
                self.cutoutdata.psf_blurring = new_blur
            
        psf_model_total = self.data - resid
        psf_model_total -= np.nanmin(psf_model_total) # PSFs are forced to be positive, so minimum is always zero
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
    b = (1 - ellip) * a
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

def _prepare_psf_fitters(th,psf_model,bkg_std,psf_sigma):
    finder_kwargs = config['psf']['finder_kwargs']
    daofinder = DAOStarFinder(
        threshold=th*bkg_std,
        fwhm=psf_sigma*2.33, **finder_kwargs)

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
        fit_shape          = config['psf']['PSFPhotometry_fit_shape'],
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
                      **kwargs):
    """ Performs PSF photometry. Main function to run PSF photometry.
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
            phot_result = psf_single(data, error=data_error) # init_params=init_params)
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
        plt.show()
        
    return phot_result, resid

def iterative_psf_fitting(data,psf_model,psf_sigma,
                          threshold_list,
                          progress=None,
                          progress_text='Running iPSF...',
                          **kwargs):
    ''' Iteratively run do_psf_photometry() with different threshold levels.
    This function is useful for crowded fields, where a single threshold level may fail.

    The threshold list is typically descending (high -> low). Each successful
    pass subtracts detected sources from the residual; the next pass then
    operates on the cleaner residual. The ladder early-exits when
    `th_max_consec_empty` consecutive thresholds add no new sources.

    Inputs:
        data (2d array): the data to perform PSF photometry.
        psf_model: the PSF ImagePSF model.
        psf_sigma (float): the HWHM of the PSF. Use FWHM/2
        threshold_list (1d array): the list of threshold levels to try, in background STD.
        center_mask_params (list, optional): [x_center,y_center,mask_r]. If provided, sources within the radius mask_r from (x_center,y_center) will be excluded from the final results. This is useful when the central source is very bright and causes many spurious detections nearby.
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
        logger.info(f'PSF background stats failed ({str(e)})')
        return None,None,None,None
          
    # initialize variables
    resid = data_bksub.copy()
    phot_result = None
    last_successful_th = None
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

    # loop -- repeat PSF subtraction
    if progress is not None:
        progress_psf = progress.add_task(progress_text,
                                        total=len(threshold_list))
    for th in threshold_list:
        added_sources = False
        try:
            psf_results = do_psf_photometry(resid, data_error, bkg_std,
                                            psf_model, psf_sigma,
                                            th=th,**kwargs)
            if progress is not None:
                progress.update(progress_psf, advance=1, refresh=True)
            if psf_results[0] is None:
                pass  # no detection -> empty pass
            else:
                _phot_result, _resid = psf_results
                if not np.all(~np.isfinite(_resid)) and len(_phot_result) > 0:
                    # append the results
                    resid = _resid
                    if phot_result is None:
                        phot_result = _phot_result
                    else:
                        phot_result = vstack([phot_result, _phot_result])
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

    # Final refit step. Two modes are available, picked by
    # `final_refit_method`:
    #   'iterative'    -- _final_joint_refit: iterative LM refit + leftover
    #                     detection. Honours the legacy `final_joint_refit`
    #                     bool for back-compat (false => skip the refit).
    #   'perbin_nnls'  -- _final_perbin_nnls_refit: bin sources by flux,
    #                     calibrate a per-bin extra blur, build a global
    #                     non-negative least-squares system and solve once.
    #                     Deterministic, no negative-flux outputs, runs in
    #                     a few seconds.
    #   'none'         -- skip the final refit entirely; ladder output is
    #                     returned as-is.
    method = str(config['psf'].get('final_refit_method', 'iterative')).lower()
    refit_func = None
    if phot_result is not None and len(phot_result) > 0:
        if method == 'perbin_nnls':
            refit_func = _final_perbin_nnls_refit
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


def _final_joint_refit(data_bksub, data_error, bkg_std,
                       psf_model, psf_sigma, phot_result,
                       progress=None, **kwargs):
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
    s_pass, _ = filter_psfphot_results(
        phot_result, center_mask_params=center_mask_params,
        bkg_std=bkg_std)
    if int(s_pass.sum()) == 0:
        return None, None

    init = QTable()
    init['x'] = np.asarray(phot_result['x_fit'][s_pass], dtype=float)
    init['y'] = np.asarray(phot_result['y_fit'][s_pass], dtype=float)
    init['flux'] = np.asarray(phot_result['flux_fit'][s_pass], dtype=float)

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

    # photutils v3 PSFPhotometry requires xy_bounds to be strictly positive
    # (it does not accept 0 or None as "fixed"). Substitute a tiny positive
    # value when the user has asked for pinned positions.
    # The ladder uses a small fit_shape ([3,3] by default) so that crowded
    # groups don't blow up; the joint refit can afford a wider window
    # because positions are pinned and the LM only has to solve for fluxes.
    refit_fit_shape = config['psf'].get(
        'final_refit_fit_shape',
        config['psf']['PSFPhotometry_fit_shape'])
    psfphot = PSFPhotometry(
        psf_model,
        fit_shape       = refit_fit_shape,
        finder          = None,
        grouper         = grouper,
        local_bkg_estimator = localbkg,
        aperture_radius = config['psf']['PSFPhotometry_aperture_radius'],
        fitter_maxiters = config['psf']['PSFPhotometry_fitter_maxiters'],
        xy_bounds       = max(xy_bound, 1e-6),
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

    if progress is not None:
        task = progress.add_task(
            f'final joint refit ({len(init)} sources)', total=n_iter)

    for it in range(n_iter):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                new_phot = psfphot(data_bksub, error=data_error,
                                   init_params=init)
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
                leftover = finder(new_resid)
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


def _final_perbin_nnls_refit(data_bksub, data_error, bkg_std,
                             psf_model, psf_sigma, phot_result,
                             progress=None, **kwargs):
    ''' Final refit by per-bin extra-blur + non-negative least squares.

    Origin: tests/psf_experiments/proto_08_brightness_dependent_blur.

    Idea
    ----
    1. Take the ladder's quality-passing sources as the input list
       (positions pinned).
    2. Bin sources by initial flux (default: 3 bins from
       `perbin_flux_percentiles`).
    3. For each bin, render OTHER bins' contribution with the central
       PSF, then scan a small set of "extra blur" sigmas (in oversampled
       PSF px units) on top of the current PSF and pick the one that
       minimises the median local residual MAD over that bin's sources.
    4. Build a global non-negative least-squares system in which each
       column is a unit-flux PSF rendered at one source's position with
       its bin's chosen extra-blur. Solve with `scipy.optimize.nnls` via
       the Cholesky-Gram form (much faster than the full tall system).
    5. The solver inherently produces flux >= 0, so the catastrophic
       compensating-pair LM solutions that plagued the iterative refit
       cannot occur.

    The "extra blur" parametrisation only ADDS blur on top of the
    current `psf_model`. That's intentional: the existing per-call
    blur calibration in PSFFitter has already chosen the best single
    blur (which usually fits the dim end well); extra blur for the
    bright bin lets bright stars use a slightly wider effective PSF
    where saturation broadening or scattered-light wings exceed what
    the global blur captures.

    Returns (new_phot_result, new_residual_image) on success, or
    (None, None) if no quality-passing sources exist or if the NNLS
    solve fails.
    '''
    from astropy.convolution import convolve, Gaussian2DKernel
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
    f_all = np.asarray(phot_result['flux_fit'][s_pass], dtype=float)
    n_total = len(x_all)

    # 2. bin by flux percentile (portable across datasets)
    pct = list(config['psf'].get('perbin_flux_percentiles', [33.0, 67.0]))
    pos_flux = f_all[f_all > 0]
    if len(pos_flux) >= len(pct) + 1:
        edges = np.percentile(pos_flux, pct)
    else:
        edges = np.array([])  # 1 bin only
    bin_idx = np.zeros(n_total, dtype=int)
    for e in edges:
        bin_idx += (f_all > e).astype(int)
    n_bins = len(edges) + 1

    # 3. extract base PSF (already at the best blur from PSFFitter
    # calibration) and oversample factor from the model
    base_psf_img = np.asarray(psf_model.data, dtype=float)
    psf_oversample = int(np.atleast_1d(psf_model.oversampling)[0])

    # candidate extra-blur sigmas (oversampled-PSF px)
    extra_blurs = list(config['psf'].get(
        'perbin_extra_blurs', [0.0, 1.0, 2.0, 3.0, 4.0]))

    # render shape (data px), reuse the same as the threshold ladder
    render_shape = tuple(config['psf']['modelimg_render_shape'])

    if progress is not None:
        task = progress.add_task(
            f'final per-bin NNLS refit (N={n_total}, bins={n_bins})',
            total=n_bins + 2)

    # build candidate PSF models keyed by extra_sigma
    psf_models_by_blur = {}
    for s in extra_blurs:
        s = float(s)
        if s == 0.0:
            img = base_psf_img.copy()
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                img = convolve(base_psf_img, Gaussian2DKernel(s))
        if img.sum() > 0:
            img = img / float(img.sum())
        psf_models_by_blur[s] = ImagePSF(img, flux=1.0,
                                         oversampling=psf_oversample,
                                         fill_value=0.0)

    # 4. per-bin blur calibration
    bin_best_extra = {}
    for bi in range(n_bins):
        sel = (bin_idx == bi)
        n = int(sel.sum())
        if n < 2:
            bin_best_extra[bi] = 0.0
            if progress is not None:
                progress.update(task, advance=1, refresh=True)
            continue
        # other bins rendered once with the central (extra=0) PSF
        other_sel = ~sel
        other_img = _render_unit_image(
            psf_models_by_blur[0.0],
            x_all[other_sel], y_all[other_sel], f_all[other_sel],
            data_bksub.shape, render_shape)
        data_minus_other = data_bksub - other_img
        scores = {}
        for s in extra_blurs:
            this_img = _render_unit_image(
                psf_models_by_blur[float(s)],
                x_all[sel], y_all[sel], f_all[sel],
                data_bksub.shape, render_shape)
            scores[float(s)] = _local_residual_mad(
                data_minus_other - this_img,
                x_all[sel], y_all[sel], half=8)
        bin_best_extra[bi] = min(scores, key=scores.get)
        if progress is not None:
            progress.update(task, advance=1, refresh=True)

    # 5. build NNLS design matrix (one column per source)
    H, W = data_bksub.shape
    cols = np.zeros((H * W, n_total), dtype=np.float32)
    for i in range(n_total):
        s = bin_best_extra[bin_idx[i]]
        single = _render_unit_image(
            psf_models_by_blur[s], [x_all[i]], [y_all[i]], [1.0],
            data_bksub.shape, render_shape)
        cols[:, i] = single.ravel().astype(np.float32)
    if progress is not None:
        progress.update(task, advance=1, refresh=True)

    # NNLS via Gram + Cholesky (much faster than the full M x N system)
    b = data_bksub.ravel().astype(np.float32)
    finite = np.isfinite(b)
    if (~finite).any():
        b = np.where(finite, b, 0.0)
        cols[~finite, :] = 0.0
    G = (cols.T @ cols).astype(np.float64)
    rhs = (cols.T @ b).astype(np.float64)
    if not (np.all(np.isfinite(G)) and np.all(np.isfinite(rhs))):
        logger.warning('per-bin NNLS: Gram/rhs has non-finite entries; '
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
            flux_fit, _ = nnls(cols.astype(np.float64),
                               b.astype(np.float64), maxiter=20000)
        except Exception as e:
            logger.warning(f'per-bin NNLS solve failed: {e}')
            if progress is not None:
                progress.remove_task(task)
            return None, None

    # 6. build final model image with refit fluxes per bin
    model = np.zeros_like(data_bksub, dtype=float)
    for bi in range(n_bins):
        sel = (bin_idx == bi)
        if int(sel.sum()) == 0:
            continue
        s = bin_best_extra[bi]
        sub = _render_unit_image(
            psf_models_by_blur[s], x_all[sel], y_all[sel], flux_fit[sel],
            data_bksub.shape, render_shape)
        model += sub
    new_resid = data_bksub - model

    if progress is not None:
        progress.update(task, advance=1, refresh=True)
        progress.remove_task(task)

    # 7. produce a phot_result with the refit fluxes; copy the original
    # quality-passing rows so downstream code (save, aperture phot) sees
    # all the columns it expects.
    new_phot = phot_result[s_pass].copy()
    new_phot['flux_fit'] = flux_fit
    # flag negative-flux entries as 0 since NNLS guarantees >= 0;
    # preserve other columns (cfit, qfit, flags, ...) as-is from the
    # ladder fit -- they describe the LM solve, which is the only
    # quality info we have for these sources.

    bin_best_str = ', '.join(f'b{bi}:Δσ={bin_best_extra[bi]:.1f}'
                             for bi in range(n_bins))
    logger.info(f'per-bin NNLS refit: {n_total} sources in {n_bins} bins '
                f'({bin_best_str})')
    return new_phot, new_resid