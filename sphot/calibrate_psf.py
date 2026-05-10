"""Per-iteration PSF calibration: sphot's mainloop hook.

`calibrate_psf_step` is called per Sersic+iPSF mainloop iteration via
`sphot.core._maybe_recalibrate_psf` when `[psf-calib] in_mainloop = true`.
Each call:

  1. Pulls quality-passing anchors from `cutoutdata.psf_table`. With
     no usable photometry the bootstrap blur grid picks a starting PSF
     from source counts.
  2. Fits a kernel `K` (Gaussian / Moffat / drizzle) against the
     anchors via the multi-source NNLS objective.
  3. Replaces `cutoutdata.psf` with `library_psf * K`.
  4. Scans `dao_fwhm_factor` on a 3-point bracket and picks the factor
     that maximises quality-passing source count under the new PSF.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np


# =====================================================================
# Section 1 — utility
# =====================================================================

def _normalize_psf(img):
    img = np.asarray(img, dtype=float)
    s = img.sum()
    if s != 0 and np.isfinite(s):
        img = img / s
    return img


def _kernel_data_to_oversampled(K_data, oversample, target_oversampled_shape):
    """Up-sample a data-px kernel to oversampled-PSF resolution and
    pad/crop to target shape. Result sums to 1 (preserves PSF flux
    after convolution).
    """
    from scipy.ndimage import zoom
    K_oversampled = zoom(K_data, oversample, order=3, mode='reflect')
    s = K_oversampled.sum()
    if s > 0:
        K_oversampled = K_oversampled / s
    h_t, w_t = target_oversampled_shape
    h_n, w_n = K_oversampled.shape
    if h_n > h_t or w_n > w_t:
        yy0 = (h_n - h_t) // 2
        xx0 = (w_n - w_t) // 2
        K_oversampled = K_oversampled[yy0:yy0 + h_t, xx0:xx0 + w_t]
    elif h_n < h_t or w_n < w_t:
        canvas = np.zeros((h_t, w_t), dtype=float)
        y0 = (h_t - h_n) // 2
        x0 = (w_t - w_n) // 2
        canvas[y0:y0 + h_n, x0:x0 + w_n] = K_oversampled
        K_oversampled = canvas
    s = K_oversampled.sum()
    if s > 0:
        K_oversampled = K_oversampled / s
    return K_oversampled


def _apply_kernel(library_psf_oversampled, K_oversampled):
    """library * K, normalised. Returns library_psf size."""
    from scipy.signal import fftconvolve
    out = fftconvolve(library_psf_oversampled, K_oversampled, mode='same')
    out[out < 0] = 0
    s = out.sum()
    if s > 0:
        out = out / s
    return out


def _crop_centered(img, half):
    """Return the centered `(2*half+1)` square crop of `img`.

    If the image is already smaller than the requested size, return as-is.
    Used to shrink the library PSF for the kernel-fit's inner FFT and
    spline interpolation; the full library is preserved for the final
    `_build_effective_psf` call that updates `cd.psf`.
    """
    H, W = img.shape
    cy = (H - 1) // 2
    cx = (W - 1) // 2
    half = int(min(half, cy, cx))
    return img[cy - half : cy + half + 1, cx - half : cx + half + 1]


def _empirical_fwhm_factor(psf_image, oversample, psf_sigma,
                            *, fit_half_pix=20, fwhm_init=None):
    """Compute the canonical `dao_fwhm_factor` from the calibrated PSF.

    Runs `photutils.psf.fit_fwhm` on the central core of an oversampled PSF
    image, converts the fitted FWHM to data pixels, and returns
    `F_eff_in_data_px / psf_sigma`. This is the matched-filter-theorem
    answer for DAO's FWHM argument and replaces the bracket scan when
    `[psf-calib].fwhm_method == 'empirical'`.

    Returns `None` if the fit fails or yields a non-finite value.
    """
    try:
        from photutils.psf import fit_fwhm
    except Exception as e:
        logger.warning(f'_empirical_fwhm_factor: photutils.psf.fit_fwhm '
                       f'unavailable ({e})')
        return None

    psf = np.asarray(psf_image, dtype=float)
    psf = np.where(np.isfinite(psf), psf, 0.0)
    H, W = psf.shape
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    fit_half = int(max(1, fit_half_pix * int(oversample)))
    fit_shape = (2 * fit_half + 1, 2 * fit_half + 1)
    init = float(fwhm_init) if fwhm_init is not None else 4.0 * float(oversample)
    try:
        fwhms = fit_fwhm(psf, xypos=[(cx, cy)], fit_shape=fit_shape, fwhm=init)
        fwhm_oversampled = float(np.atleast_1d(fwhms)[0])
    except Exception as e:
        logger.warning(f'_empirical_fwhm_factor: fit_fwhm failed ({e})')
        return None
    if not np.isfinite(fwhm_oversampled) or fwhm_oversampled <= 0:
        return None
    fwhm_data_px = fwhm_oversampled / float(oversample)
    if not np.isfinite(psf_sigma) or psf_sigma <= 0:
        return None
    return fwhm_data_px / float(psf_sigma)


def effective_fwhm_data_px(cutoutdata):
    """FWHM of `cd.psf` in data pixels (post-kernel calibration).

    Used for sizing the centre-mask radius and any other "where is the
    PSF actually wide" calculation. Falls back to the library Gaussian
    equivalent (2.355 × psf_sigma) if `photutils.psf.fit_fwhm` fails so
    the caller always gets a finite, sane number.
    """
    psf_sigma = float(getattr(cutoutdata, 'psf_sigma', 0.0))
    fallback = 2.3548 * psf_sigma if psf_sigma > 0 else 0.0
    psf_image = getattr(cutoutdata, 'psf', None)
    oversample = int(getattr(cutoutdata, 'psf_oversample', 1) or 1)
    if psf_image is None or not np.isfinite(psf_sigma) or psf_sigma <= 0:
        return fallback
    factor = _empirical_fwhm_factor(psf_image, oversample, psf_sigma)
    if factor is None or not np.isfinite(factor) or factor <= 0:
        return fallback
    return float(factor) * psf_sigma


# =====================================================================
# Section 2 — kernel families
# =====================================================================

def _gaussian_kernel(shape, sigma):
    sh = shape[0]
    cy = (sh - 1) / 2.0
    yy, xx = np.indices(shape)
    rr = (xx - cy) ** 2 + (yy - cy) ** 2
    if sigma <= 0:
        out = np.zeros(shape)
        out[int(round(cy)), int(round(cy))] = 1.0
        return out
    g = np.exp(-rr / (2.0 * sigma ** 2))
    return g / g.sum()


def _moffat_kernel(shape, alpha, beta):
    sh = shape[0]
    cy = (sh - 1) / 2.0
    yy, xx = np.indices(shape)
    rr = np.sqrt((xx - cy) ** 2 + (yy - cy) ** 2)
    if alpha <= 0 or beta <= 1.01:
        return _gaussian_kernel(shape, max(alpha, 0.1))
    m = (1.0 + (rr / alpha) ** 2) ** (-beta)
    return m / m.sum()


def _drizzle_kernel(shape, sigma, pixel_size=1.0):
    """Drizzle pixel-square (top-hat) ⊗ Gaussian. Models drizzling's
    pixel-response broadening + an optical Gaussian.
    """
    sh = shape[0]
    cy = (sh - 1) / 2.0
    yy, xx = np.indices(shape)
    in_pix = (np.abs(xx - cy) <= 0.5 * pixel_size) & \
             (np.abs(yy - cy) <= 0.5 * pixel_size)
    tophat = in_pix.astype(float)
    if tophat.sum() == 0:
        tophat[int(round(cy)), int(round(cy))] = 1.0
    tophat = tophat / tophat.sum()
    if sigma <= 0:
        return tophat
    g = _gaussian_kernel(shape, sigma)
    from scipy.signal import fftconvolve
    out = fftconvolve(tophat, g, mode='same')
    out[out < 0] = 0
    out = out / out.sum()
    return out


def _build_kernel(family, params, shape):
    if family == 'gaussian':
        return _gaussian_kernel(shape, params['sigma_data'])
    if family == 'moffat':
        return _moffat_kernel(shape, params['alpha_data'], params['beta'])
    if family == 'drizzle':
        return _drizzle_kernel(shape, params['sigma_data'],
                                params.get('pixel_size', 1.0))
    raise ValueError(f'unknown kernel family: {family}')


def _build_effective_psf(library_norm, oversample, family, params, half=20):
    sh = 2 * half + 1
    K_data = _build_kernel(family, params, (sh, sh))
    K_oversampled = _kernel_data_to_oversampled(
        K_data, oversample, library_norm.shape)
    return _apply_kernel(library_norm, K_oversampled)


def _make_psf_evaluator(eff_psf_oversampled, oversample):
    """Fast PSF evaluator using `scipy.ndimage.map_coordinates`.

    Bypasses photutils ImagePSF's `RectBivariateSpline` (which dominated
    the kernel-fit profile at ~44%). Returns a callable
    `evaluate(xx, yy, x_0, y_0) -> array` giving interpolated PSF
    values at data positions (xx, yy) for a source centered at (x_0, y_0).

    The output scale is arbitrary (not normalised to data-pixel flux) —
    this is fine for the kernel fit because the per-anchor LM solves
    for flux, so the residual is invariant to a constant scaling of m.

    The spline pre-filter is computed ONCE here. `map_coordinates`
    re-spline-filters the full input on every call when
    `prefilter=True` (the default), which would dominate the profile;
    we pass the pre-filtered image with `prefilter=False`.
    """
    from scipy.ndimage import map_coordinates, spline_filter
    H, W = eff_psf_oversampled.shape
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    img = np.ascontiguousarray(eff_psf_oversampled, dtype=float)
    img_filtered = spline_filter(img, order=3, mode='constant')

    def evaluate(xx, yy, x_0, y_0):
        x_over = (xx - x_0) * oversample + cx
        y_over = (yy - y_0) * oversample + cy
        coords = np.stack([y_over.ravel(), x_over.ravel()], axis=0)
        vals = map_coordinates(img_filtered, coords, order=3,
                                mode='constant', cval=0.0,
                                prefilter=False)
        return vals.reshape(xx.shape)
    return evaluate


def _per_source_residual_score_multi(
    K_data, library_psf_oversampled, psf_oversample,
    data, x_anchor, y_anchor, half=8,
    *,
    all_x, all_y, all_flux=None,
    max_neighbors=15, neighbor_pad=4.0,
):
    """Per-anchor stamp residual with a **multi-source** closed-form
    least-squares flux fit (positions pinned to iPSF centroids).

    Counters the σ₂ runaway in crowded fields with wide PSFs: when a
    stamp contains the anchor plus blended neighbours, the
    single-source LM in `_per_source_residual_score` is forced to
    explain the neighbours' flux via the model's wings — driving
    `σ₂` to the upper bound. Here every source within
    `half + neighbor_pad` data px gets its own column in the
    design matrix, so neighbour flux is accounted for and the kernel
    only has to match the *shape* of each source's PSF.

    Parameters
    ----------
    all_x, all_y : 1D arrays
        Positions (data px) of ALL quality-passing sources from
        `cd.psf_table`, used to populate per-stamp neighbour lists.
    all_flux : 1D array, optional
        Flux per source; used to keep the brightest neighbours when
        a stamp's neighbour count exceeds `max_neighbors`.
    """
    K_oversampled = _kernel_data_to_oversampled(
        K_data, psf_oversample, library_psf_oversampled.shape)
    eff_psf = _apply_kernel(library_psf_oversampled, K_oversampled)
    evaluate_psf = _make_psf_evaluator(eff_psf, psf_oversample)
    H, W = data.shape
    yy0, xx0 = np.indices((2 * half + 1, 2 * half + 1), dtype=float)
    radius = float(half) + float(neighbor_pad)
    radius2 = radius * radius
    all_x = np.asarray(all_x, dtype=float)
    all_y = np.asarray(all_y, dtype=float)
    if all_flux is not None:
        all_flux = np.asarray(all_flux, dtype=float)

    total = 0.0
    n_pix = 0
    for x_a, y_a in zip(x_anchor, y_anchor):
        ix = int(round(x_a)); iy = int(round(y_a))
        if (ix - half < 0 or iy - half < 0
                or ix + half + 1 > W or iy + half + 1 > H):
            continue
        sy = slice(iy - half, iy + half + 1)
        sx = slice(ix - half, ix + half + 1)
        stamp = np.asarray(data[sy, sx], dtype=float)
        finite = np.isfinite(stamp)
        if not finite.any():
            continue
        if not finite.all():
            stamp = np.where(finite, stamp, 0.0)
        xx = xx0 + (ix - half)
        yy = yy0 + (iy - half)

        d2 = (all_x - x_a) ** 2 + (all_y - y_a) ** 2
        in_range = d2 <= radius2
        nx = all_x[in_range]; ny = all_y[in_range]
        if len(nx) > max_neighbors:
            if all_flux is not None:
                order = np.argsort(-all_flux[in_range])[:max_neighbors]
            else:
                order = np.argsort(d2[in_range])[:max_neighbors]
            nx = nx[order]; ny = ny[order]
        n_n = len(nx)
        if n_n == 0:
            # Anchor itself fell outside the search radius? Shouldn't
            # happen, but be defensive: fall back to anchor-only.
            nx = np.array([x_a]); ny = np.array([y_a]); n_n = 1

        # Design matrix: one column per source + 1 pedestal column.
        L = np.empty((stamp.size, n_n + 1), dtype=float)
        for j in range(n_n):
            mj = evaluate_psf(xx, yy, nx[j], ny[j])
            mj = np.where(np.isfinite(mj), mj, 0.0)
            L[:, j] = mj.ravel()
        L[:, -1] = 1.0

        finite_flat = finite.ravel().astype(float)
        L_w = L * finite_flat[:, None]
        b_w = stamp.ravel() * finite_flat

        try:
            params, *_ = np.linalg.lstsq(L_w, b_w, rcond=None)
            resid = b_w - L_w @ params
        except Exception:
            # Numerically degenerate (e.g., colocated sources); skip.
            resid = b_w
        total += float(np.sum(resid * resid))
        n_pix += int(finite.sum())
    return total / max(n_pix, 1)


def _per_source_residual_score(
    K_data, library_psf_oversampled, psf_oversample,
    data, x_arr, y_arr, flux_arr, half=8,
):
    """Per-source residual L2 with a per-anchor (x, y, f, c) PSF refit.

    For each anchor we run a 4-parameter Levenberg-Marquardt fit of
    (dx, dy, flux, pedestal) against the stamp, with the candidate
    kernel-blurred PSF as the model. The input `flux_arr`, `x_arr`,
    `y_arr` from the prior iteration's photometry are used only as
    initial guesses — the LM resolves the centroid/flux bias that the
    too-sharp prior PSF introduces, so the kernel optimizer sees a
    fair "shape match" score independent of the prior's flux scale.
    """
    from scipy.optimize import least_squares
    K_oversampled = _kernel_data_to_oversampled(
        K_data, psf_oversample, library_psf_oversampled.shape)
    eff_psf = _apply_kernel(library_psf_oversampled, K_oversampled)
    evaluate_psf = _make_psf_evaluator(eff_psf, psf_oversample)
    H, W = data.shape
    yy0, xx0 = np.indices((2 * half + 1, 2 * half + 1), dtype=float)
    total = 0.0
    n_pix = 0
    xy_bound = 4.0
    max_nfev = 10
    for x_init, y_init, f_init in zip(x_arr, y_arr, flux_arr):
        x0 = float(x_init); y0 = float(y_init); f0 = float(f_init)
        ix = int(round(x0)); iy = int(round(y0))
        if (ix - half < 0 or iy - half < 0
                or ix + half + 1 > W or iy + half + 1 > H):
            continue
        sy = slice(iy - half, iy + half + 1)
        sx = slice(ix - half, ix + half + 1)
        stamp = np.asarray(data[sy, sx], dtype=float)
        finite = np.isfinite(stamp)
        if not finite.any():
            continue
        if not finite.all():
            stamp = np.where(finite, stamp, 0.0)
        xx = xx0 + (ix - half)
        yy = yy0 + (iy - half)
        # Closed-form (f, c) at the initial position seeds the LM with
        # an unbiased flux estimate; LM then refines (dx, dy) and
        # tweaks (f, c). Without this seed the LM starts at f=f_init
        # which is biased low and can take many iterations to recover.
        m0 = evaluate_psf(xx, yy, x0, y0)
        m0 = np.where(np.isfinite(m0), m0, 0.0)
        N = float(stamp.size)
        Sm = float(m0.sum()); Smm = float(np.sum(m0 * m0))
        Sd = float(stamp.sum()); Sdm = float(np.sum(stamp * m0))
        det = Smm * N - Sm * Sm
        if det > 0:
            f_seed = (N * Sdm - Sm * Sd) / det
            c_seed = (Smm * Sd - Sm * Sdm) / det
            if f_seed <= 0:
                f_seed = max(f0, 0.0); c_seed = 0.0
        else:
            f_seed = max(f0, 0.0); c_seed = 0.0
        finite_flat = finite.ravel()

        def residuals(p):
            dx, dy, fl, c = p
            m = evaluate_psf(xx, yy, x0 + dx, y0 + dy)
            m = np.where(np.isfinite(m), m, 0.0)
            return ((stamp - fl * m - c).ravel()) * finite_flat

        try:
            res = least_squares(
                residuals,
                [0.0, 0.0, float(f_seed), float(c_seed)],
                bounds=([-xy_bound, -xy_bound, 0.0, -np.inf],
                        [ xy_bound,  xy_bound, np.inf,  np.inf]),
                method='trf', max_nfev=max_nfev)
            r = res.fun
        except Exception:
            # LM failure on a single anchor mustn't kill the whole
            # kernel evaluation; fall back to the closed-form score.
            r = ((stamp - f_seed * m0 - c_seed).ravel()) * finite_flat
        total += float(np.sum(r * r))
        n_pix += int(finite.sum())
    return total / max(n_pix, 1)


# =====================================================================
# Section 3 — kernel fitters (per-source-residual objective)
# =====================================================================

def _fit_kernel_to_data(
    library_psf_oversampled, psf_oversample, data,
    x_arr, y_arr, flux_arr, family, half=8,
    *, on_seed_done=None, objective='multi_source',
    all_x=None, all_y=None, all_flux=None,
    max_neighbors=15, neighbor_pad=4.0,
):
    """Fit a single-parameter kernel `family` to per-source residual.

    `objective`: 'multi_source' (closed-form NNLS lstsq with each
    nearby psf_table source as its own column; positions pinned) or
    'single_source' (legacy per-anchor LM with free dx, dy, flux, c).
    The multi_source path needs `all_x`, `all_y` (and optionally
    `all_flux`).

    `on_seed_done` is a zero-arg callback fired after the (single)
    minimisation completes — used to advance an outer progress bar.
    """
    from scipy.optimize import minimize, minimize_scalar
    sh = 2 * max(half, 10) + 1
    kshape = (sh, sh)

    if objective == 'multi_source' and all_x is not None and all_y is not None:
        def loss(K_data):
            return _per_source_residual_score_multi(
                K_data, library_psf_oversampled, psf_oversample,
                data, x_arr, y_arr, half=half,
                all_x=all_x, all_y=all_y, all_flux=all_flux,
                max_neighbors=max_neighbors, neighbor_pad=neighbor_pad)
    else:
        def loss(K_data):
            return _per_source_residual_score(
                K_data, library_psf_oversampled, psf_oversample,
                data, x_arr, y_arr, flux_arr, half=half)

    def _tick():
        if on_seed_done is not None:
            try: on_seed_done()
            except Exception: pass

    if family == 'gaussian':
        res = minimize_scalar(
            lambda s: loss(_gaussian_kernel(kshape, s)),
            bounds=(0.01, 5.0), method='bounded')
        K = _gaussian_kernel(kshape, res.x)
        _tick()
        return {'family': family,
                'params': {'sigma_data': float(res.x)},
                'kernel_data': K, 'residual_l2': float(res.fun),
                'n_seeds': 1}

    if family == 'moffat':
        res = minimize(
            lambda x: loss(_moffat_kernel(kshape, x[0], x[1])),
            np.array([1.5, 3.0]), method='L-BFGS-B',
            bounds=[(0.3, 6.0), (1.5, 10.0)])
        K = _moffat_kernel(kshape, res.x[0], res.x[1])
        _tick()
        return {'family': family,
                'params': {'alpha_data': float(res.x[0]),
                           'beta': float(res.x[1])},
                'kernel_data': K, 'residual_l2': float(res.fun),
                'n_seeds': 1}

    if family == 'drizzle':
        res = minimize_scalar(
            lambda s: loss(_drizzle_kernel(kshape, s)),
            bounds=(0.01, 4.0), method='bounded')
        K = _drizzle_kernel(kshape, res.x)
        _tick()
        return {'family': family,
                'params': {'sigma_data': float(res.x), 'pixel_size': 1.0},
                'kernel_data': K, 'residual_l2': float(res.fun),
                'n_seeds': 1}

    raise ValueError(f'unknown kernel family: {family!r} '
                     f'(supported: gaussian, moffat, drizzle)')


# =====================================================================
# Section 4 — DAO fwhm scan via temporary _prepare_psf_fitters patch
# =====================================================================

def _single_pass_phot(data, psf_oversampled, oversample, psf_sigma,
                       th=None, fwhm_factor=None, mask=None):
    """One photometry pass at threshold `th` (default config th_min).

    Optionally monkey-patches `_prepare_psf_fitters` so DAO uses
    `psf_sigma * fwhm_factor` as the matched-filter width instead of
    sphot's hard-coded 2.33.

    Bootstrap and fwhm scans only need to RANK candidates by
    n_passing; full ladder iteration would be ~10x slower for the
    same ranking signal.

    `mask` (optional 2D bool array, True = exclude) is forwarded to
    `do_psf_photometry` so DAO + LM both skip the masked region —
    used to keep the under-modelled Sersic core out of bootstrap and
    scan candidate evaluation.
    """
    from photutils.psf import ImagePSF, PSFPhotometry, SourceGrouper
    from photutils.detection import DAOStarFinder
    from photutils.background import MMMBackground, LocalBackground, MADStdBackgroundRMS
    from . import psf as sp
    from .psf import do_psf_photometry
    from .config import config

    psf_model = ImagePSF(psf_oversampled, flux=1.0,
                         oversampling=oversample, fill_value=0.0)
    if th is None:
        th = float(config['psf'].get('th_min', 1.0))
    bkg_std = float(MADStdBackgroundRMS()(np.nan_to_num(data, nan=0.0)))
    data_error = np.full_like(data, bkg_std, dtype=float)

    original = sp._prepare_psf_fitters
    if fwhm_factor is not None:
        def patched(th_, pmod, bkg_std_, ps):
            finder_kwargs = config['psf']['finder_kwargs']
            daofinder = DAOStarFinder(
                threshold=th_ * bkg_std_,
                fwhm=ps * fwhm_factor,
                **finder_kwargs)
            grouper = SourceGrouper(
                min_separation=config['psf']['grouper_separation_in_psfsigma'] * ps)
            localbkg = LocalBackground(
                config['psf']['localbkg_bounds_in_psfsigma'][0] * ps,
                config['psf']['localbkg_bounds_in_psfsigma'][1] * ps,
                MMMBackground())
            from .psf import _resolve_fit_shape
            return PSFPhotometry(
                pmod,
                fit_shape=_resolve_fit_shape(ps),
                finder=daofinder, grouper=grouper,
                local_bkg_estimator=localbkg,
                aperture_radius=config['psf']['PSFPhotometry_aperture_radius'],
                fitter_maxiters=config['psf']['PSFPhotometry_fitter_maxiters'],
                group_warning_threshold=config['psf']['PSFPhotometry_group_warning_threshold'],
            )
        sp._prepare_psf_fitters = patched
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            phot, _ = do_psf_photometry(
                data, data_error, bkg_std,
                psf_model, psf_sigma, th=th, plot=False, mask=mask)
    finally:
        sp._prepare_psf_fitters = original
    return phot, bkg_std


def _scan_dao_fwhm_factor(
    data, psf_oversampled, oversample, psf_sigma, fwhm_factor_grid,
    log=print, on_progress=None, mask=None,
):
    """Return (best_factor, list_of_records). Best = max n_passing.
    Uses single-pass photometry per candidate (~10x faster than
    iterating the full threshold ladder); the relative ranking is
    preserved, which is all the scan needs.

    `on_progress` is an optional zero-arg callback invoked once per
    grid candidate (success or failure) so an outer progress bar can
    advance phase-by-phase.
    """
    from .psf import filter_psfphot_results
    results = []
    for fac in fwhm_factor_grid:
        try:
            phot, bkg_std = _single_pass_phot(
                data, psf_oversampled, oversample, psf_sigma,
                fwhm_factor=fac, mask=mask)
        except Exception as e:
            log(f'  [fwhm_scan] fac={fac:.2f}: error {e}')
            if on_progress is not None:
                try: on_progress()
                except Exception: pass
            continue
        if phot is None or len(phot) == 0:
            log(f'  [fwhm_scan] fac={fac:.2f}: no detections')
            if on_progress is not None:
                try: on_progress()
                except Exception: pass
            continue
        try:
            s_pass, _ = filter_psfphot_results(phot, bkg_std=bkg_std)
            n_pass = int(s_pass.sum())
        except Exception:
            n_pass = int(len(phot))
        n_det = int(len(phot))
        results.append({
            'fwhm_factor': float(fac),
            'dao_fwhm_data_px': float(psf_sigma * fac),
            'n_det': n_det, 'n_pass': n_pass,
        })
        log(f'  [fwhm_scan] fac={fac:.2f} (fwhm={psf_sigma*fac:.2f}px): '
            f'n_det={n_det}, n_pass={n_pass}')
        if on_progress is not None:
            try: on_progress()
            except Exception: pass
    if not results:
        return None, []
    best = max(results, key=lambda r: r['n_pass'])
    return best['fwhm_factor'], results


def _scan_dao_fwhm_factor_nnls(
    data, psf_oversampled, oversample, psf_sigma, fwhm_factor_grid,
    *, metric='sum_snr', top_k=100, th=None,
    log=print, on_progress=None, mask=None,
):
    """NNLS-based variant of `_scan_dao_fwhm_factor`.

    For each fwhm_factor candidate: DAOStarFinder finds positions, optional
    top-K cap by DAO peak, then `forced_psf_photometry` (closed-form NNLS)
    fits fluxes at those fixed positions in one solve. Ranking is done on
    a quantitative metric of fit quality, not detection count, so the
    top-K cap composes legitimately (no saturation bias).

    metric  : 'sum_snr' (max), 'resid_mad' (min), or 'count' (max)
    top_k   : 0 / None disables the cap
    th      : DAO threshold in bkg_std units; defaults to config['psf']['th_min']
    """
    from photutils.psf import ImagePSF
    from photutils.detection import DAOStarFinder
    from photutils.background import MADStdBackgroundRMS
    from .psf import forced_psf_photometry
    from .config import config

    psf_model = ImagePSF(psf_oversampled, flux=1.0,
                         oversampling=oversample, fill_value=0.0)
    if th is None:
        th = float(config['psf'].get('th_min', 1.0))
    bkg_std = float(MADStdBackgroundRMS()(np.nan_to_num(data, nan=0.0)))
    finder_kwargs = config['psf']['finder_kwargs']
    snr_min = float(config['psf'].get('cuts_flux_SNR_min', 1.0))

    results = []
    for fac in fwhm_factor_grid:
        try:
            daofinder = DAOStarFinder(
                threshold=th * bkg_std,
                fwhm=psf_sigma * fac,
                **finder_kwargs)
            data_for_dao = np.where(np.isfinite(data), data, 0.0)
            star_cat = daofinder.find_stars(data_for_dao, mask=mask)
        except Exception as e:
            log(f'  [nnls_scan] fac={fac:.2f}: DAO error {e}')
            if on_progress is not None:
                try: on_progress()
                except Exception: pass
            continue
        if star_cat is None or len(star_cat) == 0:
            log(f'  [nnls_scan] fac={fac:.2f}: no detections')
            if on_progress is not None:
                try: on_progress()
                except Exception: pass
            continue

        x_init = np.asarray(star_cat['xcentroid'], dtype=float)
        y_init = np.asarray(star_cat['ycentroid'], dtype=float)
        peak = np.asarray(star_cat['peak'], dtype=float)

        # honour the explicit center mask (DAO may still report sources
        # whose centroids round to a masked pixel)
        if mask is not None:
            ix = np.clip(np.round(x_init).astype(int), 0, mask.shape[1] - 1)
            iy = np.clip(np.round(y_init).astype(int), 0, mask.shape[0] - 1)
            keep = ~mask[iy, ix]
            x_init = x_init[keep]
            y_init = y_init[keep]
            peak = peak[keep]
        if len(x_init) == 0:
            results.append({'fwhm_factor': float(fac), 'n_det': 0, 'n_pass': 0,
                            'sum_snr': 0.0, 'resid_mad': np.inf})
            continue

        if top_k and len(x_init) > top_k:
            order = np.argsort(-peak)[:int(top_k)]
            x_init = x_init[order]
            y_init = y_init[order]

        phot, resid = forced_psf_photometry(
            data, psf_model, psf_sigma, x_init, y_init,
            center_mask_params=None,
        )

        sum_snr, resid_mad, n_pass = 0.0, np.inf, 0
        if phot is not None and len(phot) > 0:
            flux = np.asarray(phot['flux_fit'], dtype=float)
            ferr = np.asarray(phot['flux_err'], dtype=float)
            ferr = np.where(ferr > 0, ferr, np.inf)
            snr = flux / ferr
            valid = np.isfinite(snr) & (snr > 0)
            if valid.any():
                sum_snr = float(snr[valid].sum())
                n_pass = int(((flux > 0) & (snr >= snr_min) & valid).sum())
        if resid is not None:
            r = resid.ravel()
            r = r[np.isfinite(r)]
            if r.size > 0:
                resid_mad = float(np.median(np.abs(r - np.median(r))))

        results.append({
            'fwhm_factor': float(fac),
            'n_det': int(len(x_init)), 'n_pass': n_pass,
            'sum_snr': sum_snr, 'resid_mad': resid_mad,
        })
        log(f'  [nnls_scan top_k={top_k} m={metric}] '
            f'fac={fac:.2f}: n_det={int(len(x_init))} '
            f'sum_snr={sum_snr:.1f} resid_mad={resid_mad:.3e}')
        if on_progress is not None:
            try: on_progress()
            except Exception: pass

    if not results:
        return None, []
    if metric == 'resid_mad':
        best = min(results, key=lambda r: r['resid_mad'])
    elif metric == 'count':
        best = max(results, key=lambda r: r['n_pass'])
    else:  # 'sum_snr' default
        best = max(results, key=lambda r: r['sum_snr'])
    return best['fwhm_factor'], results


# =====================================================================
# Section 5 — per-iter helper: bootstrap blur
# =====================================================================

def _bootstrap_blur(
    cutoutdata,
    *,
    initial_grid=(0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0),
    expand_step=3.0,
    max_blur=25.0,
    log=print,
    on_progress=None,
    mask=None,
):
    """Find a sensible Gaussian blur from scratch when iPSF returned
    nothing usable. Scans a grid of blurs (oversampled-PSF px) using
    single-pass photometry; picks the blur with the most
    quality-passing sources. If no candidate yields any quality-passing
    sources, expands the grid by `expand_step` up to `max_blur` and
    retries.

    On success, calls `cutoutdata.blur_psf(best)` so cd.psf is updated
    in place. Returns the chosen blur, or None if all attempts fail.
    """
    from .psf import filter_psfphot_results
    from photutils.background import MADStdBackgroundRMS
    from astropy.convolution import convolve as ap_convolve, Gaussian2DKernel

    library = np.asarray(getattr(cutoutdata, '_psf_raw', cutoutdata.psf),
                         dtype=float)
    library = _normalize_psf(library)
    psf_oversample = int(cutoutdata.psf_oversample)
    psf_sigma = float(cutoutdata.psf_sigma)
    data = np.asarray(cutoutdata.sersic_residual)
    bkg_std = float(MADStdBackgroundRMS()(np.nan_to_num(data, nan=0.0)))

    grid = list(initial_grid)
    seen = set()
    attempts = []
    while True:
        for cand in grid:
            if cand in seen:
                continue
            seen.add(cand)
            psf_cand = library.copy()
            if cand > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    psf_cand = ap_convolve(psf_cand, Gaussian2DKernel(cand))
                psf_cand = _normalize_psf(psf_cand)
            try:
                phot, _ = _single_pass_phot(
                    data, psf_cand, psf_oversample, psf_sigma,
                    mask=mask)
            except Exception as e:
                log(f'[bootstrap_blur] blur={cand:.2f}: phot error: {e}')
                attempts.append((cand, 0))
                if on_progress is not None:
                    try: on_progress()
                    except Exception: pass
                continue
            if phot is None or len(phot) == 0:
                attempts.append((cand, 0))
                log(f'[bootstrap_blur] blur={cand:.2f}: 0 detections')
                if on_progress is not None:
                    try: on_progress()
                    except Exception: pass
                continue
            try:
                s_pass, _ = filter_psfphot_results(phot, bkg_std=bkg_std)
                n_pass = int(np.asarray(s_pass).sum())
            except Exception:
                n_pass = int(len(phot))
            attempts.append((cand, n_pass))
            log(f'[bootstrap_blur] blur={cand:.2f}: n_pass={n_pass}')
            if on_progress is not None:
                try: on_progress()
                except Exception: pass

        # any positive yield?
        positive = [(c, n) for c, n in attempts if n > 0]
        if positive:
            best, n_best = max(positive, key=lambda r: r[1])
            log(f'[bootstrap_blur] chose blur={best:.2f} (n_pass={n_best})')
            try:
                cutoutdata.blur_psf(float(best))
            except Exception as e:
                log(f'[bootstrap_blur] cd.blur_psf failed: {e}')
                return None
            return float(best)

        # expand the grid upward
        max_seen = max(seen)
        if max_seen >= max_blur:
            log(f'[bootstrap_blur] no candidate worked up to '
                f'blur={max_blur}; giving up')
            return None
        new_blurs = []
        b = max_seen + expand_step
        while b <= max_blur:
            new_blurs.append(round(b, 2))
            b += expand_step
        if not new_blurs:
            log('[bootstrap_blur] grid exhausted')
            return None
        grid = new_blurs
        log(f'[bootstrap_blur] expanding to {grid}')


# =====================================================================
# Section 6 — per-iter helpers: bracket fwhm + skip-on-stable
# =====================================================================

def _bracket_fwhm_grid(prev_fac, lo=0.5, hi=3.5):
    """3-point bracket centered on `prev_fac`, clamped to [lo, hi]."""
    grid = sorted({
        round(max(lo, prev_fac * 0.7), 3),
        round(min(max(prev_fac, lo), hi), 3),
        round(min(hi, prev_fac * 1.4), 3),
    })
    return grid


def _kernels_close(K_new, K_prev, threshold=1e-3):
    """Per-pixel RMS distance between two same-shape kernels < threshold.
    Used to skip the fwhm scan when the calibration has stabilised.
    """
    if K_prev is None or K_new is None:
        return False
    K_new = np.asarray(K_new, dtype=float)
    K_prev = np.asarray(K_prev, dtype=float)
    if K_new.shape != K_prev.shape:
        return False
    return float(np.sqrt(np.mean((K_new - K_prev) ** 2))) < threshold


# =====================================================================
# Section 7 — per-iter calibration step (in-mainloop hook)
# =====================================================================

def calibrate_psf_step(
    cutoutdata,
    *,
    family: str = 'gaussian',
    K: int = 30,
    fwhm_default_grid: Sequence[float] = (1.0, 1.5, 2.355),
    bracket_lo: float = 0.5,
    bracket_hi: float = 3.5,
    skip_stable_threshold: float = 1e-3,
    progress=None,
    log=print,
):
    """One PSF calibration step. Reuses the just-completed photometry
    and `cd.sersic_residual` to fit a kernel and update `cd.psf`.

    Steps:
      1. Pull (x, y, flux) of the K brightest quality-passing sources
         from `cd.psf_table` (or run bootstrap blur if none).
      2. Fit kernel parameters via the multi-source NNLS objective.
      3. `cd.psf <- library * K`.
      4. Bracket-scan `dao_fwhm_factor`, or reuse the previous one if
         the kernel L2 distance is below `skip_stable_threshold`.
      5. Persist `cd.kernel_params`, `cd.dao_fwhm_factor`,
         `cd._calibrate_psf_prev_kernel`.

    `family`: 'gaussian' (default) | 'moffat' | 'drizzle'.
    """
    import time
    from .psf import filter_psfphot_results, _build_center_mask
    from .config import config
    from photutils.background import MADStdBackgroundRMS

    t0 = time.time()
    sersic_residual = np.asarray(cutoutdata.sersic_residual)
    # Treat the FIRST cd.psf we see as the un-blurred library, since
    # subsequent calibration applies K on top of `cd._psf_raw` and
    # would otherwise compose K onto an already-modulated PSF.
    if (not hasattr(cutoutdata, '_psf_raw')
            or getattr(cutoutdata, '_psf_raw', None) is None):
        cutoutdata._psf_raw = np.asarray(cutoutdata.psf, dtype=float).copy()
    library_psf = np.asarray(cutoutdata._psf_raw, dtype=float)
    library_norm = _normalize_psf(library_psf)
    psf_oversample = int(cutoutdata.psf_oversample)
    psf_sigma = float(cutoutdata.psf_sigma)

    bkg_std = float(MADStdBackgroundRMS()(
        np.nan_to_num(sersic_residual, nan=0.0)))

    # Build the center-exclusion mask used by the bootstrap blur scan,
    # fwhm scan, and anchor selection. Radius is in units of psf_sigma
    # (so a wider effective PSF gets a proportionally wider mask).
    # Distinct from the iPSF mask `[psf].center_mask_r_pix` (data px).
    center_mask = None
    center_mask_params = None
    try:
        sp = cutoutdata.sersic_params_physical
        cm_factor = float(config.get('psf-calib', {}).get(
            'center_mask_r_in_fwhm', 2.5))
        if cm_factor > 0:
            # Mask is sized in units of the EFFECTIVE PSF FWHM (not library
            # psf_sigma). The Sersic-residual blob at the galaxy core scales
            # with the convolved PSF, so library σ — which is sub-pixel for
            # sharp filters — is way too small a unit.
            cm_r = effective_fwhm_data_px(cutoutdata) * cm_factor
            center_mask_params = [float(sp['x_0']), float(sp['y_0']), cm_r]
            center_mask = _build_center_mask(sersic_residual.shape,
                                              center_mask_params)
    except Exception:
        center_mask = None
        center_mask_params = None

    # 1. extract anchors from psf_table (already-fitted)
    phot = cutoutdata.psf_table
    no_phot = phot is None or len(phot) == 0
    if not no_phot:
        try:
            s_pass, _ = filter_psfphot_results(phot, bkg_std=bkg_std)
            s_pass = np.asarray(s_pass)
        except Exception:
            s_pass = np.ones(len(phot), dtype=bool)
        no_pass = int(s_pass.sum()) == 0
    else:
        no_pass = True

    if no_phot or no_pass:
        # iPSF found nothing usable. Run an expanding-range blur scan
        # to recover a sensible PSF from the data; the next mainloop
        # iter's iPSF will pick up the updated cd.psf.
        log('[calibrate_psf_step] no usable photometry; running blur '
            'bootstrap')
        bootstrap_grid = (0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
        if progress is not None:
            boot_task = progress.add_task(
                'Calibrating PSF (bootstrap blur)',
                total=len(bootstrap_grid))
            def _tick_bootstrap():
                progress.update(boot_task, advance=1, refresh=True)
        else:
            boot_task = None
            _tick_bootstrap = None
        try:
            chosen = _bootstrap_blur(
                cutoutdata, initial_grid=bootstrap_grid, log=log,
                on_progress=_tick_bootstrap, mask=center_mask)
        finally:
            if progress is not None and boot_task is not None:
                try: progress.remove_task(boot_task)
                except Exception: pass
        if chosen is None:
            log('[calibrate_psf_step] blur bootstrap failed; '
                'leaving cd.psf unchanged')
            return None
        # Reset per-cutout state; the kernel/fwhm will be recalibrated on
        # the next call once iPSF produces a phot_table.
        cutoutdata.kernel_params = None
        cutoutdata.dao_fwhm_factor = 2.33
        cutoutdata._calibrate_psf_prev_kernel = None
        config['psf']['dao_fwhm_factor'] = 2.33
        return {
            'kernel_family': None,
            'kernel_params': None,
            'dao_fwhm_factor': 2.33,
            'bootstrapped_blur': float(chosen),
            'wall_s': float(time.time() - t0),
        }
    phot_pass = phot[s_pass]
    x = np.asarray(phot_pass['x_fit'], dtype=float)
    y = np.asarray(phot_pass['y_fit'], dtype=float)
    f = np.asarray(phot_pass['flux_fit'], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(f) & (f > 0)
    # Drop any anchors that fall inside the centre-mask radius. iPSF
    # already excludes them via PSFPhotometry's mask, but stale
    # cd.psf_table from older runs may still carry such rows.
    if center_mask_params is not None:
        x_c, y_c, r_c = center_mask_params
        valid &= ((x - x_c) ** 2 + (y - y_c) ** 2 > r_c ** 2)
    x, y, f = x[valid], y[valid], f[valid]
    if len(x) == 0:
        log('[calibrate_psf_step] all anchors invalid; skipping')
        return None
    # FULL passing list — used by the multi-source objective for
    # neighbour lookup. Stays separate from the anchor sub-selection.
    all_x_pass = x.copy()
    all_y_pass = y.copy()
    all_flux_pass = f.copy()
    order = np.argsort(-f)[:K]
    x_anchor, y_anchor, f_anchor = x[order], y[order], f[order]
    n_pass_input = int(s_pass.sum())

    # 2. kernel fit (per-source residual against sersic_residual)
    prev_params = getattr(cutoutdata, 'kernel_params', None)
    cfg_calib = config.get('psf-calib', {})

    objective = str(cfg_calib.get('kernel_fit_objective',
                                   'multi_source')).lower()
    max_neighbors = int(cfg_calib.get('kernel_fit_max_neighbors', 15))
    neighbor_pad = float(cfg_calib.get('kernel_fit_neighbor_pad_pix', 4.0))

    # Stamp size: multi_source scales with the effective PSF so the
    # wings actually fit inside the stamp; single_source keeps 17×17.
    stamp_half_min = int(cfg_calib.get('kernel_fit_stamp_half_min', 8))
    stamp_half_factor = float(cfg_calib.get(
        'kernel_fit_stamp_half_in_sigmaeff', 3.0))
    if objective == 'multi_source':
        if prev_params is not None and 'sigma_data' in prev_params:
            sigma_kernel_prev = float(prev_params['sigma_data'])
        else:
            sigma_kernel_prev = 1.5
        sigma_eff_est = float(np.sqrt(psf_sigma ** 2 + sigma_kernel_prev ** 2))
        half_anchor = max(stamp_half_min,
                          int(np.ceil(stamp_half_factor * sigma_eff_est)))
    else:
        half_anchor = stamp_half_min

    # Progress task: 1 kernel-fit phase + 3 fwhm-scan phases. The scan
    # portion is bulk-advanced if skip-on-stable fires.
    n_seeds_expected = 1
    n_scan_phases = 3
    if progress is not None:
        calib_task = progress.add_task(
            f'Calibrating PSF (kernel fit 0/{n_seeds_expected})',
            total=n_seeds_expected + n_scan_phases)
        seeds_done = [0]
        def _tick_seed():
            seeds_done[0] += 1
            try:
                progress.update(
                    calib_task, advance=1,
                    description=(f'Calibrating PSF '
                                 f'(kernel fit {seeds_done[0]}/'
                                 f'{n_seeds_expected})'),
                    refresh=True)
            except Exception:
                pass
    else:
        calib_task = None
        _tick_seed = None

    # Crop the library to the central region used by the fit. The fit
    # only needs eff_psf within ±half_anchor data px of each anchor,
    # plus the kernel's own support; cropping shrinks the per-call
    # fftconvolve and spline interpolations dramatically (663²→401²
    # is ~2.7× faster). Use the full library for the final eff_psf
    # written to cd.psf.
    fit_half_oversampled = max(
        200, int(np.ceil(10.0 * psf_sigma * psf_oversample)))
    library_for_fit = _normalize_psf(
        _crop_centered(library_norm, fit_half_oversampled))
    try:
        fit = _fit_kernel_to_data(
            library_for_fit, psf_oversample, sersic_residual,
            x_anchor, y_anchor, f_anchor,
            family, half=half_anchor,
            on_seed_done=_tick_seed,
            objective=objective,
            all_x=all_x_pass, all_y=all_y_pass, all_flux=all_flux_pass,
            max_neighbors=max_neighbors, neighbor_pad=neighbor_pad)
    except Exception:
        if progress is not None and calib_task is not None:
            try: progress.remove_task(calib_task)
            except Exception: pass
        raise
    log(f'[calibrate_psf_step] kernel={family} '
        f'params={fit["params"]}, fit_l2={fit["residual_l2"]:.3g}')

    # 3. apply: cd.psf ← library * K
    new_psf = _build_effective_psf(
        library_norm, psf_oversample, family, fit['params'],
        half=max(20, int(np.ceil(6.0 * psf_sigma))))
    cutoutdata.psf = new_psf

    # 4. skip-on-stable check (only honoured when the user opts in via
    # `[psf-calib].kernel_fit_skip_on_convergence`). Default off — the
    # fwhm scan runs every iteration so a still-evolving kernel always
    # gets a fresh matched-filter width.
    skip_on_conv = bool(cfg_calib.get('kernel_fit_skip_on_convergence', False))
    prev_kernel = getattr(cutoutdata, '_calibrate_psf_prev_kernel', None)
    skip_scan = (skip_on_conv
                 and _kernels_close(fit['kernel_data'], prev_kernel,
                                     threshold=skip_stable_threshold))
    prev_fac = getattr(cutoutdata, 'dao_fwhm_factor', None)

    fwhm_method = str(cfg_calib.get('fwhm_method', 'scan')).lower()

    if skip_scan and prev_fac is not None:
        log(f'[calibrate_psf_step] kernel stable (Δrms<{skip_stable_threshold});'
            f' reusing fwhm_factor={prev_fac}')
        chosen_fac = float(prev_fac)
        scan_records = None
        # Bulk-advance through the (skipped) scan phases so the bar
        # reaches 100% rather than appearing stuck at the kernel-fit step.
        if progress is not None and calib_task is not None:
            try:
                progress.update(
                    calib_task, advance=n_scan_phases,
                    description='Calibrating PSF (scan skipped)',
                    refresh=True)
            except Exception:
                pass
    elif fwhm_method == 'empirical':
        # Skip the bracket scan entirely. Use photutils.fit_fwhm on the
        # calibrated PSF (matched-filter theorem). Falls back to prev_fac
        # if the fit fails so a transient numerical issue doesn't blow up
        # the whole calibration.
        emp_fac = _empirical_fwhm_factor(new_psf, psf_oversample, psf_sigma)
        if emp_fac is None or not np.isfinite(emp_fac):
            chosen_fac = float(prev_fac if prev_fac is not None else 2.355)
            log(f'[calibrate_psf_step] empirical FWHM fit failed; '
                f'reusing fwhm_factor={chosen_fac}')
        else:
            chosen_fac = float(emp_fac)
            log(f'[calibrate_psf_step] empirical FWHM: '
                f'fwhm_factor={chosen_fac:.4f} '
                f'(F_eff={chosen_fac*psf_sigma:.3f} data px)')
        scan_records = None
        if progress is not None and calib_task is not None:
            try:
                progress.update(
                    calib_task, advance=n_scan_phases,
                    description='Calibrating PSF (empirical FWHM)',
                    refresh=True)
            except Exception:
                pass
    else:
        if prev_fac is None:
            grid = list(fwhm_default_grid)
        else:
            grid = _bracket_fwhm_grid(prev_fac, lo=bracket_lo, hi=bracket_hi)
        if progress is not None and calib_task is not None:
            scan_done = [0]
            def _tick_scan():
                scan_done[0] += 1
                try:
                    progress.update(
                        calib_task, advance=1,
                        description=(f'Calibrating PSF '
                                     f'(fwhm scan {scan_done[0]}/'
                                     f'{len(grid)})'),
                        refresh=True)
                except Exception:
                    pass
        else:
            _tick_scan = None
        scan_method = str(cfg_calib.get('scan_method', 'lm')).lower()
        if scan_method == 'nnls':
            scan_metric = str(cfg_calib.get('scan_metric', 'sum_snr')).lower()
            scan_top_k = int(cfg_calib.get('scan_top_k', 100) or 0)
            chosen_fac, scan_records = _scan_dao_fwhm_factor_nnls(
                sersic_residual, new_psf, psf_oversample, psf_sigma,
                grid, metric=scan_metric, top_k=scan_top_k,
                log=log, on_progress=_tick_scan, mask=center_mask)
        else:
            chosen_fac, scan_records = _scan_dao_fwhm_factor(
                sersic_residual, new_psf, psf_oversample, psf_sigma,
                grid, log=log, on_progress=_tick_scan,
                mask=center_mask)
        if chosen_fac is None:
            chosen_fac = float(prev_fac if prev_fac is not None else 2.33)

    # 5. publish dao_fwhm_factor for the next iPSF call
    config['psf']['dao_fwhm_factor'] = float(chosen_fac)

    # 6. persist state on cutoutdata
    cutoutdata.kernel_family = family
    cutoutdata.kernel_params = dict(fit['params'])
    cutoutdata.dao_fwhm_factor = float(chosen_fac)
    cutoutdata._calibrate_psf_prev_kernel = np.asarray(
        fit['kernel_data'], dtype=float).copy()

    # diagnostic n_pass: pull from the chosen scan record if scan ran
    n_pass_after = n_pass_input
    if scan_records:
        for rec in scan_records:
            if abs(rec['fwhm_factor'] - chosen_fac) < 1e-6:
                n_pass_after = int(rec['n_pass'])
                break

    out = {
        'kernel_family': family,
        'kernel_params': dict(fit['params']),
        'dao_fwhm_factor': float(chosen_fac),
        'n_passing_input': n_pass_input,
        'n_passing_post_scan': n_pass_after,
        'fit_residual_l2': float(fit['residual_l2']),
        'skipped_scan': bool(skip_scan and prev_fac is not None),
        'wall_s': float(time.time() - t0),
    }
    log(f'[calibrate_psf_step] done in {out["wall_s"]:.1f}s '
        f'(skipped_scan={out["skipped_scan"]}, '
        f'fwhm_factor={chosen_fac:.2f})')
    if progress is not None and calib_task is not None:
        try:
            progress.remove_task(calib_task)
        except Exception:
            pass
    return out
