from .utils import load_and_crop, prep_model, update_model_with_isophot_fit
from .psf import PSFFitter
from .fitting import ModelFitter, ModelScaleFitter
from .plotting import plot_sphot_results
from .aperture import aperture_routine
from .calibrate_psf import calibrate_psf_step
from .config import config

import glob
import sys
import warnings
import numpy as np
from rich.progress import (Progress, TimeElapsedColumn,
                           TimeRemainingColumn, BarColumn, TextColumn)
from rich.live import Live
from rich import get_console
import logging
import matplotlib.pyplot as plt

from .logging import logger


def _seed_dao_fwhm_factor(cutoutdata):
    """Sync `config['psf']['dao_fwhm_factor']` to this cutoutdata's own
    stored value (or 2.33 default). Required because DAOStarFinder is
    built from the global config inside `_prepare_psf_fitters`, but
    each filter has its own optimal matched-filter width. Without this
    reset, a previous filter's `dao_fwhm_factor` leaks into the next
    filter's photometry and can drive DAO to "No detection" when the
    kernels don't match (e.g. F090W's 1.0 vs F160W's wider PSF).
    """
    config['psf']['dao_fwhm_factor'] = float(
        getattr(cutoutdata, 'dao_fwhm_factor', 2.33))


def _refresh_sersic_modelimg(cutoutdata, fit_complex_model=False):
    """Re-render `cd.sersic_modelimg` / `cd.sersic_residual` using the
    CURRENT `cd.psf` and the SAVED `cd.sersic_params`. Called after
    `calibrate_psf_step` so the PSF-convolved Sersic model tracks the
    just-updated effective PSF (no Sersic re-fit, just a re-render).
    """
    if not hasattr(cutoutdata, 'sersic_params'):
        return
    if cutoutdata.sersic_params is None:
        return
    try:
        model = prep_model(cutoutdata, simple=not fit_complex_model)
        fitter = ModelFitter(model, cutoutdata)
        fitter.data = cutoutdata._rawdata
        fitter.bounds_physical = fitter.model.get_bounds()
        new_img = fitter.eval_model(np.asarray(cutoutdata.sersic_params))
        cutoutdata.sersic_modelimg = new_img
        cutoutdata.sersic_residual = cutoutdata._rawdata - new_img
        # Also refresh residual / residual_masked consistently — they're
        # what `remove_sky` fits sky to. Without this, a stale
        # residual_masked (already several layers sky-subtracted from
        # per-iter remove_sky calls) would give a near-zero sky model on
        # the next remove_sky_sersic call, leaving sersic_residual
        # carrying the raw background. Mirrors ModelFitter.fit's
        # post-fit residual reset.
        psf_modelimg = getattr(cutoutdata, 'psf_modelimg', 0)
        cutoutdata.residual = cutoutdata._rawdata - psf_modelimg - new_img
        cutoutdata.residual_masked = cutoutdata.residual.copy()
    except Exception as e:
        logger.warning(f'Sersic re-render after PSF calibration failed: {e}')


def _refresh_fitter_psf(fitter, cutoutdata, simple=True):
    """Rebuild `fitter.model` so the next fit convolves with the latest `cd.psf`.

    `prep_model` captures `cd.psf` at construction (via SphotModel /
    petrofit's PSFConvolvedModel2D) and stores its own preprocessed copy.
    Reassigning `cd.psf = new_psf` inside `_maybe_recalibrate_psf`
    therefore has zero effect on the captured copy — every subsequent
    `fitter.fit(...)` would otherwise convolve the Sersic against the
    stale PSF. This helper rebuilds the model with the current `cd.psf`,
    preserving warm-start params (`model.x0`) and any fixed-parameter
    map so the next fit picks up where the optimizer left off.
    """
    if fitter is None:
        return
    try:
        new_model = prep_model(cutoutdata, simple=simple)
        old_x0 = getattr(fitter.model, 'x0', None)
        old_fixed = getattr(fitter.model, 'fixed_params', None)
        if old_fixed:
            try:
                new_model.set_fixed_params(old_fixed)
            except Exception:
                pass
        if old_x0 is not None:
            new_model.x0 = old_x0
        fitter.model = new_model
        fitter.bounds_physical = new_model.get_bounds()
    except Exception as e:
        logger.warning(f'_refresh_fitter_psf failed: {e}')


def _maybe_recalibrate_psf(cutoutdata, progress=None,
                            fit_complex_model=False):
    """Run PSF kernel + DAO fwhm recalibration on `cutoutdata` if
    `[psf-calib].in_mainloop` is true. After a successful call, also
    re-render `cd.sersic_modelimg` / `cd.sersic_residual` against the
    new effective PSF so downstream code (next iPSF, plotting, save)
    sees a self-consistent state.

    `progress` (optional): a rich.Progress (or QueueProgressProxy in
    parallel mode) — passed through so calibrate_psf_step can render a
    child progress bar in the main loop's task slot.
    """
    cfg = config.get('psf-calib', {})
    if not cfg.get('in_mainloop', False):
        return
    try:
        calibrate_psf_step(
            cutoutdata,
            family=cfg.get('kernel_family', 'gaussian'),
            K=int(cfg.get('kernel_anchor_K', 30)),
            skip_stable_threshold=float(cfg.get(
                'kernel_skip_threshold', 1e-3)),
            progress=progress,
            log=lambda msg: logger.info(msg),
        )
    except Exception as e:
        import traceback
        logger.warning(f'calibrate_psf_step failed: {e}; '
                       f'continuing without recalibration\n'
                       f'{traceback.format_exc()}')
        return
    _refresh_sersic_modelimg(cutoutdata, fit_complex_model=fit_complex_model)


def _params_converged(prev, curr, atol):
    ''' True when standardized sersic params are within atol of the previous iteration '''
    if prev is None or curr is None:
        return False
    prev = np.asarray(prev)
    curr = np.asarray(curr)
    if prev.shape != curr.shape:
        return False
    return np.allclose(prev, curr, atol=atol, rtol=0)

def ignorewarnings(func):
    def wrapper(*args,**kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return func(*args,**kwargs)
    return wrapper
            
def showprogress(func):
    def wrapper(*args,**kwargs):
        if kwargs.get('progress',None) is None:
            console = kwargs.get('console',None)
            if console is not None:
                logger.info('console object detected: switching output to given console')
            else:
                logger.info('console object is not detected: using the standard output')
                console = get_console()
                
            # initialize progress bar object
            progress =  Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=False,
                console=console,
            )
            live = Live(progress, console=console, refresh_per_second=4,
                        transient=False)
            with live:
                kwargs.update(dict(progress=progress))
                return func(*args,**kwargs)
        else:
            return func(*args,**kwargs)
    return wrapper

@ignorewarnings
@showprogress
def run_basefit(galaxy,base_filter,
               fit_complex_model=False,blur_psf=None,
               N_mainloop_iter=5,
               progress=None,
               verbose=False,plot=False,**kwargs):
    kwargs['verbose'] = verbose
    kwargs['plot'] = plot
    
    # convenience kwargs
    kwargs_sersic_init  = dict(fit_to='data',max_iter=20,progress=progress)
    kwargs_sersic_init2 = dict(fit_to='psf_sub_data',method='iterative_NM',max_iter=30,progress=progress)
    kwargs_sersic_iter  = dict(fit_to='psf_sub_data',method='iterative_NM', max_iter=15, progress=progress)
    kwargs_psf          = dict(fit_to='sersic_residual',plot=False,progress=progress)
    kwargs_rmsky_sersic = dict(fit_to='residual_masked',remove_from=['sersic_residual','residual_masked','residual'],repeat=3,**kwargs)
    kwargs_rmsky_psf    = dict(fit_to='residual_masked',remove_from=['psf_sub_data','residual_masked','residual'],repeat=3,**kwargs)

    # 1. select base filter to fit & initialize
    cutoutdata = galaxy.images[base_filter]
    for attrname in ['sersic_modelimg','psf_modelimg']:
        setattr(cutoutdata,attrname,0) # remove previous results
    cutoutdata.perform_bkg_stats()
    # Apply user-supplied blur when given; otherwise leave cd.psf as
    # the library and let the calibrator's bootstrap pick a blur from
    # the data on the first `_maybe_recalibrate_psf` call.
    if blur_psf is not None:
        cutoutdata.blur_psf(blur_psf)
    _seed_dao_fwhm_factor(cutoutdata)

    # 2. make models & prepare fitters
    model_1 = prep_model(cutoutdata,simple=True)
    model_2 = prep_model(cutoutdata,simple=False) if fit_complex_model else model_1
    model_1 = update_model_with_isophot_fit(model_1,cutoutdata,fit_to='data')
    model_2 = update_model_with_isophot_fit(model_2,cutoutdata,fit_to='psf_sub_data') if fit_complex_model else model_1
    fitter_1 = ModelFitter(model_1,cutoutdata)
    fitter_2 = ModelFitter(model_2,cutoutdata) if fit_complex_model else fitter_1
    fitter_psf = PSFFitter(cutoutdata)

    # 3. Sphot base fitting
    logger.info(f'Fitting the base filter {base_filter}...')
    progress_main = progress.add_task('Sphot main loop', total=N_mainloop_iter+1)
        
    # initial fit
    fitter_1.fit(**kwargs_sersic_init)
    cutoutdata.remove_sky(**kwargs_rmsky_sersic)
    fitter_psf.fit(**kwargs_psf)
    cutoutdata.remove_sky(**kwargs_rmsky_psf)
    _maybe_recalibrate_psf(cutoutdata, progress=progress, fit_complex_model=fit_complex_model)
    _refresh_fitter_psf(fitter_2, cutoutdata, simple=not fit_complex_model)
    if fit_complex_model:
        # initial fit for the complex model if needed
        fitter_2.fit(**kwargs_sersic_init2)
        cutoutdata.remove_sky(**kwargs_rmsky_sersic)
        fitter_psf.fit(**kwargs_psf)
        cutoutdata.remove_sky(**kwargs_rmsky_psf)
        _maybe_recalibrate_psf(cutoutdata, progress=progress, fit_complex_model=fit_complex_model)
        _refresh_fitter_psf(fitter_2, cutoutdata, simple=not fit_complex_model)
    progress.update(progress_main, advance=1, refresh=True)

    # repeat fitting (main loop) -- early-exit when Sersic params stop moving
    conv_atol = config['core'].get('mainloop_convergence_atol', 1e-3)
    conv_patience = config['core'].get('mainloop_convergence_patience', 2)
    min_iter = config['core'].get('mainloop_min_iter', 2)
    use_dual_annealing = config['core'].get('use_dual_annealing', True)
    use_final_polish = config['core'].get('use_final_polish', True)
    prev_params = np.array(cutoutdata.sersic_params, copy=True)
    consec_converged = 0
    for i in range(N_mainloop_iter):
        # Iter 0 of this loop is the SECOND main-loop iter overall (the
        # initial fit ran above on raw `data`); it's the first one fit to
        # `psf_sub_data`. That's where the chi^2 landscape is finally clean
        # enough for a global escape, so we route it through dual_annealing.
        if i == 0 and use_dual_annealing:
            kw = dict(kwargs_sersic_iter, method='dual_annealing',
                      progress=progress)
            fitter_2.fit(**kw)
        else:
            fitter_2.fit(**kwargs_sersic_iter)
        cutoutdata.remove_sky(**kwargs_rmsky_sersic)
        fitter_psf.fit(**kwargs_psf)
        cutoutdata.remove_sky(**kwargs_rmsky_psf)
        _maybe_recalibrate_psf(cutoutdata, progress=progress, fit_complex_model=fit_complex_model)
        _refresh_fitter_psf(fitter_2, cutoutdata, simple=not fit_complex_model)
        progress.update(progress_main, advance=1, refresh=True)

        curr_params = np.array(cutoutdata.sersic_params, copy=True)
        if _params_converged(prev_params, curr_params, conv_atol):
            consec_converged += 1
        else:
            consec_converged = 0
        prev_params = curr_params
        if (i + 1) >= min_iter and consec_converged >= conv_patience:
            logger.info(f'Sersic params converged after {i+1} main-loop '
                        f'iterations; stopping early.')
            progress.update(progress_main,
                            completed=N_mainloop_iter+1, refresh=True)
            break

    # If kernel calibration ran in the loop, the most recent
    # _maybe_recalibrate_psf updated cd.psf but its kernel was never used
    # for a photometry pass — cd.residual still reflects the previous
    # iteration's PSF. Run one final fitter_psf.fit() so cd.residual
    # matches the saved kernel_params.
    if config.get('psf-calib', {}).get('in_mainloop', False):
        fitter_psf.fit(**kwargs_psf)
        cutoutdata.remove_sky(**kwargs_rmsky_psf)

    # Final L-BFGS-B polish + one more PSF photometry pass.
    # iNM/dual_annealing leaves a few % of chi^2 on the table; gradient
    # polish picks it up cheaply once we are inside the basin. The polish
    # is followed by a sky-sub + fitter_psf.fit() pass so the saved
    # psf_modelimg / psf_table / sersic_residual are consistent with the
    # polished Sersic.
    if use_final_polish:
        fitter_2.fit(fit_to='psf_sub_data', method='lbfgsb_polish',
                     progress=progress)
        cutoutdata.remove_sky(**kwargs_rmsky_sersic)
        fitter_psf.fit(**kwargs_psf)
        cutoutdata.remove_sky(**kwargs_rmsky_psf)

    # final sky subtraction
    logger.info(f'*** Base model fit completed ***')
      
@ignorewarnings  
@showprogress
def run_scalefit(galaxy,filtername,base_params,allow_refit,
               fit_complex_model,blur_psf,
               N_mainloop_iter,
               progress=None,
               verbose=False,plot=False,**kwargs):
    kwargs['verbose'] = verbose
    kwargs['plot'] = plot
    
    # convenience kwargs
    kwargs_sersic_init  = dict(fit_to='data',progress=progress)
    kwargs_sersic_iter  = dict(fit_to='psf_sub_data',progress=progress)
    kwargs_psf          = dict(fit_to='sersic_residual',progress=progress,**kwargs)
    kwargs_rmsky_sersic = dict(fit_to='residual_masked',
                               remove_from=['sersic_residual','residual_masked','residual'],repeat=3,**kwargs)
    kwargs_rmsky_psf    = dict(fit_to='residual_masked',
                               remove_from=['psf_sub_data','residual_masked','residual'],repeat=3,**kwargs)
    if allow_refit:
        kwargs_sersic_refit = dict(fit_to='psf_sub_data',max_iter=20,progress=progress) 
        kwargs_sersic_refit_iter = dict(fit_to='psf_sub_data',max_iter=10,progress=progress)
    
    # 0. load and initialize
    logger.info(f'*** working on {filtername} ***')
    cutoutdata = galaxy.images[filtername]
    for attrname in ['sersic_modelimg','psf_modelimg']:
        setattr(cutoutdata,attrname,0) # remove previous results
        
    # 1. basic statistics
    cutoutdata.perform_bkg_stats()
    if blur_psf is not None:
        cutoutdata.blur_psf(blur_psf)
    _seed_dao_fwhm_factor(cutoutdata)

    # 2. initialize model and fitters
    _model = prep_model(cutoutdata,simple=False) if fit_complex_model else prep_model(cutoutdata,simple=True)
    fitter_scale = ModelScaleFitter(_model,cutoutdata,base_params)
    if allow_refit:
        fitter_2 = ModelFitter(_model,cutoutdata)
    fitter_psf = PSFFitter(cutoutdata)
        
    # 3. run fitting
    progress_main = progress.add_task('Sphot main loop',total=N_mainloop_iter+1)

    # -- initial fit
    fitter_scale.fit(**kwargs_sersic_init)
    cutoutdata.remove_sky(**kwargs_rmsky_sersic)
    fitter_psf.fit(**kwargs_psf)
    cutoutdata.remove_sky(**kwargs_rmsky_psf)
    _maybe_recalibrate_psf(cutoutdata, progress=progress, fit_complex_model=fit_complex_model)
    _refresh_fitter_psf(fitter_scale, cutoutdata, simple=not fit_complex_model)
    if allow_refit:
        _refresh_fitter_psf(fitter_2, cutoutdata, simple=not fit_complex_model)
        fitter_2.model.x0 = cutoutdata.sersic_params
        fitter_2.fit(**kwargs_sersic_refit)
        cutoutdata.remove_sky(**kwargs_rmsky_sersic)
        fitter_psf.fit(**kwargs_psf)
        cutoutdata.remove_sky(**kwargs_rmsky_psf)
        _maybe_recalibrate_psf(cutoutdata, progress=progress, fit_complex_model=fit_complex_model)
        _refresh_fitter_psf(fitter_scale, cutoutdata, simple=not fit_complex_model)
        _refresh_fitter_psf(fitter_2, cutoutdata, simple=not fit_complex_model)
    progress.update(progress_main, advance=1, refresh=True)

    # -- repeat fitting (early-exit when Sersic params stop moving)
    conv_atol = config['core'].get('mainloop_convergence_atol', 1e-3)
    conv_patience = config['core'].get('mainloop_convergence_patience', 2)
    min_iter = config['core'].get('mainloop_min_iter', 2)
    prev_params = np.array(cutoutdata.sersic_params, copy=True)
    consec_converged = 0
    for i in range(N_mainloop_iter):
        if allow_refit:
            fitter_2.fit(**kwargs_sersic_refit_iter)
        else:
            fitter_scale.fit(**kwargs_sersic_iter)
        cutoutdata.remove_sky(**kwargs_rmsky_sersic)
        fitter_psf.fit(**kwargs_psf)
        cutoutdata.remove_sky(**kwargs_rmsky_psf)
        _maybe_recalibrate_psf(cutoutdata, progress=progress, fit_complex_model=fit_complex_model)
        _refresh_fitter_psf(fitter_scale, cutoutdata, simple=not fit_complex_model)
        if allow_refit:
            _refresh_fitter_psf(fitter_2, cutoutdata, simple=not fit_complex_model)
        progress.update(progress_main, advance=1, refresh=True)

        curr_params = np.array(cutoutdata.sersic_params, copy=True)
        if _params_converged(prev_params, curr_params, conv_atol):
            consec_converged += 1
        else:
            consec_converged = 0
        prev_params = curr_params
        if (i + 1) >= min_iter and consec_converged >= conv_patience:
            logger.info(f'{filtername}: params converged after {i+1} '
                        f'iterations; stopping early.')
            progress.update(progress_main,
                            completed=N_mainloop_iter+1, refresh=True)
            break

    # If kernel calibration ran in the loop, the most recent
    # _maybe_recalibrate_psf updated cd.psf but its kernel was never used
    # for a photometry pass. Also, _refresh_sersic_modelimg rebuilt
    # cd.sersic_residual from raw data without sky subtraction. Both need
    # a final cleanup pass to leave the saved file consistent:
    #   1. remove_sky_sersic FIRST while residual_masked is still full-bg
    #      from _refresh_sersic_modelimg (so a real sky is fit and
    #      subtracted from sersic_residual);
    #   2. fitter_psf.fit + remove_sky_psf SECOND so cd.residual matches
    #      the saved kernel_params and cd.sky_model carries the canonical
    #      raw-bg sky level.
    if config.get('psf-calib', {}).get('in_mainloop', False):
        cutoutdata.remove_sky(**kwargs_rmsky_sersic)
        fitter_psf.fit(**kwargs_psf)
        cutoutdata.remove_sky(**kwargs_rmsky_psf)

    # final sky subtraction
    logger.info(f'*** {filtername} done ***')


@ignorewarnings
@showprogress
def run_scalefit_forced(galaxy, filtername, base_filter, base_params,
                        fit_complex_model=False, blur_psf=None,
                        N_mainloop_iter=5,
                        recalibrate_psf=False,
                        progress=None,
                        verbose=False, plot=False, **kwargs):
    """Variant of `run_scalefit` that uses FIXED PSF source positions
    from `base_filter`'s saved psf_table — no detection, no threshold
    ladder. Force-extracts flux at every base-filter position.

    Useful for blurry/IR filters where DAO can't reliably detect dim
    point sources but the base filter (e.g. F150W on g260) has a
    clean source list.

    Parameters
    ----------
    base_filter : str
        Name of the base filter whose `psf_table` supplies the
        positions. Quality-filtered with `filter_psfphot_results`.
    base_params : array
        Sersic params from the base fit (same as `run_scalefit`).
    Other parameters mirror `run_scalefit`. The `allow_refit` switch
    is intentionally omitted — forced mode keeps positions pinned, so
    re-fitting Sersic with free PSF positions doesn't apply.
    """
    from .psf import run_forced_photometry_on_cutout
    kwargs['verbose'] = verbose
    kwargs['plot'] = plot

    # 0. extract base-filter positions ONCE. Filter out the photutils
    # flag bits that mark unfittable rows (out-of-bounds, no-overlap,
    # etc.). Works for both fresh QTable and h5-loaded ndarray.
    base_cd = galaxy.images[base_filter]
    base_phot = getattr(base_cd, 'psf_table', None)
    if base_phot is None or len(base_phot) == 0:
        raise ValueError(f'{base_filter}.psf_table is empty; run '
                         f'run_basefit first.')
    base_x_all = np.asarray(base_phot['x_fit'], dtype=float)
    base_y_all = np.asarray(base_phot['y_fit'], dtype=float)
    base_f_all = np.asarray(base_phot['flux_fit'], dtype=float)
    base_flags_all = np.asarray(base_phot['flags'], dtype=int)
    bad_flag_mask = base_flags_all & (2 + 4 + 32 + 64 + 128 + 256)
    valid = (np.isfinite(base_x_all) & np.isfinite(base_y_all)
             & np.isfinite(base_f_all) & (base_f_all > 0)
             & (bad_flag_mask == 0))
    base_x = base_x_all[valid]
    base_y = base_y_all[valid]
    base_f = base_f_all[valid]
    if len(base_x) == 0:
        raise ValueError(f'no quality-passing positions in '
                         f'{base_filter}.psf_table')
    logger.info(f'forced scalefit: using {len(base_x)} positions '
                f'from {base_filter}')

    kwargs_sersic_init = dict(fit_to='data', progress=progress)
    kwargs_sersic_iter = dict(fit_to='psf_sub_data', progress=progress)
    kwargs_rmsky_sersic = dict(
        fit_to='residual_masked',
        remove_from=['sersic_residual', 'residual_masked', 'residual'],
        repeat=3, **kwargs)
    kwargs_rmsky_psf = dict(
        fit_to='residual_masked',
        remove_from=['psf_sub_data', 'residual_masked', 'residual'],
        repeat=3, **kwargs)

    logger.info(f'*** working on {filtername} (forced) ***')
    cutoutdata = galaxy.images[filtername]
    for attrname in ['sersic_modelimg', 'psf_modelimg']:
        setattr(cutoutdata, attrname, 0)
    cutoutdata.perform_bkg_stats()
    if blur_psf is not None:
        cutoutdata.blur_psf(blur_psf)
    _seed_dao_fwhm_factor(cutoutdata)

    _model = (prep_model(cutoutdata, simple=False) if fit_complex_model
              else prep_model(cutoutdata, simple=True))
    fitter_scale = ModelScaleFitter(_model, cutoutdata, base_params)

    progress_main = progress.add_task(
        f'Sphot main loop (forced from {base_filter})',
        total=N_mainloop_iter + 1)

    # initial Sersic + forced PSF
    fitter_scale.fit(**kwargs_sersic_init)
    cutoutdata.remove_sky(**kwargs_rmsky_sersic)
    run_forced_photometry_on_cutout(
        cutoutdata, base_x, base_y, flux_init=base_f,
        progress=progress)
    cutoutdata.remove_sky(**kwargs_rmsky_psf)
    if recalibrate_psf:
        _maybe_recalibrate_psf(cutoutdata, progress=progress, fit_complex_model=fit_complex_model)
        _refresh_fitter_psf(fitter_scale, cutoutdata, simple=not fit_complex_model)
    progress.update(progress_main, advance=1, refresh=True)

    conv_atol = config['core'].get('mainloop_convergence_atol', 1e-3)
    conv_patience = config['core'].get('mainloop_convergence_patience', 2)
    min_iter = config['core'].get('mainloop_min_iter', 2)
    prev_params = np.array(cutoutdata.sersic_params, copy=True)
    consec_converged = 0
    for i in range(N_mainloop_iter):
        fitter_scale.fit(**kwargs_sersic_iter)
        cutoutdata.remove_sky(**kwargs_rmsky_sersic)
        run_forced_photometry_on_cutout(
            cutoutdata, base_x, base_y, flux_init=base_f,
            progress=progress)
        cutoutdata.remove_sky(**kwargs_rmsky_psf)
        if recalibrate_psf:
            _maybe_recalibrate_psf(cutoutdata, progress=progress, fit_complex_model=fit_complex_model)
            _refresh_fitter_psf(fitter_scale, cutoutdata, simple=not fit_complex_model)
        progress.update(progress_main, advance=1, refresh=True)
        curr_params = np.array(cutoutdata.sersic_params, copy=True)
        if _params_converged(prev_params, curr_params, conv_atol):
            consec_converged += 1
        else:
            consec_converged = 0
        prev_params = curr_params
        if (i + 1) >= min_iter and consec_converged >= conv_patience:
            logger.info(f'{filtername}: forced-scalefit params '
                        f'converged after {i+1} iterations; stopping early.')
            progress.update(progress_main,
                            completed=N_mainloop_iter + 1,
                            refresh=True)
            break

    # Same cleanup as run_scalefit: when _maybe_recalibrate_psf ran in
    # the loop, _refresh_sersic_modelimg rebuilt sersic_residual without
    # sky subtraction. Do remove_sky_sersic first (full-bg residual_masked
    # still available), then refresh forced photometry against the
    # current kernel and produce the canonical cd.sky_model.
    if config.get('psf-calib', {}).get('in_mainloop', False):
        cutoutdata.remove_sky(**kwargs_rmsky_sersic)
        run_forced_photometry_on_cutout(
            cutoutdata, base_x, base_y, flux_init=base_f,
            progress=progress)
        cutoutdata.remove_sky(**kwargs_rmsky_psf)

    logger.info(f'*** {filtername} done (forced) ***')


@ignorewarnings
@showprogress
def run_aperphot(**kwargs):
    returns = aperture_routine(**kwargs)
    logger.info('*** Aperture photometry completed ***')
    return returns