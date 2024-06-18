from .utils import load_and_crop, prep_model
from .psf import PSFFitter
from .fitting import ModelFitter, ModelScaleFitter
from .plotting import plot_sphot_results

import glob
import sys
import warnings
from rich.progress import (Progress, TimeElapsedColumn, 
                           TimeRemainingColumn, BarColumn, TextColumn)
import logging
import matplotlib.pyplot as plt

from .logging import logger

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
                
            # initialize progress bar object
            progress =  Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=False,
                console=console
            )
            with progress:
                kwargs.update(dict(progress=progress))
                return func(*args,**kwargs)
        else:
            return func(*args,**kwargs)
    return wrapper

@ignorewarnings
@showprogress
def run_basefit(galaxy,base_filter,
               fit_complex_model,blur_psf,
               N_mainloop_iter,
               progress=None,**kwargs):
    # 1. select base filter to fit
    cutoutdata = galaxy.images[base_filter]
    cutoutdata.perform_bkg_stats()
    cutoutdata.blur_psf(blur_psf)

    # 2. make models & prepare fitters
    model_1 = prep_model(cutoutdata,simple=True)
    fitter_1 = ModelFitter(model_1,cutoutdata)
    if fit_complex_model:
        model_2 = prep_model(cutoutdata,simple=False)
        fitter_2 = ModelFitter(model_2,cutoutdata)
    else:
        model_2 = model_1
        fitter_2 = fitter_1
    fitter_psf = PSFFitter(cutoutdata)

    # 3. fit the profile
    logger.info(f'Fitting the base filter {base_filter}...')
    progress_main = progress.add_task('Sphot main loop', total=N_mainloop_iter+1)

    # initial fit
    fitter_1.fit(fit_to='data',max_iter=20,progress=progress)
    cutoutdata.remove_sky(fit_to='residual_masked',remove_from='sersic_residual',**kwargs)
    fitter_psf.fit(fit_to='sersic_residual',plot=False,progress=progress)
    cutoutdata.remove_sky(fit_to='residual_masked',remove_from='psf_sub_data',**kwargs)
    fitter_2.fit(fit_to='psf_sub_data',method='iterative_NM',max_iter=30,progress=progress)
    cutoutdata.remove_sky(fit_to='residual_masked',remove_from='sersic_residual',**kwargs)
    progress.update(progress_main, advance=1, refresh=True)
    
    # repeat fitting
    for _ in range(N_mainloop_iter):
        fitter_2.fit(fit_to='psf_sub_data',method='iterative_NM', max_iter=15, progress=progress)
        cutoutdata.remove_sky(fit_to='residual_masked',remove_from='sersic_residual',**kwargs)
        fitter_psf.fit(fit_to='sersic_residual',plot=False, progress=progress)
        cutoutdata.remove_sky(fit_to='residual_masked',remove_from='psf_sub_data',**kwargs)
        progress.update(progress_main, advance=1, refresh=True)
        
    # final sky subtraction
    cutoutdata.remove_sky(fit_to='residual_masked',remove_from=['residual_masked','residual'],**kwargs)
    logger.info(f'*** Base model fit on {filtername} done ***')
      
@ignorewarnings  
@showprogress
def run_scalefit(galaxy,filtername,base_params,allow_refit,
               fit_complex_model,blur_psf,
               N_mainloop_iter,
               progress=None,
               **kwargs):
    logger.info(f'*** working on {filtername} ***')
    cutoutdata = galaxy.images[filtername]
    
    # basic statistics
    cutoutdata.perform_bkg_stats()
    cutoutdata.blur_psf(blur_psf)

    # initialize model and fitters
    if fit_complex_model:
        _model = prep_model(cutoutdata,simple=False)
    else:
        _model = prep_model(cutoutdata,simple=True)
    fitter_scale = ModelScaleFitter(_model,cutoutdata,base_params)
    if allow_refit:
        fitter_2 = ModelFitter(_model,cutoutdata)
    fitter_psf = PSFFitter(cutoutdata)
        
    # run fitting
    progress_main = progress.add_task('Sphot main loop',total=N_mainloop_iter+1)

    # initial fit
    fitter_scale.fit(fit_to='data',progress=progress)
    cutoutdata.remove_sky(fit_to='residual_masked',remove_from='sersic_residual',**kwargs)
    fitter_psf.fit(fit_to='sersic_residual',plot=False,progress=progress)
    cutoutdata.remove_sky(fit_to='residual_masked',remove_from='psf_sub_data',**kwargs)
    if allow_refit:
        fitter_2.model.x0 = cutoutdata.sersic_params
        fitter_2.fit(fit_to='psf_sub_data',max_iter=20,progress=progress)
        cutoutdata.remove_sky(fit_to='residual_masked',remove_from='sersic_residual',**kwargs)
    progress.update(progress_main, advance=1, refresh=True)

    # repeat fitting
    for _ in range(N_mainloop_iter):
        if allow_refit:
            fitter_2.fit(fit_to='psf_sub_data',max_iter=10,progress=progress)
        else:
            fitter_scale.fit(fit_to='psf_sub_data',progress=progress)
        cutoutdata.remove_sky(fit_to='residual_masked',remove_from='sersic_residual',**kwargs)
        fitter_psf.fit(fit_to='sersic_residual',plot=False,progress=progress)
        cutoutdata.remove_sky(fit_to='residual_masked',remove_from='psf_sub_data',**kwargs)
        progress.update(progress_main, advance=1, refresh=True)
        
    # final sky subtraction
    cutoutdata.remove_sky(fit_to='residual_masked',remove_from=['residual_masked','residual'],**kwargs)
    logger.info(f'*** {filtername} done ***')
