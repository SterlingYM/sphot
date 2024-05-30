from .utils import load_and_crop, prep_model
from .psf import PSFFitter
from .fitting import ModelFitter, ModelScaleFitter
from .plotting import plot_sphot_results

import glob
import sys
import warnings
from rich.progress import Progress

import matplotlib.pyplot as plt

def ignorewarnings(func):
    def wrapper(*args,**kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return func(*args,**kwargs)
    return wrapper
            
def showprogress(func):
    def wrapper(*args,**kwargs):
        if kwargs.get('progress',None) is None:
            # with warnings.catch_warnings():
            #     warnings.simplefilter('ignore')
            with Progress(transient=False) as progress:
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
               progress=None):
    # 1. select base filter to fit
    cutoutdata = galaxy.images[base_filter]
    cutoutdata.perform_bkg_stats()
    cutoutdata.blur_psf(blur_psf[base_filter])

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
    print(f'Fitting the base filter {base_filter}...')
    progress_main = progress.add_task('Sphot main loop', 
                                      total=N_mainloop_iter+1)

    fitter_1.fit(fit_to='data',max_iter=20,
                 progress=progress)
    fitter_psf.fit(fit_to='sersic_residual',
                   plot=False,progress=progress)
    fitter_2.fit(fit_to='psf_sub_data',
                method='iterative_NM',
                max_iter=30,progress=progress)
    progress.update(progress_main, advance=1, refresh=True)
    for _ in range(N_mainloop_iter):
        cutoutdata.fit_sky(fit_to='residual',plot=True)
        fitter_2.fit(fit_to='psf_sub_data',method='iterative_NM',
                     max_iter=15, progress=progress)
        fitter_psf.fit(fit_to='sersic_residual',plot=False,
                       progress=progress)
        progress.update(progress_main, advance=1, refresh=True)
  
@ignorewarnings  
@showprogress
def run_scalefit(galaxy,filtername,base_params,allow_refit,
               fit_complex_model,blur_psf,
               N_mainloop_iter,
               progress=None):
    print(f'*** working on {filtername} ***')
    _cutoutdata = galaxy.images[filtername]
    
    # basic statistics
    _cutoutdata.perform_bkg_stats()
    _cutoutdata.blur_psf(blur_psf)

    # initialize model and fitters
    if fit_complex_model:
        _model = prep_model(_cutoutdata,simple=False)
    else:
        _model = prep_model(_cutoutdata,simple=True)
    _fitter_scale = ModelScaleFitter(_model,_cutoutdata,base_params)
    if allow_refit:
        _fitter_2 = ModelFitter(_model,_cutoutdata)
    _fitter_psf = PSFFitter(_cutoutdata)
        
    # run fitting
    progress_main = progress.add_task('Sphot main loop', 
                                      total=N_mainloop_iter+1)

    _fitter_scale.fit(fit_to='data',progress=progress)
    _fitter_psf.fit(fit_to='sersic_residual',
                    plot=False,progress=progress)
    progress.update(progress_main, advance=1, refresh=True)

    if allow_refit:
        _fitter_2.model.x0 = _cutoutdata.sersic_params
        _fitter_2.fit(fit_to='psf_sub_data',
                      max_iter=20,progress=progress)
    for _ in range(N_mainloop_iter):
        _cutoutdata.fit_sky(fit_to='residual',plot=True)
        if allow_refit:
            _fitter_2.fit(fit_to='psf_sub_data',
                          max_iter=10,progress=progress)
        else:
            _fitter_scale.fit(fit_to='data',progress=progress)
        _fitter_psf.fit(fit_to='sersic_residual',plot=False,progress=progress)
        plt.show()
        progress.update(progress_main, advance=1, refresh=True)
