import sys
sys.path.append('../')  # Add the parent directory of your package to sys.path

import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import os
import astropy.units as u

from astropy.table import Table, vstack
from tqdm.auto import tqdm


from sphot.plotting import astroplot, plot_sersicfit_result

#######################################
# load data & perform initial analysis
#######################################
from sphot.data import (CutoutData, MultiBandCutout, 
                        load_h5data, get_data_annulus)

def load_and_crop(datafile,filters,folder_PSF,
                  base_filter,plot=True):
    # load PSFs
    psfs_data = []
    for filtername in filters:
        path = glob.glob(folder_PSF + f'*{filtername}_PSF*.npy')[0]
        psfs_data.append(np.load(path))#
    PSFs_dict = dict(zip(filters, psfs_data))

    # load data
    galaxy_ID = os.path.splitext(os.path.split(datafile)[-1])[0]
    galaxy = load_h5data(datafile, galaxy_ID, filters, PSFs_dict)
    if plot:
        galaxy.plot()

    # estimate size of the galaxy
    cutoutdata = galaxy.images[base_filter]
    cutoutdata.init_size_guess(sigma_guess=10, center_slack = 0.20,
                                plot=plot, sigma_kernel=5)

    # determine cutout size based on the initial fit
    galaxy_size = cutoutdata.size_guess
    x0, y0 = cutoutdata.x0_guess, cutoutdata.y0_guess

    cutout_size = galaxy_size * 6 * 2 # number of pixels in each axis (hence x2)
    galaxy.crop_in(x0, y0, cutout_size)
    if plot:
        galaxy.plot()
        plt.show()
    for cutoutdata in galaxy.image_list:
        cutoutdata.galaxy_size = galaxy_size
    return galaxy

# def perform_bkg_stats(cutoutdata,galaxy_size,plot=False):
    data_annulus = get_data_annulus(cutoutdata._rawdata,4*galaxy_size,plot=plot)
    bkg_mean = np.nanmean(data_annulus)
    bkg_std = np.nanstd(data_annulus)
    cutoutdata.remove_bkg(bkg_mean) # this updates data internally
    cutoutdata.data_err = np.ones_like(cutoutdata.data)*bkg_std
    return cutoutdata

#######################################
# prepare Sersic model & perform fitting
#######################################
from astropy.modeling import models
from sphot.fitting import SphotModel

def prep_model(cutoutdata,simple=False):
    # prepare model
    galaxy_size = cutoutdata.galaxy_size
    shape = cutoutdata.data.shape
    if simple:
        sersic = models.Sersic2D(amplitude=1, r_eff=galaxy_size, n=2,
                                x_0=shape[1]/2, y_0=shape[0]/2,
                                ellip=0.2, theta=np.pi/4)
        model = SphotModel(sersic, cutoutdata)
    else:
        disk = models.Sersic2D(amplitude=0.1, r_eff=galaxy_size*5, n=2,
                            x_0=shape[1]/2, y_0=shape[0]/2,
                            ellip=0.2, theta=np.pi/4)
        bulge = models.Sersic2D(amplitude=0.5, r_eff=galaxy_size/5, n=2,
                                x_0=shape[1]/2, y_0=shape[0]/2,
                                ellip=0.2, theta=np.pi/4)
        model = SphotModel(disk+bulge, cutoutdata) # some model constraints depend on the data
        model.set_conditions([('r_eff_0','r_eff_1')]) # enforce r_eff_0 >= r_eff_1
    model.set_fixed_params({})
    return model

#######################################
# fit Sersic profile
#######################################
from sphot.fitting import ModelFitter, iterative_NM
from scipy.optimize import dual_annealing
# def custom_fit_loop(fitter):
#     # fit (INM -> loop(dual annealing -> INM))
#     result,success = iterative_NM(fitter.calc_chi2, (), fitter.model.x0, 
#                                 fitter.bounds,
#                                 rtol_init=1e-3,rtol_iter=1e-4,
#                                 rtol_convergence=1e-6,xrtol=1,max_iter=5)
#     for i in range(2):
#         np.random.seed(i)
#         print('\n performing dual annealing global optimization...')
#         result = dual_annealing(fitter.calc_chi2,x0=result.x,args=(),
#                                 bounds = fitter.bounds,
#                                 maxiter = 5+i,
#                                 initial_temp = 10/(i+1),
#                                 minimizer_kwargs = dict(method='L-BFGS-B',
#                                                         bounds=fitter.bounds,
#                                                         options=dict(eps=1e-4/(i+1),
#                                                                     maxfun=50)))
        
#         print('\n performing iterative Nelder-Mead optimization...')
#         result, success = iterative_NM(fitter.calc_chi2,x0=result.x,args=(),
#                                     bounds = fitter.bounds,
#                                     max_iter = 5-i,
#                                     xrtol=1e-2/(i+1),
#                                     rtol_iter=1e-9/(i+1),
#                                     rtol_convergence=1e-10)
#         if success:
#             break
#     return result,success

# def fit_profile(model,cutoutdata,quick=False):
#     print('\nperforming Sersic profile fitting...',flush=True)
#     # prepare fitter & run fit (this takes time)
#     fitter = ModelFitter(model, cutoutdata.data, 
#                         err = cutoutdata.data_err)
#     if quick:
#         result,success = iterative_NM(fitter.calc_chi2, (), fitter.model.x0, 
#                                     fitter.bounds,
#                                     rtol_init=1e-3,rtol_iter=1e-3,
#                                     rtol_convergence=1e-6,xrtol=1,max_iter=10)
#     else:
#         result,success = custom_fit_loop(fitter)

#     # save & check results
#     bestfit_sersic_params = result.x.copy() 
#     bestfit_img = fitter.eval_model(result.x)
#     sersic_residual = cutoutdata._rawdata - bestfit_img - cutoutdata._bkg_level # always residual from raw data
#     cutoutdata.sersic_residual = sersic_residual
#     cutoutdata.sersic_modelimg = bestfit_img
#     cutoutdata.sersic_params = bestfit_sersic_params
#     return fitter, bestfit_sersic_params, bestfit_img, sersic_residual

def fit_fixed_profile(model,cutoutdata,base_params):
    ''' fit flux scale only ("quick" mode only) '''
    print('\nfitting flux scale...')
    # prepare fitter & run fit (this takes time)
    fitter = ModelFitter(model, cutoutdata.data, 
                        err = cutoutdata.data_err)
    result,success = iterative_NM(fitter.calc_chi2_fixedmodel, ([base_params]), 
                                  x0=[1], bounds=[[0,10]],
                                rtol_init=1e-3,rtol_iter=1e-3,
                                rtol_convergence=1e-6,xrtol=1,max_iter=10)

    # save & check results
    bestfit_scale = result.x.copy()
    scaled_sersic_params = fitter.scale_params(base_params,bestfit_scale)
    bestfit_img = fitter.eval_model(scaled_sersic_params)
    sersic_residual = cutoutdata._rawdata - bestfit_img - cutoutdata._bkg_level # always residual from raw data
    cutoutdata.sersic_residual = sersic_residual
    cutoutdata.sersic_modelimg = bestfit_img
    cutoutdata.sersic_params = scaled_sersic_params
    return fitter, scaled_sersic_params, bestfit_img, sersic_residual

#######################################
# perform PSF photometry
#######################################
from sphot.psf import do_psf_photometry, sigma_clip_outside_aperture, iterative_psf_fitting
from warnings import filterwarnings
filterwarnings("ignore")

# def fit_psf(sersic_residual,cutoutdata,sigma_psf,galaxy_size,**kwargs):
#     print('\nperforming PSF subtraction...')
#     psf_table, resid = iterative_psf_fitting(sersic_residual,cutoutdata.psf,
#                                             sigma_psf = sigma_psf,
#                                             psf_oversample = cutoutdata.psf_oversample,
#                                             psf_blurring=3.5,
#                                             threshold_list = np.arange(1.6,3.2,0.2)[::-1],
#                                             **kwargs)
#     psf_model_total = sersic_residual - resid

#     # generate PSF-subtracted data
#     mask = sigma_clip_outside_aperture(resid,galaxy_size,clip_sigma=5,
#                                     aper_size_in_r_eff=2,plot=True)
#     psf_subtracted_data = cutoutdata._rawdata - psf_model_total
#     psf_subtracted_data[mask] = np.nan

#     # subtract background (to be consistent)
#     data_annulus = get_data_annulus(psf_subtracted_data,4*galaxy_size,plot=False)
#     bkg_mean = np.nanmean(data_annulus)
#     bkg_std = np.nanstd(data_annulus)
#     psf_subtracted_data_bksub = psf_subtracted_data - bkg_mean
#     psf_subtracted_data_bksub_error = np.ones_like(psf_subtracted_data)*bkg_std
#     return psf_subtracted_data_bksub, psf_subtracted_data_bksub_error, psf_model_total, psf_table


###############
# plot results
###############
from sphot.plotting import plot_profile2d



if __name__ == '__main__':
    folder_PSF = '../sample_data/'
    datafile =  '../sample_data/g310.h5'
    filters = ['F555W','F814W','F090W','F150W','F160W','F277W']
    base_filter = 'F150W'
    pix_scale = 0.03 * u.arcsec
    N_mainloop_iter = 3
    fwhm_dict = dict(
        # https://hst-docs.stsci.edu/wfc3ihb/chapter-6-uvis-imaging-with-wfc3/6-6-uvis-optical-performance
        # https://hst-docs.stsci.edu/wfc3ihb/chapter-7-ir-imaging-with-wfc3/7-6-ir-optical-performance
        # https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions
        F555W = 0.067 * u.arcsec / pix_scale,
        F814W = 0.074 * u.arcsec / pix_scale,
        F090W = 0.033 * u.arcsec / pix_scale,
        F150W = 0.050 * u.arcsec / pix_scale,
        F160W = 0.151 * u.arcsec / pix_scale,
        F277W = 0.092 * u.arcsec / pix_scale,
    )
    
    # 1. load data & perform initial analysis
    galaxy_crop, galaxy_size = prep_galaxy_crop(datafile,filters,folder_PSF,base_filter,pix_scale)

    # 2. select base filter to fit
    cutoutdata = galaxy_crop.images[base_filter]
    cutoutdata = perform_bkg_stats(cutoutdata,galaxy_size)
        
    #### INITIAL FITTING ####
    # 3. make simple Sersic model & fit Sersic profile
    model = prep_model(cutoutdata,galaxy_size,simple=True)
    sersic_results = fit_profile(model,cutoutdata,quick=True)
    fitter, bestfit_sersic_params, bestfit_sersic_img, sersic_residual = sersic_results

    # 4. perform PSF photometry and subtraction
    psf_results = fit_psf(sersic_residual,cutoutdata,fwhm_dict[base_filter].value/2,galaxy_size,plot=False)
    psf_subtracted_data_bksub, psf_subtracted_data_bksub_error, psf_model_total = psf_results
    cutoutdata.data = psf_subtracted_data_bksub
    cutoutdata.error = psf_subtracted_data_bksub_error

    # 5. make 2-Sersic model
    model = prep_model(cutoutdata,galaxy_size,simple=False)

    #### 6. MAIN LOOP ####
    for i in range(N_mainloop_iter):
        # fit Sersic model
        sersic_results = fit_profile(model,cutoutdata,quick=True)
        fitter, bestfit_sersic_params, bestfit_sersic_img, sersic_residual = sersic_results
        model.x0 = bestfit_sersic_params

        # perform PSF photometry and subtraction
        psf_results = fit_psf(sersic_residual,cutoutdata,fwhm_dict[base_filter].value/2,galaxy_size,plot=False)
        psf_subtracted_data_bksub, psf_subtracted_data_bksub_error, psf_model_total = psf_results
        cutoutdata.data = psf_subtracted_data_bksub
        cutoutdata.error = psf_subtracted_data_bksub_error
        
    # 7. plot the results
    plot_sphot_results(cutoutdata._rawdata - cutoutdata._bkg_level,
                        bestfit_sersic_img,sersic_residual,
                        psf_model_total,psf_subtracted_data_bksub)