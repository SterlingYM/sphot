# fitter.py
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import logging
import io
from rich.progress import Progress
from tqdm.auto import tqdm
import h5py

from scipy import stats
from scipy.stats import (norm,exponnorm,powerlognorm,
                         powernorm,truncexpon,skewnorm,
                         multivariate_normal)
from scipy.optimize import minimize, dual_annealing, leastsq

import astropy.units as u
from astropy.stats import sigma_clip
from astropy.nddata import Cutout2D
from astropy.modeling import models
from astropy.convolution import convolve, Gaussian2DKernel
from petrofit import PSFConvolvedModel2D, model_to_image
from photutils.aperture import (CircularAperture,EllipticalAperture,
                                aperture_photometry, EllipticalAnnulus,
                                ApertureStats)

from .plotting import astroplot, plot_sersicfit_result
        
class SphotModel(PSFConvolvedModel2D):
    def __init__(self,model,cutoutdata):
        ''' A wrapper class for the petrofit model.
        Args:
            model (astropy FittableModel): model to fit.
            cutoutdata (CutoutData): the data to fit.
        '''
        super().__init__(model,
                        psf=cutoutdata.psf, 
                        psf_oversample = int(cutoutdata.psf_oversample),
                        oversample = int(cutoutdata.psf_oversample))
        self.data = cutoutdata.data
        self.free_params = self._param_names
        self.fixed_params = {}

    def parse_params(self,theta):
        ''' a helper function to parse coordinates to all relevant sub-models.
        '''
        full_params = {}
        full_params.update(self.fixed_params)
        full_params.update(dict(zip(self.free_params,theta)))

        parsed_params = []
        for key in self._param_names:
            if 'x_0' in key:
                parsed_params.append(full_params['x_0'])
            elif 'y_0' in key:
                parsed_params.append(full_params['y_0'])
            else:
                parsed_params.append(full_params[key])
        return parsed_params
    
    def list_to_params(self,theta):
        ''' convert a list of free parameters to the full model parameter array '''
        return self.parse_params(theta)

    def set_fixed_params(self,fixed_params):
        free_params = [param for param in self._param_names if param not in fixed_params.keys()]
        free_params = [param for param in free_params if ('x_0' not in param and 'y_0' not in param)]
        if 'x_0' not in fixed_params.keys():
            x_0 = [self.parameters[self._param_names.index(param)] for param in self._param_names if 'x_0' in param][0]
            y_0 = [self.parameters[self._param_names.index(param)] for param in self._param_names if 'y_0' in param][0]
            self.free_params = ['x_0','y_0',*free_params] # array of names
            self.fixed_params = fixed_params # dict
            self.x0_physical = [self.parameters[self._param_names.index(param)] for param in free_params]
            self.x0_physical = [x_0,y_0,*self.x0_physical]
        else:
            self.free_params = free_params # array of names
            self.fixed_params = fixed_params # dict
            self.x0_physical = [self.parameters[self._param_names.index(param)] for param in free_params]

    def get_bounds(self):
        data = self.data
        bounds = []
        for key in self.free_params:
            if 'x_0' in key:
                bounds.append([0,data.shape[1]])
            elif 'y_0' in key:
                bounds.append([0,data.shape[0]])
            elif 'amplitude' in key:
                bounds.append([0,np.nanmax(data)])
            elif 'r_eff' in key:
                bounds.append([0,data.shape[0]])
            elif 'n' in key:
                bounds.append([0.1,10])
            elif 'ellip' in key:
                bounds.append([0.01,1])
            elif 'theta' in key: 
                bounds.append([0,np.pi])
            elif 'psf_pa' in key:
                bounds.append([0,np.pi])
            else:
                raise ValueError(f'Unknown parameter {key}')
        return np.array(bounds)
    
    def set_conditions(self,list_of_conditions):
        ''' set condition functions.
        Args:
            list_of_conditions (list of 2-tuple): list of conditions, each as a 2-tuple. Each tuple is evaluated so that (a,b) is True if a >= b.
            Inside the tuple can be either the parameter name or numerical value. For example, [('r_eff',10),('r_eff_1','r_eff_0')] means that it returns True if (r_eff >= 10 AND r_eff_1 >= r_eff_0).
        '''
        def condition_func(theta):
            free_params_dict = dict(zip(self.free_params,theta))
            for condition in list_of_conditions:
                if isinstance(condition[0],str):
                    a = free_params_dict[condition[0]]
                else:
                    a = condition[0]
                    
                if isinstance(condition[1],str):
                    b = free_params_dict[condition[1]]
                else:
                    b = condition[1]
                if a < b:
                    return False
            return True
        self.condition_func = condition_func
        
class ModelFitter():
    def __init__(self,model,cutoutdata,**kwargs):
        self.cutoutdata = cutoutdata
        self.shape = self.cutoutdata.data.shape
        self.model = model
        self.param_names = model._param_names

    def standardize_params(self,params):
        ''' normalize parameters to be between 0 and 1 '''
        lower_bounds,upper_bounds = self.bounds_physical.T
        return (params - lower_bounds) / (upper_bounds - lower_bounds)

    def unstandardize_params(self,params):
        ''' convert back normalized parameters to the physical scale '''
        lower_bounds,upper_bounds = self.bounds_physical.T
        return params * (upper_bounds - lower_bounds) + lower_bounds

    def eval_model(self,standardized_params):
        ''' render the model image based on the given parameters '''
        params = self.unstandardize_params(standardized_params)
        
        parsed_params = self.model.parse_params(params)
        self.model.parameters = parsed_params
        _img = model_to_image(self.model, size=self.data.shape)
        return _img

    def calc_chi2(self,
                  standardized_params,
                  iterinfo='',print_val=False,chi2_min_allowed=1e-10):
        # parameter sanity check
        condition_func = getattr(self.model,'condition_func',False)
        if condition_func:
            if condition_func(standardized_params) == False:
                return np.inf
        
        # evaluate model
        model_img = self.eval_model(standardized_params)
        
        # sanity check
        if np.isfinite(model_img).sum() == 0:
            return np.inf
        
        if self.err is None:
            chi2 = np.nansum((self.data - model_img)**2)
        else:
            chi2 = np.nansum(((self.data - model_img)/self.err)**2)
        chi2 /= np.isfinite(self.data).sum()
        if chi2 <= chi2_min_allowed:
            return np.inf
        
        if print_val:
            print(f'\r {chi2:.4e} {iterinfo}   ',end='',flush=True)
        return chi2

    def fit(self,method='iterative_NM',fit_to='data',**kwargs):
        self.data = getattr(self.cutoutdata,fit_to)
        self.err  = getattr(self.cutoutdata,fit_to+'_error',None)
        
        # pre-compute the bounds
        self.bounds_physical = self.model.get_bounds()
        param_shape = len(self.model.free_params)
        self.bounds = np.vstack([np.ones(param_shape)*0,
                                 np.ones(param_shape)*1]).T
        if not hasattr(self.model,'x0'):
            self.model.x0 = self.standardize_params(self.model.x0_physical)
        
        if method == 'iterative_NM':
            iNM_kwargs = dict(rtol_init=1e-3,rtol_iter=1e-3,
                                      rtol_convergence=1e-6,xrtol=1,max_iter=10)
            iNM_kwargs.update(kwargs)
            result,success = iterative_NM(self.calc_chi2, (), self.model.x0, 
                                        self.bounds,**iNM_kwargs)
        elif method == 'triple_annealing':
            result,success = triple_annealing(self.calc_chi2,(),
                                              self.model.x0,
                                              self.bounds,
                                              **kwargs)
        elif method == 'BFGS':
            result = minimize(self.calc_chi2,self.model.x0,
                            bounds=self.bounds,
                            method='L-BFGS-B',
                            options=dict(eps=1e-4,maxfun=1000))
            success = result.success
        else:
            raise ValueError('method not recognized')
        
        # results
        bestfit_sersic_params_physical = dict(zip(self.model.free_params,
                                                self.unstandardize_params(result.x)))
        bestfit_img = self.eval_model(result.x)
        sersic_residual = self.cutoutdata._rawdata - bestfit_img # always take residual from raw data
        # sersic_residual -= self.cutoutdata._bkg_level 
        
        # update the total residual
        psf_modelimg = getattr(self.cutoutdata,'psf_modelimg',0)
        residual_img = self.cutoutdata._rawdata - psf_modelimg - bestfit_img
        residual_masked = residual_img.copy() # no sigma clipping
        
        # save all information in cutoutdata
        # sky_model = getattr(self.cutoutdata,'sky_model',0)        
        self.cutoutdata.sersic_params_physical = bestfit_sersic_params_physical
        self.cutoutdata.sersic_params = result.x
        self.cutoutdata.sersic_modelimg = bestfit_img
        self.cutoutdata.sersic_residual = sersic_residual #- sky_model
        self.cutoutdata.residual = residual_img
        self.cutoutdata.residual_masked = residual_masked
        self.model.x0 = result.x
        save_bestfit_params(self.cutoutdata,bestfit_sersic_params_physical)
        
        return self.cutoutdata

class ModelScaleFitter(ModelFitter):
    def __init__(self,model,cutoutdata,base_params=None,
                 **kwargs):
        if base_params is None:
            raise ValueWarning('ModelScaleFitter requires the base parameter to be initialized.')
        super().__init__(model,cutoutdata,**kwargs)
        self.base_params = base_params
        self.bounds_physical = self.model.get_bounds()

    def scale_params(self,flux_scale):
        ''' a helper function to scale the parameters based on the flux scale '''
        flux_scale = np.squeeze(flux_scale)
        scaled_params = self.base_params.copy()
        for param_name in self.model.free_params:
            if 'amplitude' in param_name:
                idx = self.model.free_params.index(param_name)
                scaled_params[idx] *= flux_scale
        return scaled_params

    def calc_chi2(self,flux_scale,
                  iterinfo='',print_val=False,chi2_min_allowed=1e-10):
        
        scaled_modelparams = self.scale_params(flux_scale)  
        
        # parameter sanity check
        condition_func = getattr(self.model,'condition_func',False)
        if condition_func:
            if condition_func(scaled_modelparams) == False:
                return np.inf
        
        # evaluate model
        model_img = self.eval_model(scaled_modelparams)
        
        # sanity check
        if np.isfinite(model_img).sum() == 0:
            return np.inf
        
        if self.err is None:
            chi2 = np.nansum((self.data - model_img)**2)
        else:
            chi2 = np.nansum(((self.data - model_img)/self.err)**2)
        chi2 /= np.isfinite(self.data).sum()
        if chi2 <= chi2_min_allowed:
            return np.inf
        
        if print_val:
            print(f'\r {chi2:.4e} {iterinfo}   ',end='',flush=True)
        return chi2
    
    def fit(self,method='iterative_NM',fit_to='data',**kwargs):
        self.data = getattr(self.cutoutdata,fit_to)
        self.err  = getattr(self.cutoutdata,fit_to+'_error',None)
        
        # pre-compute the bounds
        self.bounds = [[0,10]]
        if not hasattr(self.model,'x0'):
            self.model.x0 = [1]
        
        # run fitting
        if method == 'iterative_NM':
            iNM_kwargs = dict(rtol_init=1e-3,rtol_iter=1e-3,
                                      rtol_convergence=1e-6,xrtol=1,max_iter=10)
            iNM_kwargs.update(kwargs)
            result,success = iterative_NM(self.calc_chi2, (), self.model.x0, 
                                        self.bounds,**iNM_kwargs)
        else:
            raise ValueError('method not recognized')
        
        # parse results
        scaled_modelparams = self.scale_params(result.x)  
        bestfit_sersic_params_physical = dict(zip(self.model.free_params,
                                                self.unstandardize_params(scaled_modelparams)))
        bestfit_img = self.eval_model(scaled_modelparams)
        sersic_residual = self.cutoutdata._rawdata - bestfit_img # always take residual from raw data
        # sersic_residual -= self.cutoutdata._bkg_level 
        
        # update the total residual
        psf_modelimg = getattr(self.cutoutdata,'psf_modelimg',0)
        residual_img = self.cutoutdata._rawdata - psf_modelimg - bestfit_img
        residual_masked = residual_img.copy() # no sigma clipping
        
        # save all information in cutoutdata
        # sky_model = getattr(self.cutoutdata,'sky_model',0)        
        self.cutoutdata.sersic_params_physical = bestfit_sersic_params_physical
        self.cutoutdata.sersic_params = scaled_modelparams
        self.cutoutdata.sersic_modelimg = bestfit_img
        self.cutoutdata.sersic_residual = sersic_residual #- sky_model
        self.cutoutdata.residual = residual_img
        self.cutoutdata.residual_masked = residual_masked
        self.model.x0 = result.x
        save_bestfit_params(self.cutoutdata,bestfit_sersic_params_physical)
        return self.cutoutdata
    
def triple_annealing(func,args=(),x0=None,bounds=None,max_iter=2,**kwargs):
    # fit (INM -> loop(dual annealing -> INM))
    result,success = iterative_NM(func, args, x0, bounds,
                                rtol_init=1e-3,rtol_iter=1e-4,
                                rtol_convergence=1e-6,xrtol=1,max_iter=5)
    for i in range(max_iter):
        np.random.seed(i)
        print('\n performing dual annealing global optimization...')
        result = dual_annealing(func,x0=result.x,args=args,
                                bounds = bounds,
                                maxiter = 5+i,
                                initial_temp = 10/(i+1),
                                minimizer_kwargs = dict(method='L-BFGS-B',
                                                        bounds=bounds,
                                                        options=dict(eps=1e-4/(i+1),
                                                                    maxfun=1000)))
        
        print('\n performing iterative Nelder-Mead optimization...')
        result, success = iterative_NM(func,x0=result.x,args=args,
                                    bounds = bounds,
                                    max_iter = 5-i,
                                    xrtol=1e-2/(i+1),
                                    rtol_iter=1e-9/(i+1),
                                    rtol_convergence=1e-10)
        if success:
            break
    return result,success

def iterative_NM(func,args,x0,bounds,
                rtol_init=1e-3,rtol_iter=1e-4,
                rtol_convergence=1e-6,xrtol=1,max_iter=20,
                maxfev_eachiter=100,
                progress=None,
                progress_text='Running iNM...',**kwargs):
    ''' Iterative Nelder-Mead minimization.
    The original implementation by Scipy tends to miss the global minimum.
    Rather than setting the tolerance to be small, the success rate tends to be higher
    when the tolerance is set to be larger and the minimization is run multiple times.
    '''

    ##  initial fit
    # set starting value
    # print('',end='',flush=True)
    chi2_init = func(x0,*args)
    chi2_vals = [chi2_init]
    convergence = False
    
    # set tolerance
    xatol = xrtol* max(np.abs(x0))
    fatol = rtol_init * func(x0,*args)
    
    # run minimization
    result = minimize(func,x0,bounds=bounds,
                    method='Nelder-Mead',
                    args = (*args,'(iter=0)'),
                    options = dict(maxfev=maxfev_eachiter,fatol=fatol,xatol=xatol))
    chi2_vals.append(result.fun)

    # run minimization multiple times
    if progress is not None:
        progress_task = progress.add_task(progress_text, total=max_iter)
    for i in range(max_iter):
        x0 = result.x
        xatol = xrtol* max(np.abs(x0))
        fatol = rtol_iter * func(x0,*args)
        result = minimize(func,x0,bounds=bounds,
                        method='Nelder-Mead',
                        args = (*args,f'(iter={i+1}: fatol={fatol:.2e})'),
                        options = dict(maxfev=maxfev_eachiter,fatol=fatol,xatol=xatol))
        chi2_vals.append(result.fun)
        if progress is not None:
            progress.update(progress_task, advance=1, refresh=True)

        if np.allclose(chi2_vals[-2:],chi2_vals[-1],rtol=rtol_convergence):
            if np.isfinite(chi2_vals[-1]):
                convergence = True
                # print('\nIterative Nelder-Mead method Converged')
                break
    if progress is not None:
        progress.remove_task(progress_task)
    return result, convergence

def save_bestfit_params(cutoutdata,bestfit_sersic_params_physical,):
    for key,val in bestfit_sersic_params_physical.items():
        setattr(cutoutdata,key,val)
        
        
#######################
# photometry functions
#######################
class ProfileStats():
    def __init__(self,):
        pass
    
    def plot_profile():
        pass

def make_annulus(bestfit_params_physical,a_in,a_out,multi_sersic_index=0):
    ''' create annulus '''
    if 'r_eff' in bestfit_params_physical.keys():
        suffix = ''
    else:
        suffix = '_'+str(multi_sersic_index)
    x_0 = bestfit_params_physical['x_0']
    y_0 = bestfit_params_physical['y_0']
    ellip_disk = bestfit_params_physical['ellip'+suffix]
    theta_disk = bestfit_params_physical['theta'+suffix]
    b_out = (1 - ellip_disk) * a_out
    annulus = EllipticalAnnulus((x_0, y_0), a_in, a_out, b_out, theta=theta_disk)
    return annulus

def make_aperture(bestfit_params_physical,a,multi_sersic_index=0):
    ''' create aperture '''
    if 'r_eff' in bestfit_params_physical.keys():
        suffix = ''
    else:
        suffix = '_'+str(multi_sersic_index)
    x_0 = bestfit_params_physical['x_0']
    y_0 = bestfit_params_physical['y_0']
    ellip_disk = bestfit_params_physical['ellip'+suffix]
    theta_disk = bestfit_params_physical['theta'+suffix]
    b = (1 - ellip_disk) * a
    aperture = EllipticalAperture((x_0, y_0), a, b, theta=theta_disk)
    return aperture

def profile_stats(cutoutdata,fit_to='psf_sub_data',
                  sersic_params = None,
                  max_size_relative=6,
                  interval_relative=0.1,plot=True,
                  radius_param_name='r_eff_0',
                  multi_sersic_index=0):
    if sersic_params is None:
        sersic_params = cutoutdata.sersic_params_physical
        
    if 'r_eff' not in sersic_params.keys():
        r_eff = sersic_params[radius_param_name]
    else:
        r_eff = sersic_params['r_eff']
    data = getattr(cutoutdata,fit_to)
    shape = data.shape
    
    # take rolling stats with elliptical annulus
    a_vals = np.arange(1,r_eff*max_size_relative,interval_relative*r_eff)
    data_filled = data.copy()
    error_filled = getattr(cutoutdata,fit_to+'_error',None)
    if error_filled is None:
        sphot.info(fit_to+'_error not available: using psf_sub_data_error')
        error_filled = getattr(cutoutdata,'psf_sub_data_error',None)
    if error_filled is not None:
        error_filled = error_filled.copy()
    a_centers = []
    counts_std = []
    counts_mean = []
    counts_sum = []
    
    # innermost aperture
    aperture = make_aperture(sersic_params,a_vals[0],
                             multi_sersic_index=multi_sersic_index)
    aper_stats = ApertureStats(data,aperture,sigma_clip=None)
    a_centers.append(a_vals[0]/2)
    counts_std.append(aper_stats.std)
    counts_mean.append(aper_stats.mean)
    counts_sum.append(aper_stats.sum)
    
    # annulus
    for i in range(len(a_vals)-1):
        # do aperture statistics
        a_in,a_out = a_vals[i],a_vals[i+1]
        a_centers.append((a_in+a_out)/2)
        annulus = make_annulus(sersic_params,a_in,a_out,
                               multi_sersic_index=multi_sersic_index)
        aper_stats = ApertureStats(data,annulus,sigma_clip=None)
        counts_std.append(aper_stats.std)
        counts_mean.append(aper_stats.mean)
        counts_sum.append(aper_stats.sum)
        
        # replace NaNs with mean counts if it's just a few pixels
        mask = annulus.to_mask(method='center')
        mask_img = mask.to_image(shape).astype(bool)
        s = mask_img & np.isnan(data_filled)
        if s.sum() <= 0.2 * mask_img.sum():
            data_filled[s] = aper_stats.mean
        else:
            data_filled[s] = cutoutdata.sersic_modelimg[s]
        if error_filled is not None:
            error_filled[s] = np.maximum(aper_stats.std,error_filled[s])      
            
    # replace any NaN values within 1-reff with Sersic modelimg
    aperture = make_aperture(sersic_params,r_eff,
                            multi_sersic_index=multi_sersic_index)
    mask = aperture.to_mask(method='center')
    mask_img = mask.to_image(shape).astype(bool)
    s = mask_img & np.isnan(data_filled)
    if s.sum() > 0:
        data_filled[s] = cutoutdata.sersic_modelimg[s]
              
    stats = ProfileStats()
    stats.sersic_params = sersic_params
    stats.r_eff = r_eff
    stats.a_vals =  np.array(a_centers)
    stats.flux_sum = np.array(counts_sum)
    stats.flux_contained = np.cumsum(counts_sum)
    stats.flux_mean = np.array(counts_mean)
    stats.flux_std = np.array(counts_std)
    stats.data_filled = data_filled
    if error_filled is not None:
        stats.error_filled = error_filled
    return stats
    
def do_aperture_photometry(stats,
                           aperture_size=2,
                           annulus_size = [3,6],
                           plot=True,
                           multi_sersic_index=0,
                           ax=None):
    ''' perform aperture photometry '''

    # run photometry
    annulus = make_annulus(stats.sersic_params,
                           annulus_size[0]*stats.r_eff,
                           annulus_size[1]*stats.r_eff,
                           multi_sersic_index=multi_sersic_index)
    annulus_stats = ApertureStats(stats.data_filled,annulus,sigma_clip=None)
    annulus_mean,annulus_std = annulus_stats.mean, annulus_stats.std
    
    aperture = make_aperture(stats.sersic_params,
                             aperture_size*stats.r_eff,
                             multi_sersic_index=multi_sersic_index)
    phot = aperture_photometry(stats.data_filled-annulus_mean,
                               aperture,error=stats.error_filled)
    counts,counts_error = phot['aperture_sum'][0], phot['aperture_sum_err'][0]
        
    if plot:
        no_ax = False
        if ax is None:
            no_ax = True
            fig,ax = plt.subplots(1,1,figsize=(5,5))
        astroplot(stats.data_filled,ax=ax)
        aperture.plot(ax=ax,color='magenta')
        annulus.plot(ax=ax,color='lawngreen')
        if no_ax:
            plt.show()
    return counts,counts_error,annulus_mean


# def correct_flux_fraction()
