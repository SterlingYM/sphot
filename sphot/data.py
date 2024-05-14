# core.py
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import logging
from tqdm.auto import tqdm
import h5py
import json

from scipy.optimize import minimize, leastsq
from scipy import stats
from scipy.stats import multivariate_normal

from photutils import CircularAperture
from photutils.detection import DAOStarFinder
from photutils.background import MMMBackground, MADStdBackgroundRMS

import astropy.units as u
from astropy.stats import sigma_clip
from astropy.nddata import Cutout2D
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.functional_models import Gaussian2D
from astropy.convolution import convolve, Gaussian2DKernel
from petrofit.modeling import PSFConvolvedModel2D, model_to_image

from .plotting import astroplot, plot_sersicfit_result


def calc_mag(counts_Mjy_per_Sr,errors_Mjy_per_Sr=None,pixel_scale=0.03):
    PIXAR_SR = ((pixel_scale*u.arcsec)**2).to(u.sr).value
    magAB = -6.10 - 2.5 *np.log10(counts_Mjy_per_Sr*PIXAR_SR)
    if errors_Mjy_per_Sr is not None:
        magAB_err = 2.5/np.log(10) * errors_Mjy_per_Sr/counts_Mjy_per_Sr
        return magAB,magAB_err
    return magAB
               
class CutoutData():
    def __init__(self,data=None,psf=None,psf_oversample=None,filtername=None,**kwargs):
        '''
        Args:
            data (2D array): cutout image data
            psf (2D array): PSF data. It can be oversampled (pixel scale could be an integer multiple of the image)
            PSF_oversample (int): oversampling factor of the PSF
            filtername (str): filter name
        '''
        if data is not None:
            self.data = data
            self._rawdata = self.data.copy()
            self.psf = psf
            self.psf_oversample = psf_oversample
            self.filtername = filtername
            for key,val in kwargs.items():
                setattr(self,key,val)
            self.fix_psf()
            self.calc_psf_sigma()
        else:
            # create an empty CutoutData object
            # this is useful when loading from a file
            pass
        
    def fix_psf(self):
        ''' make sure the sum of PSF is 1 '''
        self.psf /= np.sum(self.psf)
    
    def calc_psf_sigma(self):
        ''' estimate Gaussian sigma-equivalent size of the PSF '''
        psf_sigma = _calc_psf_sigma(self.psf,self.psf_oversample)
        self.psf_sigma = psf_sigma   
    
    def blur_psf(self,psf_blurring=None):
        ''' blur the PSF '''
        if psf_blurring is None:
            psf_blurring = self.psf_blurring
        kernel = Gaussian2DKernel(psf_blurring)
        if not hasattr(self,'_psf_raw'):
            self._psf_raw = self.psf.copy()
        self.psf = convolve(self._psf_raw, kernel)
        self.psf_blurring = psf_blurring
        self.fix_psf()
        self.calc_psf_sigma()
    
    def determine_psf_blurring(self):
        ''' determine the best PSF blurring value based on the number of stars detected '''
        psfimg = self._psf_raw.copy()
        _data = get_data_annulus(self._rawdata,
                                 2.5*self.galaxy_size,
                                 plot=False,flatten=False)
        _data = np.ma.masked_invalid(_data)
        
        # prepare background-subtracted data
        mmm_bkg = MMMBackground()
        bkg_level = mmm_bkg(_data)
        data_bksub = _data - bkg_level

        # determine background STD
        bkgrms = MADStdBackgroundRMS()
        bkg_std = bkgrms(data_bksub)
        
        psf_blurring_vals = np.arange(4,12,1)
        psf_sigma_vals = []
        N_stars_found = []
        for psf_blurring in psf_blurring_vals:
            kernel = Gaussian2DKernel(psf_blurring)
            blurred_psf = convolve(psfimg, kernel)
            psf_sigma = _calc_psf_sigma(blurred_psf,
                                        self.psf_oversample)
            daofinder = DAOStarFinder(threshold=bkg_std*2,
                                      fwhm=psf_sigma*2.33,
                                      roundhi=1.0, roundlo=-1.0,
                                      sharplo=0.20, sharphi=1.0)
            star_cat = daofinder.find_stars(data_bksub)
            psf_sigma_vals.append(psf_sigma)
            N_stars_found.append(len(star_cat))
        plt.figure(figsize=(6,4))
        plt.plot(psf_blurring_vals/4,N_stars_found)
        plt.xlabel('sigma of Gaussian blurring kernel [data pix]')
        plt.ylabel('# stars detected')
        
        best_psf_blurring = np.mean(psf_blurring_vals[np.argmax(N_stars_found)])
        plt.axvline(best_psf_blurring/4,c='r',ls=':')
        plt.show()

        self.psf_blurring = best_psf_blurring
    
    def plot(self):
        astroplot(self.data,title=self.filtername)
        
    def init_size_guess(self,sigma_guess=10,center_slack = 0.20,
                        plot=False,sigma_kernel=5,
                        clip_lower_counts_percentile=10):
        '''roughly estimate the effective radius using Gaussian profile.
        Args:
            sigma_guess (float): initial guess for the standard deviation of the Gaussian profile (in pixels)
            center_slack (float): the fraction of the image size (from center) within which the center of the galaxy is expected. Default is 5%
        Returns:
            float: rough estimate of the effective radius (in pixels)
        '''

        from scipy.optimize import curve_fit
        from scipy.ndimage import gaussian_filter
        
        if plot:
            fig,axes = plt.subplots(1,2,figsize=(8,3))
        
        centers,sigmas  = [],[]
        for i,axis in enumerate([1,0]):
            if np.isfinite(self.data).sum() == 0:
                means_smooth = gaussian_filter(self.data,sigma=sigma_kernel).mean(axis=axis)
            else:
                kernel = Gaussian2DKernel(x_stddev=sigma_kernel,y_stddev=sigma_kernel)
                smooth_img = convolve(self.data, kernel)
                s1 = np.any(np.isfinite(self.data),axis=axis)
                means_smooth = np.nanmean(smooth_img[s1],axis=axis)
            
            # fit Gaussian to smoothed counts
            axis_pixels = np.arange(self.data.shape[axis-1])[s1]
            shape = len(axis_pixels)
            gaussian = lambda x,mu,sigma,amp,offset: amp*np.exp(-(x-mu)**2/(2*sigma**2))+offset
            bounds = ([shape/2 - center_slack*shape,0,0,-np.inf],
                      [shape/2 + center_slack*shape,shape,np.inf,np.inf])

            # clip pixels when the counts are too low
            lower_clip = np.nanpercentile(means_smooth,
                                          clip_lower_counts_percentile)
            s2 = means_smooth > lower_clip
            
            x0_guess = axis_pixels[np.argmax(means_smooth)]
            popt,_ = curve_fit(gaussian,axis_pixels[s2],
                               means_smooth[s2],
                            p0=[x0_guess,sigma_guess,means_smooth.max(),0],
                            bounds=bounds)
            centers.append(popt[0])
            sigmas.append(popt[1])
            
            if plot:
                ax = axes[i]
                ax.plot(axis_pixels,means_smooth,c='k')
                ax.plot(axis_pixels,gaussian(axis_pixels,*popt),c='orange',lw=3)
                ax.axvline(popt[0],c='yellowgreen',ls=':',label=f'center={popt[0]:.1f}')
                ax.axvspan(popt[0]-popt[1],popt[0]+popt[1],color='yellowgreen',alpha=0.3,label=f'sigma={popt[1]:.1f}')
                ax.legend(frameon=False)
                ax.set_xlim(0,self.data.shape[axis-1])
                ax.tick_params(direction='in')

                axes[0].set_xlabel('x (pixels)')
                axes[1].set_xlabel('y (pixels)')
                axes[0].set_ylabel('summed counts')
                fig.suptitle('Initial estimation of galaxy shape')
        
        # check if any fit 'failed':
        if np.all(np.array(sigmas) > np.array(self.data.shape)/1.5):
            raise ValueError('looks like cutout is too small or smoothing parameter is not set correctly')
        elif np.any(np.array(sigmas) > np.array(self.data.shape)/1.5):
            s = np.array(sigmas) < np.array(self.data.shape)/1.5
            size_guess = np.array(sigmas)[s][0]
        else:
            size_guess = np.mean(sigmas)
        self.x0_guess = centers[1]
        self.y0_guess = centers[0]
        self.size_guess = size_guess
        
    def remove_bkg(self,bkg_level):
        self._bkg_level = bkg_level
        self.data = self._rawdata.copy() - bkg_level
        
    def perform_bkg_stats(self,plot=False):
        data_annulus = get_data_annulus(self._rawdata,4*self.galaxy_size,plot=plot)
        bkg_mean = np.nanmean(data_annulus)
        bkg_std = np.nanstd(data_annulus)
        self.remove_bkg(bkg_mean) # this updates data internally
        self.data_error = np.ones_like(self.data)*bkg_std
        
        
class MultiBandCutout():
    ''' a container for CutoutData '''
    def __init__(self,name=None,**kwargs):
        self.name = name
        for key,val in kwargs.items():
            setattr(self,key,val)
        if not hasattr(self,'filters'):
            self.filters = []
    
    def add_image(self,filtername,data):
        if hasattr(self,filtername):
            raise ValueError("The specified filter already exists")
        setattr(self,filtername,data)
        self.filters.append(filtername)
        
    @property
    def images(self):
        return dict(zip(self.filters,[getattr(self,filtername) for filtername in self.filters]))
    
    @property
    def image_list(self):
        return [getattr(self,filtername) for filtername in self.filters]  
    
    def plot(self):
        fig,axes = plt.subplots(1,len(self.filters),
                                figsize=(4*len(self.filters),4))
        for filt, ax in zip(self.filters,axes):
            astroplot(self.images[filt].data,ax=ax)
            ax.set_title(filt)
        fig.suptitle(self.name,fontsize=15)    
        
    def crop_in(self,x0,y0,size):
        ''' crop-in and re-generate the MultiBandCutout object
        Args:
            multiband_obj (MultiBandCutout): the object to crop in
            x0,y0 (float): the center of the new image
            size (int): the size of the new image (in pixels)
        Returns:
            galaxy_crop (MultiBandCutout): the cropped object
        '''
        for filtername in self.filters:
            # cutout at the center of the galaxy
            cutoutdata = self.images[filtername]
            image = cutoutdata.data.copy()
            new_cutout = Cutout2D(image, (x0,y0), size)

            # define new CutoutData object    
            cutoutdata.data = new_cutout.data
            cutoutdata._rawdata = new_cutout.data.copy()

    def save(self,filepath):
        ''' save the MultiBandCutout object to a h5 file '''
        with h5py.File(filepath,'w') as f:
            for g_key,g_val in self.__dict__.items():
                if isinstance(g_val,CutoutData):
                    group = f.create_group(g_key)
                    for key,val in g_val.__dict__.items():
                        try:
                            if isinstance(val,dict):
                                json_str = json.dumps(val)
                                dataset = group.create_dataset(key,data=json_str.encode('utf-8'))
                            else:
                                group.create_dataset(key,data=str_to_json(val))
                        except Exception as e:
                            print(f'Error with {key}: {e}')
                            continue
                else:
                    f.create_dataset(g_key,data=str_to_json(g_val))
        print(f'Saved to {filepath}')

def load_h5data(filepath,name,filters,PSFs_dict):
    galaxy = MultiBandCutout(name = name)
    with h5py.File(filepath,'r') as f:
        for filtername in filters:
            image = f[filtername][:]
            psf = PSFs_dict[filtername]
            cutoutdata = CutoutData(data = image, 
                                    psf = psf,
                                    psf_oversample = 4,
                                    filtername = filtername)
            galaxy.add_image(filtername, cutoutdata)
        f.close()
    return galaxy

def get_data_annulus(data,aper_r,plot=True,flatten=True):
    ''' extract flattened data outside the annulus '''
    # apply circular aperture
    aperture = CircularAperture((data.shape[0]/2,data.shape[1]/2), 
                                aper_r)
    aperture_mask = aperture.to_mask(method='center')
    aperture_mask_img = ~aperture_mask.to_image(data.shape).astype(bool)
    data_masked = aperture_mask_img * data
    data_masked[data_masked==0] = np.nan
    data_annulus = data_masked.flatten()[np.isfinite(data_masked.flatten())]
    if plot:
        astroplot(data_masked)
    if not flatten:
        return data_masked
    return data_annulus

def _calc_psf_sigma(data,psf_oversample):
    ''' estimate Gaussian sigma-equivalent size of the PSF '''
    shape = data.shape
    xx,yy = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))
    gaussian2d = Gaussian2D(amplitude = np.nanmax(data),
                            x_mean=shape[1]/2,
                            y_mean=shape[0]/2)
    fitted_model = LevMarLSQFitter()(gaussian2d,xx,yy,data)
    sigma_x = fitted_model.x_stddev.value
    sigma_y = fitted_model.y_stddev.value
    psf_sigma = (sigma_x + sigma_y) / 2 / psf_oversample
    return psf_sigma

def str_to_json(s):
    ''' encode string to json-readable format. 
    This is a helper function for saving h5 files.'''
    if isinstance(s,str):
        return json.dumps(s).encode('utf-8')
    elif isinstance(s,list):
        if isinstance(s[0],str):
            return json.dumps(s).encode('utf-8')
    else:
        return s

def decode_if_bytestring(val):
    ''' reverse of str_to_json '''
    if isinstance(val,bytes):
        return json.loads(val.decode('utf-8'))
    else:
        return val

def read_sphot_h5(filepath):
    ''' load h5 file to sphot objects '''
    galaxy_loaded = MultiBandCutout()
    with h5py.File(filepath,'r') as f:
        for key in f.keys():
            # check if f[key] is a group
            if isinstance(f[key],h5py.Group):
                group = f[key]
                cutoutdata_loaded = CutoutData()
                for attr in group.keys():
                    try:
                        _val = decode_if_bytestring(group[attr][()])
                        setattr(cutoutdata_loaded,attr,_val)
                    except:
                        print('Unable to load: ',_key)
                setattr(galaxy_loaded,key,cutoutdata_loaded)
            else:
                group_val = decode_if_bytestring(f[key][()])
                setattr(galaxy_loaded,key,group_val)
    return galaxy_loaded