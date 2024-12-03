import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


from astropy.convolution import convolve
from astropy.nddata import block_reduce, Cutout2D
from astropy import units as u
from photutils.psf.matching import TopHatWindow, create_matching_kernel
from photutils.aperture import EllipticalAperture, aperture_photometry

from copy import deepcopy
from csaps import csaps
from skimage import measure
import cv2
from .plotting import astroplot
from .logging import logger

def aperture_routine(galaxy,petro=0.5,center_mask=3.5,plot=True,
                     isophot_base_filter = None,
                     isophot_frac_min = 0.05,
                     isophot_frac_max = 0.95,
                     fit_isophot_to='sersic_modelimg',
                     measure_on = 'psf_sub_data',
                     error_on = 'psf_sub_data_error',
                     measure_sky_on='residual_masked',
                     fill_max_nan_frac=0.7,
                     fill_replace_with='sersic_modelimg',
                     correct_PSF = True,
                     PSF_corr_base_filter='F090W',
                     custom_aperture=None,
                     **kwargs):
    ''' Run aperture photometry. 
    '''
    logger.info(f'*** running aperture photometry ***')
    logger.info(f'- Petrosian index: {petro}')
    logger.info(f'- Using data: {measure_on}')
    logger.info(f'- Using error: {error_on}')
    logger.info(f'- Using sky: {measure_sky_on}')
    # 1. fill NaNs using isophot
    if isophot_base_filter is not None:
        logger.info('Filling NaNs using isophot...')
        iso_apers = IsoPhotApertures(galaxy.images[isophot_base_filter])
        iso_apers.create_apertures(fit_to=fit_isophot_to,
                                frac_enc=np.linspace(isophot_frac_min,isophot_frac_max,100))
        fill_nans(galaxy,
                iso_apers.apertures,
                apply_to=[measure_on,error_on,measure_sky_on],
                max_nan_frac=fill_max_nan_frac,
                replace_with=fill_replace_with)
    else:
        logger.info('Skipping NaN filling using isophot...')
        for filt in galaxy.filters:
            cutoutdata = galaxy.images[filt]
            setattr(cutoutdata,measure_on+'_filled',getattr(cutoutdata,measure_on,None))
            setattr(cutoutdata,error_on+'_filled',getattr(cutoutdata,error_on,None))
            setattr(cutoutdata,measure_sky_on+'_filled',getattr(cutoutdata,measure_sky_on,None))

    # 2. define aperture
    if custom_aperture is None:
        logger.info('Calculating aperture for photometry...')
        iso_apers.measure_flux(measure_on= measure_on + '_filled')
        iso_apers.calc_petrosian_indices(bin_size=2)
        aper_sci = iso_apers.get_aper_at(petro=petro)
        if plot:
            iso_apers.plot()
    else:
        logger.info('Using custom aperture...')
        aper_sci = custom_aperture
    
    # 3. run aperture photometry
    logger.info('Running aperture photometry...')
    for filt in galaxy.filters:
        cutoutdata = galaxy.images[filt]
        aperphot = CutoutDataPhotometry(cutoutdata,aper_sci)
        aperphot.measure_flux(measure_on= measure_on + '_filled',
                                error_on= error_on + '_filled')
        aperphot.measure_sky(measure_on = measure_sky_on + '_filled',
                            center_mask = center_mask,
                            mode='grid')
        aperphot.calc_mag()
        if plot:
            aperphot.plot()
        cutoutdata.aper_petro = petro
        cutoutdata.mag_raw = aperphot.magAB
        cutoutdata.mag_raw_err = aperphot.magAB_err
    if not correct_PSF:
        return aper_sci
    
    # 4. PSF correction
    logger.info('Running PSF correction...')
    data_to_blur = [measure_on+'_filled',
                    error_on+'_filled',
                    measure_sky_on+'_filled']

    for filt in galaxy.filters:
        # convolve
        blurring_kernel,_,_ = prepare_blurring_kernel(
            galaxy,
            PSF_corr_base_filter,
            filt)
        cutoutdata = galaxy.images[filt]
        _cutoutdata = deepcopy(galaxy.images[PSF_corr_base_filter])
        for attr in data_to_blur:
            data_convolved = convolve(
                getattr(_cutoutdata,attr),
                blurring_kernel,
                preserve_nan=False if 'residual' in attr else True)
            if '_error_filled' in attr:
                new_attr = attr.replace('_error_filled','_filled_blurred_error')
            elif '_error' in attr:
                new_attr = attr.replace('_error','_blurred_error')
            else:
                new_attr = attr+'_blurred'
            setattr(_cutoutdata,new_attr,data_convolved)
            
        # run photometry and measure the difference
        aperphot = CutoutDataPhotometry(_cutoutdata,aper_sci)
        aperphot.measure_flux(measure_on=measure_on+'_filled_blurred',
                              error_on=error_on.replace('_error','_filled_blurred_error'))
        aperphot.measure_sky(measure_on=measure_sky_on+'_filled_blurred',
                            center_mask = center_mask,
                            mode='grid')
        aperphot.calc_mag()
        if plot:
            aperphot.plot()
        cutoutdata.dmag_PSFcorr = aperphot.magAB - _cutoutdata.mag_raw
        cutoutdata.dmag_PSFcorr_err = np.sqrt(aperphot.magAB_err**2 + _cutoutdata.mag_raw_err**2)
    
    for filt in galaxy.filters:
        cutoutdata = galaxy.images[filt]
        cutoutdata.dmag_PSFcorr -= galaxy.images[PSF_corr_base_filter].dmag_PSFcorr
        cutoutdata.mag_corr = cutoutdata.mag_raw - cutoutdata.dmag_PSFcorr
        cutoutdata.mag_corr_err = np.sqrt(cutoutdata.mag_raw_err**2 + cutoutdata.dmag_PSFcorr_err**2)
        
    return aper_sci

def prepare_blurring_kernel(galaxy,filt,blur_to,
                            window=TopHatWindow(0.35)):
    ''' Create a PSF-matching kernel '''
    cutoutdata = galaxy.images[filt]
    cutoutdata_blurbase = galaxy.images[blur_to]

    psf = cutoutdata.psf
    psf_base = cutoutdata_blurbase.psf

    assert cutoutdata.psf_oversample == cutoutdata_blurbase.psf_oversample, 'PSF oversample must be the same'
    os = cutoutdata.psf_oversample
    
    # Set the target size to the smaller one
    # ensure the new shape is an integer multiple of oversample 
    # AND the size is odd after downsampling
    _target_size = min(psf.shape[0],psf_base.shape[0])
    target_size = os*((_target_size//os)//2*2-1)
        
    # crop the PSF to the target size
    psf_cropped = Cutout2D(psf,np.array(psf.shape)/2,[target_size,target_size]).data
    psf_base_cropped = Cutout2D(psf_base,np.array(psf_base.shape)/2,[target_size,target_size]).data

    # downsample the PSF
    psf_cropped_ds = block_reduce(psf_cropped,os)
    psf_base_cropped_ds = block_reduce(psf_base_cropped,os)

    # Calculate the blurring kernel
    psf_matching_kernel = create_matching_kernel(psf_cropped_ds, psf_base_cropped_ds, window=window)

    return psf_matching_kernel, psf_cropped_ds, psf_base_cropped_ds

def fill_nans(galaxy,apertures,
              apply_to='psf_sub_data',
              replace_with='sersic_modelimg',
              max_nan_frac=0.5):
    ''' fill in NaN values in the data using median of the finite pixels within annulus defined by consequtive apertures. 
    Creates a new attribute with '_filled' suffix.
    
    Args:
        galaxy (MultiBandCutout): the galaxy object
        apertures (list): a list of apertures to define the annulus
        apply_to (str or list of str): the attributes of the cutoutdata to fill NaNs in
        replace_with (str): the attribute to use as a reference for filling NaNs if the fraction of NaNs is too high
    '''
    raise_replace_warning = False
    apply_to = np.atleast_1d(apply_to)
    
    for apply_attr in apply_to:
        for filt in galaxy.filters:
            cutoutdata = galaxy.images[filt]
            data = getattr(cutoutdata,apply_attr).copy()
            data_filled = data.copy()
            
            for i in range(len(apertures)):
                if i == 0:
                    aper = apertures[i]
                    mask = aper.to_mask(method='center').to_image(data.shape).astype(bool)
                    s = ~np.isfinite(data) & mask
                    if s.sum() == 0:
                        continue
                    # elif s.sum() <= max_nan_frac * mask.sum():
                    data_filled[s] = getattr(cutoutdata,replace_with)[s].copy()
                    logger.warning('NaN pixels exists near the center. Replacing with the reference image')
                    # else:
                        # logger.error('NaN pixels near the center...')
                else:                    
                    aper_outer = apertures[i]
                    aper_inner = apertures[i-1]
                    mask_outer = aper_outer.to_mask(method='center').to_image(data.shape).astype(bool)
                    mask_inner = aper_inner.to_mask(method='center').to_image(data.shape).astype(bool)
                    mask_img = mask_outer & (~mask_inner)
                    median_val = np.nanmedian(data[mask_img])
                    s = mask_img & (~np.isfinite(data))
                    if s.sum() <= max_nan_frac * mask_img.sum():
                        data_filled[s] = median_val
                    else:
                        data_filled[s] = getattr(cutoutdata,replace_with)[s].copy()
                        raise_replace_warning = True
                if i == len(apertures)-1: 
                    aper = apertures[i]
                    mask = aper.to_mask(method='center').to_image(data.shape).astype(bool)
                    s = ~np.isfinite(data) & ~mask # outside the last aperture
                    if s.sum() == 0:
                        continue
                    elif s.sum() <= max_nan_frac * mask.sum():
                        median_val = np.nanmedian(data[~mask])
                        data_filled[s] = median_val
                        # logger.warning('filling NaN pixels outside the largest aperture with the median value')
                    else:
                        logger.error('too many NaN pixels outside the largest aperture. Skipping...')

            setattr(cutoutdata,apply_attr+'_filled',data_filled)
    if raise_replace_warning:
        logger.warning('some NaN values were replaced with the reference image')
    # return galaxy

class IsoPhotApertures():
    def __init__(self,cutoutdata):
        self.cutoutdata = cutoutdata
        
    def create_apertures(self,
                        fit_to='sersic_modelimg',
                        frac_enc=np.arange(0.1,0.9,100)):
        ''' create isophotal apertures, equally spaced in the surface brightness levels
        
        Args:
            fit_to (str): the attribute of the cutoutdata to fit the isophotal apertures to
            frac_enc (list): approx. fractional levels of enclosed flux within aperture
        '''
        # prepare data for aperture analysis
        Z = getattr(self.cutoutdata,fit_to).copy()

        # Set the fractional level for convenience
        Z_flatten = np.sort(Z.flatten())[::-1]
        X,Y = np.meshgrid(np.arange(Z.shape[1]),np.arange(Z.shape[0]))
        Z_cumsum = np.cumsum(Z_flatten)
        Z_cumsum /= Z_cumsum.max()
        inverse_cumsum_interp = interp1d(Z_cumsum,Z_flatten)

        # Find the contour at the specified levels
        frac_enc_passed = []
        apertures = []
        semi_major_axes = []
        areas = []
        for frac in frac_enc:
            try:
                lvl = inverse_cumsum_interp(frac)
                contours = measure.find_contours(Z, lvl)
                contours = np.squeeze(contours)
                contours = np.flip(contours,axis=1)
                ellip = cv2.fitEllipse(contours.astype(np.float32))
                (x0,y0),(ax_major,ax_minor),angle = ellip
                aperture_kwargs = dict(positions=(x0,y0),a=ax_major/2,b=ax_minor/2,theta=np.radians(angle))
                aperture = EllipticalAperture(**aperture_kwargs)
                frac_enc_passed.append(frac)
                apertures.append(aperture)
                semi_major_axes.append(ax_major/2)
                areas.append(aperture.area)
            except Exception:
                continue
        self.frac_enc = frac_enc_passed
        self.apertures = apertures
        self.areas = np.array(areas)
        self.semi_major_axes = np.array(semi_major_axes)    

    def fill_nans(self,apply_to='psf_sub_data',
                  fill_method='median',
                  max_nan_frac=0.5,
                  add_noise = False):
        ''' fill in NaN values in the data. creates a new attribute with '_filled' suffix
        
        Args:
            fill_method (str): 'noise' or 'median' -- how to fill in NaN values in the data
        '''
        # replace NaNs with mean counts if it's just a few pixels
        data = getattr(self.cutoutdata,apply_to).copy()
        data_filled = data.copy()
        
        for i in range(len(self.apertures)):
            if i == 0:
                continue
            aper_outer = self.apertures[i]
            aper_inner = self.apertures[i-1]
            mask_outer = aper_outer.to_mask(method='center').to_image(data.shape).astype(bool)
            mask_inner = aper_inner.to_mask(method='center').to_image(data.shape).astype(bool)
            mask_img = mask_outer & (~mask_inner)
            median_val = np.nanmedian(data[mask_img])
            std_val    = np.nanstd(data[mask_img])
            s = mask_img & (~np.isfinite(data))
            if s.sum() <= max_nan_frac * mask_img.sum():
                if add_noise:
                    fill_val = np.random.normal(median_val,std_val,s.sum())
                else:
                    fill_val = median_val
                data_filled[s] = fill_val
            else:
                # TODO: add replace_with option?
                pass
        setattr(self.cutoutdata,apply_to+'_filled',data_filled)

    def measure_flux(self,measure_on='psf_sub_data'):
        ''' 
        perform aperture photometry using pre-constructed isophotal apertures.
        
        Args:
            measure_on (str): the attribute of the cutoutdata to measure the flux on
        '''
        data = getattr(self.cutoutdata,measure_on).copy()
        data_err = getattr(self.cutoutdata,measure_on+'_error',None)
        
        fluxes_enclosed = []
        fluxes_enclosed_err = []
        for aper in self.apertures:
            phot = aperture_photometry(data,aper,error=data_err)
            fluxes_enclosed.append(phot['aperture_sum'].value[0])
            if data_err is not None:
                fluxes_enclosed_err.append(phot['aperture_sum_err'].value[0])
            else:
                fluxes_enclosed_err.append(np.nan)
        self.flux_data = data
        self.fluxes_enclosed = np.array(fluxes_enclosed)
        self.fluxes_enclosed_err = np.array(fluxes_enclosed_err)
        
    def calc_petrosian_indices(self,bin_size=2):
        if bin_size%2 != 0:
            return ValueError('bin size must be an integer multiple of 2')
        # calculate necessary quantities
        flux_in_annulus = self.fluxes_enclosed[bin_size:] - self.fluxes_enclosed[:-bin_size]
        area_annulus = self.areas[bin_size:] - self.areas[:-bin_size]
        sb_in_annulus = flux_in_annulus / area_annulus
    
        # calculate petrosian indices
        s_ = np.s_[int(bin_size/2):-int(bin_size/2)]
        petro_idx = np.ones_like(self.fluxes_enclosed) * np.nan
        petro_idx[s_] = sb_in_annulus * self.areas[s_] / self.fluxes_enclosed[s_]
        self.petro_s_ = s_ # slice at which petrosian indices are calculated
        self.petro_idx = petro_idx
        self.petrosian_indices = self.petro_idx
        
    def plot(self,x_attr='semi_major_axes'):
        xdata = getattr(self,x_attr)   
        ydata = self.petro_idx

        if not hasattr(self,'petro_idx_interp'):   
            # interpolate petrosian indices
            # note: normalizedsmooth=True assures that xdata and y_interp are invariant
            s = np.isfinite(xdata) & np.isfinite(ydata)
            interp_func = csaps(xdata[s],ydata[s],normalizedsmooth=True)
            y_interp = interp_func(xdata,extrapolate=False)
        else:
            y_interp = self.petro_idx_interp

        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
        astroplot(self.flux_data,ax=ax1)
        for aper in self.apertures:
            aper.plot(ax=ax1,color='magenta',lw=1,alpha=0.5)
        ax2.scatter(xdata,ydata,c='k',s=10,label='raw measurement')
        ax2.plot(xdata,y_interp,c='r',label='interpolated')    
        
        # some formatting            
        ax1.set_xlabel('Data',fontsize=13)
        ax2.set_xlim(0,)
        ax2.set_ylabel('Petrosian index',fontsize=13)
        ax2.set_xlabel(x_attr,fontsize=13)
        ax2.legend(frameon=False,loc='upper right',fontsize=13)
        ax2.tick_params(direction='in')
        # plt.show()
        
    def get_aper_at(self,petro=None,flux_frac=None,x_attr='semi_major_axes'):


        if not hasattr(self,'petro_idx_interp'):   
            # interpolate petrosian indices
            # note: normalizedsmooth=True assures that xdata and y_interp are invariant
            xdata = getattr(self,x_attr)   
            ydata = self.petro_idx
            s = np.isfinite(xdata) & np.isfinite(ydata)
            interp_func = csaps(xdata[s],ydata[s],normalizedsmooth=True)
            y_interp = interp_func(xdata,extrapolate=False)
            self.petro_idx_interp = y_interp
            
        if petro is not None:
            idx = np.nanargmin(np.abs(self.petro_idx_interp - petro))
        elif sb is not None:
            idx = np.nanargmin(np.abs(self.flux_frac_levels - flux_frac))
        else:
            raise ValueError('either petro or flux_frac must be provided')
        return self.apertures[idx]
    
class CutoutDataPhotometry():
    def __init__(self,cutoutdata,aperture):
        self.cutoutdata = cutoutdata
        self.aperture = aperture
        
    def measure_sky(self,measure_on='residual_masked',N_apers=200,center_mask=3,
                    mode='random'):
        ''' Estimate the uncertainty in aperture photometry using the background and moving aperture 
        
        Args:
            measure_on (str): the attribute of the galaxy to measure the sky on
            center_mask (float): the size of the mask near the center to avoid when measuring sky, in units of aperture radius
            mode (str): 'random' or 'grid' -- whether to sample the sky apertures randomly or in a grid
        '''
        data_sky = getattr(self.cutoutdata,measure_on).copy()
        err = getattr(self.cutoutdata,measure_on+'_error',None)
        
        # prepare locations
        if mode == 'random':
            x0_vals = np.random.uniform(0,data_sky.shape[1],N_apers)
            y0_vals = np.random.uniform(0,data_sky.shape[0],N_apers)
            theta_vals = np.random.uniform(0,np.pi,N_apers)
        elif mode == 'grid':
            nrows = np.ceil(np.sqrt(N_apers)).astype(int)
            _x0_vals = np.linspace(0,data_sky.shape[1],nrows)
            _y0_vals = np.linspace(0,data_sky.shape[0],nrows)
            x0_vals,y0_vals = np.meshgrid(_x0_vals,_y0_vals)
            x0_vals = x0_vals.flatten()
            y0_vals = y0_vals.flatten()
            theta_vals = np.random.uniform(0,np.pi,int(nrows**2))
        
        mean_counts = []
        aperture_sum_vals = []
        sky_apertures = []
        for x0,y0,theta in zip(x0_vals,y0_vals,theta_vals):
            aperture_sky = self.aperture.copy()
            aperture_mask = aperture_sky.copy()
            aperture_mask.a *= center_mask
            aperture_mask.b *= center_mask
            aperture_mask_img = aperture_mask.to_mask(method='center').to_image(data_sky.shape).astype(bool)
            if int(x0) >= data_sky.shape[1] or int(y0) >= data_sky.shape[0]:
                continue
            if aperture_mask_img[int(y0),int(x0)]:
                # avoid center -- this area can be biased by Sersic profile fit
                continue
            
            # update aperture and do photometry
            aperture_sky.positions = np.array([x0,y0])
            aperture_sky.theta = theta
            
            # check if every pixel is contained in the aperture
            aper_mask = aperture_sky.to_mask(method='exact')
            aper_cutout = aper_mask.cutout(data_sky,fill_value=np.nan)
            if not np.isfinite(aper_cutout.sum()):
                continue
            
            phot = aperture_photometry(data_sky,aperture_sky,error=err)
            if np.isfinite(phot['aperture_sum'].value[0]):
                aperture_sum_vals.append(phot['aperture_sum'].value[0])
                sky_apertures.append(aperture_sky)
        self.data_sky = data_sky
        self.sky_apertures = sky_apertures
        self.sky_values = np.array(aperture_sum_vals)
        self.sky_mean = self.sky_values.mean()
        self.sky_std = self.sky_values.std()
        
    def measure_flux(self,measure_on='psf_sub_data',error_on=None):
        ''' perform aperture photometry using pre-constructed isophotal apertures '''
        data = getattr(self.cutoutdata,measure_on).copy()
        if error_on is not None:
            data_err = getattr(self.cutoutdata,error_on).copy()
        else:
            data_err = getattr(self.cutoutdata,measure_on+'_error',None)
        phot = aperture_photometry(data,self.aperture,error=data_err)
        self.flux = phot['aperture_sum'].value[0]
        self.data_flux = data
        if data_err is not None:
            self.flux_err = phot['aperture_sum_err'].value[0]      
        else:
            self.flux_err = np.nan          
                        
    def calc_mag(self,pixel_scale=0.03):
        counts_Mjy_per_Sr = self.flux - self.sky_mean
        errors_Mjy_per_Sr = np.sqrt(self.flux_err**2 + self.sky_std**2)
        
        PIXAR_SR = ((pixel_scale*u.arcsec)**2).to(u.sr).value
        self.magAB = -6.10 - 2.5 *np.log10(counts_Mjy_per_Sr*PIXAR_SR)
        if np.isfinite(errors_Mjy_per_Sr):
            self.magAB_err = 2.5/np.log(10) * errors_Mjy_per_Sr/counts_Mjy_per_Sr
        else:
            self.magAB_err = np.nan
            
    def plot(self):
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
        plt.subplots_adjust(wspace=0.01)
        norm,offset = astroplot(self.data_flux,ax=ax1)
        astroplot(self.data_sky,ax=ax2,norm=norm,offset=offset)
        self.aperture.plot(ax=ax1,color='magenta',lw=2,alpha=0.5)
        for aper in self.sky_apertures:
            aper.plot(ax=ax2,color='yellowgreen',lw=1,alpha=0.5)
        ax1.text(0.05,0.95,f'{self.cutoutdata.filtername}: {self.magAB:.2f} +/- {self.magAB_err:.2f} mag',
                 transform=ax1.transAxes,fontsize=10,bbox=dict(facecolor='white',alpha=0.3,edgecolor='none'),
                 va='top',ha='left')
            
    def __repr__(self):
        return f'{self.cutoutdata.filtername}: {self.magAB:.2f} +/- {self.magAB_err:.2f} mag'