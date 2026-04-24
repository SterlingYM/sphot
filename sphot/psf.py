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
        
    def fit(self,fit_to='sersic_residual',**kwargs):
        ''' Perform PSF fitting. 
        This function calls iterative_psf_fitting, which wraps our main function do_psf_photometry. 
        The role of iterative_psf_fitting is to change the detection threshold level so that the PSF fitter does not end up fitting >1000 sources at the same time in a highly crowded field.
        
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
        
        # initialize psf
        blur_psf_dict = config['prep'].get('blur_psf',False)
        blur_psf = blur_psf_dict.get(self.cutoutdata.filtername,False) if isinstance(blur_psf_dict,dict) else blur_psf_dict
        if blur_psf:
            self.update_psf_blur(blur_psf)
            blur_psf_values = np.array(config['psf']['psf_blur_factors']) * blur_psf
        else:
            logger.debug(f'Using original PSF without blurring')
            blur_psf_values = None
            
        x0 = self.cutoutdata.sersic_params_physical['x_0']
        y0 = self.cutoutdata.sersic_params_physical['y_0']
        center_mask_params = [x0,y0,self.psf_sigma*2]
                
        # perform PSF fitting
        # threshold_list = np.arange(th_min,th_max+th_increment,th_increment)[::-1]
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
                psf_blur_values = blur_psf_values,
                psf_blur_func = self.update_psf_blur,
                **kwargs)
            
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
                          psf_blur_values = None,
                          psf_blur_func = None,
                          **kwargs):
    ''' Iteratively run do_psf_photometry() with different threshold levels.
    This function is useful for crowded fields, where a single threshold level may fail. 
    
    Inputs:
        data (2d array): the data to perform PSF photometry.
        psfimg (2d array): the PSF image.
        psf_sigma (float): the HWHM of the PSF. Use FWHM/2
        psf_oversample (int): the oversampling factor of the PSF.
        threshold_list (1d array): the list of threshold levels to try, in background STD.
        center_mask_params (list, optional): [x_center,y_center,mask_r]. If provided, sources within the radius mask_r from (x_center,y_center) will be excluded from the final results. This is useful when the central source is very bright and causes many spurious detections nearby.
        psf_blur_fpsf_blur_valuesactors (list, optional): the list of Gaussian sigma values to convolve the PSF with, in pixel units. If provided, the PSF will be convolved with the Gaussian kernel at the end of the iterative fitting to check if any additional sources can be found/fitted with dfferent PSF widths. If not provided, this additional check will not be performed.
        psf_blur_func (function, optional): the function to convolve the PSF with a Gaussian kernel. The function should take a single argument (the sigma value) and output the convolved PSF. If not provided, this additional check will not be performed.
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

    # optional override: use a fixed threshold for the PSF-blur sweep
    psf_blur_th_override = config['psf'].get('psf_blur_th', None)
    if psf_blur_th_override is not None:
        last_successful_th = psf_blur_th_override

    # additional loop -- repeat PSF subtraction with different PSF widths
    if (psf_blur_values is not None) and (psf_blur_func is not None) and (last_successful_th is not None):
        if progress is not None:
            progress_psf = progress.add_task(f'Fitting at th={last_successful_th:.2f} with sharper and blurrier PSF...', 
                                            total=len(psf_blur_values))
        for psf_blur in psf_blur_values:
            try:
                psf_model,psf_sigma = psf_blur_func(psf_blur)
                psf_results = do_psf_photometry(resid, data_error, bkg_std, 
                                                psf_model, psf_sigma,
                                                th=last_successful_th,**kwargs)
                if progress is not None:
                    progress.update(progress_psf, advance=1, refresh=True)
                if psf_results[0] is None:
                    continue
                else:
                    _phot_result, _resid = psf_results
                    if np.all(~np.isfinite(_resid)):
                        continue
                    # append the results
                    resid = _resid
                    if phot_result is None:
                        phot_result = _phot_result
                    else:
                        phot_result = vstack([phot_result, _phot_result])
            except Exception as e:
                if config['psf']['raise_error']:
                    raise 
                logger.error(f'Skipping PSF fitting with th={th:.2f}, blur={psf_blur}. {str(e)}')
                continue

        if progress is not None:
            progress.remove_task(progress_psf)
        
    # make the residual image without background subtraction
    psf_modelimg_all = data_bksub - resid
    resid_all = data.copy() - psf_modelimg_all.copy()
    return phot_result, resid_all