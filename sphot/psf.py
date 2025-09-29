import numpy as np
import matplotlib.pyplot as plt
from .plotting import astroplot
from tqdm.auto import tqdm
import warnings
from astropy.utils.exceptions import AstropyUserWarning

from scipy.ndimage import gaussian_filter
from astropy.nddata import overlap_slices
from astropy.table import QTable, vstack
from astropy.stats import sigma_clip, SigmaClip, sigma_clipped_stats

from photutils.aperture import CircularAperture, EllipticalAperture
from photutils.psf import (SourceGrouper, IterativePSFPhotometry, 
                           PSFPhotometry, ImagePSF)#FittableImageModel)
from photutils.detection import DAOStarFinder
from photutils.background import (MMMBackground, MADStdBackgroundRMS,
                                  LocalBackground, MedianBackground,
                                  Background2D)
from photutils.datasets.images import make_model_image as _make_model_image

from .data import get_data_annulus
from .logging import logger
from .config import config

class PSFFitter():
    ''' A class to perform PSF fitting. '''
    def __init__(self,cutoutdata):
        self.cutoutdata = cutoutdata
        self.psf_sigma = cutoutdata.psf_sigma
        
            # PSF image to model
        self.psf_model = ImagePSF(
            cutoutdata.psf, flux=1.0,
            x_0=0, y_0=0, 
            oversampling=cutoutdata.psf_oversample, 
            fill_value=0.0
            )   
        
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
            logger.info(f'Convolving (blurring) PSF with sigma={blur_psf} pix')
            self.cutoutdata.blur_psf(blur_psf)
        else:
            logger.info(f'Using original PSF without blurring')
            
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

def make_modelimg(psffitter,shape,psf_shape):
    ''' modified version of photutil's function.
    No background is added.
    
    Args:
        fit_models: list of PSF models
    Returns:
        model_img: rendered model image    
    '''
    
    if isinstance(psffitter, PSFPhotometry):
        psf_model = psffitter.psf_model
        fit_params = psffitter._fit_model_params
        local_bkgs = psffitter.init_params['local_bkg']
    else:
        psf_model = psffitter._psfphot.psf_model
        if psffitter.mode == 'new':
            # collect the fit params and local backgrounds from each
            # iteration
            local_bkgs = []
            for i, psfphot in enumerate(psffitter.fit_results):
                if i == 0:
                    fit_params = psfphot._fit_model_params
                else:
                    fit_params = vstack((fit_params,
                                            psfphot._fit_model_params))
                local_bkgs.append(psfphot.init_params['local_bkg'])

            local_bkgs = _flatten(local_bkgs)
        else:
            # use the fit params and local backgrounds only from the
            # final iteration, which includes all sources
            fit_params = self.fit_results[-1]._fit_model_params
            local_bkgs = self.fit_results[-1].init_params['local_bkg']

        model_params = fit_params

        if include_localbkg:
            # add local_bkg
            model_params = model_params.copy()
            model_params['local_bkg'] = local_bkgs

        try:
            x_name = psf_model.x_name
            y_name = psf_model.y_name
        except AttributeError:
            x_name = 'x_0'
            y_name = 'y_0'

        return _make_model_image(shape, psf_model, model_params,
                                 model_shape=psf_shape,
                                 x_name=x_name, y_name=y_name,
                                 progress_bar=progress_bar)
    return model_img


# def filter_psfphot_results(phot_result):
    
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
    npixfit = phot_result['npixfit']
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

def _update_filter_criteria(phot_result,min_sources=50,target_passing_fraction=0.5,
                            maxiter=10,cfit_increment=0.05,qfit_increment=0.05,
                            **kwargs):
    ''' update filter criteria for PSF photometry.
    Determines the filtering criteria should be loosened based on:
        - the number of sources detected
        - the number of sources that passed the filter
        
    Args:
        min_sources (int): the minimum number of detected sources required for evaluation. No changes will be made is the number of sources detected is smaller than this.
        
    Returns:
        kwargs (dict): the updated kwargs.
    '''
    verbose = kwargs.get('verbose',False)
    N_src = len(phot_result)
    
    # check if we have enough number of sources
    if N_src < min_sources:
        return kwargs
    
    # set initial values if needed
    if not hasattr(kwargs,'cfit_abs_max'):
        kwargs['cfit_abs_max'] = 0.01
    if not hasattr(kwargs,'qfit_max'):
        kwargs['qfit_max'] = 0.05
        
    # loop to find the right criteria
    s,s_dict,msg = filter_psfphot_results(phot_result,full_output=True,**kwargs)
    
    # initialize stopping criteria
    s_prev = s.sum()
    s_prev_prev = s_prev
    init_run =True
    for _ in range(maxiter):
        if s.sum()/N_src > target_passing_fraction:
            break
        if verbose:
            logger.info(f'too many sources are cut ({s.sum()} out of {N_src}). Updating the filter criteria.')
        # determine which criteria to loosen
        cfit_pass_frac = s_dict['s_cfit'].sum()/N_src
        qfit_pass_frac = s_dict['s_qfit'].sum()/N_src
        if cfit_pass_frac < qfit_pass_frac:
            kwargs['cfit_abs_max'] = np.round(kwargs['cfit_abs_max'] + cfit_increment,5)
            if verbose:
                logger.info(f'cfit_abs_max updated to {kwargs["cfit_abs_max"]}')
        else:
            kwargs['qfit_max'] = np.round(kwargs['qfit_max'] + qfit_increment,5)
            if verbose:
                logger.info(f'qfit_max updated to {kwargs["qfit_max"]}')
                
        # run filtering and see if there is any improvement
        s,s_dict,msg = filter_psfphot_results(phot_result,full_output=True,**kwargs)
        if s.sum() <= s_prev and s_prev_prev == s_prev and not init_run:
            if verbose:
                logger.info(f'no improvement in the number of sources passed the cut. Stopping the loop.')
            break
        s_prev_prev = s_prev
        s_prev = s.sum()
        init_run = False
    return kwargs

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

    #### run photmetry
    # psf_iter = IterativePSFPhotometry(
    #     psf_model, 
    #     finder             = daofinder,
    #     grouper            = grouper,
    #     localbkg_estimator = localbkg_estimator,
    #     fit_shape       = config['psf']['PSFPhotometry_fit_shape'],
    #     mode            = config['psf']['PSFPhotometry_mode'],
    #     aperture_radius = config['psf']['PSFPhotometry_aperture_radius'],
    #     maxiters        = config['psf']['PSFPhotometry_maxiters'],
    #     fitter_maxiters = config['psf']['PSFPhotometry_fitter_maxiters'],
    #     group_warning_threshold = config['psf']['PSFPhotometry_group_warning_threshold'],
    #     multiprocessing = config['psf']['PSFPhotometry_multiprocessing'],
    #     )
    psf_iter = None # TODO remove this
    
    psf_single = PSFPhotometry(
        psf_model, 
        finder             = daofinder,
        grouper            = grouper,
        localbkg_estimator = localbkg_estimator,
        fit_shape       = config['psf']['PSFPhotometry_fit_shape'],
        aperture_radius = config['psf']['PSFPhotometry_aperture_radius'],
        fitter_maxiters = config['psf']['PSFPhotometry_fitter_maxiters'],
        group_warning_threshold = config['psf']['PSFPhotometry_group_warning_threshold'],
        multiprocessing = config['psf']['PSFPhotometry_multiprocessing'],
    )
    return psf_iter, psf_single

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
    verbose = kwargs.get('verbose',False)
    # tools

    # initialize fitters
    psf_iter, psf_single = _prepare_psf_fitters(th, psf_model, bkg_std, psf_sigma)

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
            logger.error(f'Too many blended sources. Aborting th={th}...')
            raise
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
    This function is useful for crowded fields, where a single threshold level may fail. '''
      
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

    # loop -- repeat PSF subtraction
    if progress is not None:
        progress_psf = progress.add_task(progress_text, 
                                        total=len(threshold_list))
    for th in threshold_list:
        try:
            psf_results = do_psf_photometry(resid, data_error, bkg_std, 
                                            psf_model, psf_sigma,
                                            th=th,**kwargs)
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
            logger.error(f'Skipping PSF fitting with th={th:.2f}. {str(e)}')
            continue
    if progress is not None:
        progress.remove_task(progress_psf)
        
    # make the residual image without background subtraction
    psf_modelimg_all = data_bksub - resid
    resid_all = data.copy() - psf_modelimg_all.copy()
    return phot_result, resid_all