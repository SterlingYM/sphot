import numpy as np
import matplotlib.pyplot as plt
from .plotting import astroplot
from tqdm.auto import tqdm
import warnings

from scipy.ndimage import gaussian_filter
from astropy.nddata import overlap_slices
from astropy.table import QTable, vstack
from astropy.stats import sigma_clip

from photutils.aperture import CircularAperture
from photutils.psf import (SourceGrouper, IterativePSFPhotometry, 
                           PSFPhotometry, FittableImageModel)
from photutils.detection import DAOStarFinder
from photutils.background import (LocalBackground, MMMBackground,
                                  MADStdBackgroundRMS)
from photutils.datasets.images import make_model_image as _make_model_image

from .data import get_data_annulus
from .logging import logger

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
        
    # model_img = np.zeros(shape)
    # for fit_model in fit_models:
    #     x0 = fit_model[2]#getattr(fit_model, 'x_0').value
    #     y0 = fit_model[3]#getattr(fit_model, 'y_0').value
    #     try:
    #         slc_lg, _ = overlap_slices(shape, psf_shape, (y0, x0),
    #                                     mode='trim')
    #     except Exception:
    #         continue
    #     yy, xx = np.mgrid[slc_lg]
    #     model_img[slc_lg] += fit_model(xx, yy)
    return model_img

class PSFFitter():
    ''' A class to perform PSF fitting. '''
    def __init__(self,cutoutdata):
        self.cutoutdata = cutoutdata
        self.psf_sigma = cutoutdata.psf_sigma
        
    def fit(self,fit_to='sersic_residual',**kwargs):
        self.data = getattr(self.cutoutdata,fit_to)
            
        x0 = self.cutoutdata.sersic_params_physical['x_0']
        y0 = self.cutoutdata.sersic_params_physical['y_0']
        center_mask_params = [x0,y0,self.psf_sigma*2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psf_table, resid = iterative_psf_fitting(self.data,
                                                     self.cutoutdata.psf,
                                                    psf_sigma = self.psf_sigma,
                                                    psf_oversample = self.cutoutdata.psf_oversample,
                                                    threshold_list = np.arange(1,3.2,0.2)[::-1],
                                                    center_mask_params=center_mask_params,
                                                    **kwargs)
        psf_model_total = self.data - resid
        psf_model_total -= np.nanmin(psf_model_total) # PSFs are forced to be positive, so minimum is always zero
        # TODO: handle the case where all pixels are filled with PSF

        # generate PSF-subtracted data
        mask = sigma_clip_outside_aperture(resid,
                                           self.cutoutdata.galaxy_size,clip_sigma=4,
                                           aper_size_in_r_eff=2,
                                           plot=True)
        psf_subtracted_data = self.cutoutdata._rawdata - psf_model_total
        psf_subtracted_data[mask] = np.nan

        # subtract background (to be consistent)
        data_annulus = get_data_annulus(psf_subtracted_data,5*self.cutoutdata.galaxy_size,plot=False)
        bkg_std = np.nanstd(data_annulus)
        psf_subtracted_data_bksub = psf_subtracted_data
        psf_subtracted_data_bksub_error = np.ones_like(psf_subtracted_data)*bkg_std
        
        # make the residual image
        sersic_modelimg = getattr(self.cutoutdata,'sersic_modelimg',0)
        residual_img = self.cutoutdata._rawdata - psf_model_total - sersic_modelimg
        residual_masked = residual_img.copy()
        residual_masked[mask] = np.nan
        
        # save data
        self.cutoutdata.residual = residual_img
        self.cutoutdata.residual_masked = residual_masked
        self.cutoutdata.psf_modelimg = psf_model_total
        self.cutoutdata.psf_sub_data = psf_subtracted_data_bksub 
        self.cutoutdata.psf_sub_data_error = psf_subtracted_data_bksub_error
        self.cutoutdata.psf_table = psf_table
        return self.cutoutdata

def do_psf_photometry(data,psfimg,psf_oversample,psf_sigma,
                      th=2,Niter=2,fit_shape=(3,3),render_shape=(25,25),
                      max_relative_error_flux=0.2,
                      plot=True,
                      finder_kwargs=dict(roundhi=1.0, roundlo=-1.0,
                                         sharplo=0.20, sharphi=1.0),
                      localbkg_bounds=(2,5),
                      grouper_sep=3.0,
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
        model_img (2d array): the model image.
        resid (2d array): the residual image.
    """    
    # tools
    bkgrms = MADStdBackgroundRMS()
    mmm_bkg = MMMBackground()

    # PSF image to model
    psf_model = FittableImageModel(psfimg, flux=1.0, 
                                   x_0=0, y_0=0, 
                                   oversampling=psf_oversample, 
                                   fill_value=0.0)

    # take data stats & prepare background-subtracted data
    try:
        bkg_level = mmm_bkg(data)
        assert np.isfinite(bkg_level), 'background is not finite'
        data_bksub = data - bkg_level
        bkg_std = bkgrms(data_bksub)
        error = np.ones_like(data_bksub) * bkg_std
        kwargs.update({'data_shape':data_bksub.shape})
    except Exception as e:
        logger.info(f'PSF background stats failed ({str(e)})')
        return None,None,None,None

    # more tools
    daofinder = DAOStarFinder(
        threshold=th*bkg_std, 
        fwhm=psf_sigma*2.33, **finder_kwargs)
    localbkg_estimator = LocalBackground(
        localbkg_bounds[0]*psf_sigma, 
        localbkg_bounds[1]*psf_sigma, 
        mmm_bkg)
    grouper = SourceGrouper(min_separation=3.0 * psf_sigma) 

    #### run phootmetry
    psf_iter = IterativePSFPhotometry(
        psf_model, fit_shape, 
        finder=daofinder,
        mode='new',
        grouper=grouper,
        localbkg_estimator=localbkg_estimator,
        aperture_radius=3,
        maxiters=5,
        fitter_maxiters=300
        )
    try:
        phot_result = psf_iter(data_bksub, error=error)
    except Exception as e:
        logger.info(f'IterativePSFPhotometry failed due to: {str(e)}')
        phot_result = None
    if phot_result is None:
        logger.info('No detection.')
        return None,None,None,None
    
    # repeat psf_iter Niter times to (hopefully) re-fit flagged sources
    for _ in range(Niter):
        s,msg = filter_psfphot_results(phot_result,**kwargs)
        if (s is None) or (s.sum() == 0):
            logger.info(f'No source passed the cut ({len(phot_result)} detection).'+msg)
            return None,None,None,None
        init_params = QTable()
        init_params['x'] = phot_result['x_fit'].value[s]
        init_params['y'] = phot_result['y_fit'].value[s]
        try:
            phot_result = psf_iter(data_bksub, 
                                   error=error, 
                                   init_params=init_params)
        except Exception as e:
            pass

    if phot_result is None:
        logger.info('PSFPhotometry failed.')
        return None,None,None,None
    
    #### final run -- fit all at once
    s,msg = filter_psfphot_results(phot_result,**kwargs)
    if s.sum() == 0:
        logger.info('all sources are flagged.'+msg)
        return None,None,None,None
    if kwargs.get('verbose',False):
        logger.info('photmoetry filter info:\n'+msg)
    init_params = QTable()
    init_params['x'] = phot_result['x_fit'].value[s]
    init_params['y'] = phot_result['y_fit'].value[s]
    psfphot = PSFPhotometry(
        psf_model, fit_shape, 
        finder=daofinder, 
        localbkg_estimator=localbkg_estimator,
        grouper=grouper,
        aperture_radius=3,
        fitter_maxiters=300
        )
    phot_result = psfphot(data_bksub, 
                          error=error, 
                          init_params=init_params)
    if phot_result is None:
        logger.info('PSFPhotometry failed.')
        return None,None,None,None
    
    # generate model image    
    s,msg = filter_psfphot_results(phot_result,**kwargs)
    params_table = QTable()
    params_table['x_0'] = phot_result['x_fit'].value[s]
    params_table['y_0'] = phot_result['y_fit'].value[s]
    params_table['flux'] = phot_result['flux_fit'].value[s]
    model_img = _make_model_image(
        shape = data_bksub.shape, 
        model = psfphot.psf_model, #psf_iter._psfphot.psf_model, 
        params_table = params_table, 
        model_shape = render_shape,
        x_name='x_0', y_name='y_0')
    resid = data_bksub - model_img

    # plot the results if necessary
    if plot:
        fig,axes = plt.subplots(1,3,figsize=(15,5))
        norm,offset = astroplot(data_bksub,ax=axes[0],percentiles=[0.1,99.9])
        astroplot(model_img,ax=axes[1],norm=norm,offset=offset)
        astroplot(resid,ax=axes[2],norm=norm,offset=offset)
        plt.show()
        
    return phot_result, data_bksub, model_img, resid

def filter_psfphot_results(phot_result,
                           center_mask_params=None,
                           cfit_abs_max=0.01,
                           qfit_max=0.05,
                           max_relative_error_flux=0.2,
                           data_shape=None,
                           **kwargs):
    ''' Filter the PSF photometry results. '''
    
    cfit_min,cfit_max = -1 * cfit_abs_max, cfit_abs_max
    qfit_min,qfit_max = 0, qfit_max
    
    s_flags = phot_result['flags'] < 1
    s_cfit = (phot_result['cfit'] >= cfit_min) & (phot_result['cfit'] <= cfit_max)
    s_qfit = (phot_result['qfit'] >= qfit_min) & (phot_result['qfit'] <= qfit_max)
    s_fluxerr = (phot_result['flux_err']/phot_result['flux_fit'] <= max_relative_error_flux)
    
    s = s_flags & s_cfit & s_qfit & s_fluxerr
    msg = f'\nsources that passed each cut (not cumulative, out of {len(phot_result)}):\n\
    flags: {s_flags.sum()},\n\
    cfit: {s_cfit.sum()},\n\
    qfit: {s_qfit.sum()},\n\
    flux_err/flux_fit: {s_fluxerr.sum()},\n'
    
    if center_mask_params is not None:
        x_center,y_center,mask_r = center_mask_params
        xdist = phot_result['x_fit'] - x_center
        ydist = phot_result['y_fit'] - y_center
        s_centermask = (xdist**2 + ydist**2 > mask_r**2)
        s = s & s_centermask
        msg += f'    center_mask: {int(s_centermask.sum())}\n'
        
    if data_shape is not None:
        x,y = phot_result['x_fit'],phot_result['y_fit']
        s_edge = (x > 0) & (x < data_shape[1]) & (y > 0) & (y < data_shape[0])
        s = s & s_edge
        msg += f'    edge: {int(s_edge.sum())}\n'    
    
    msg += f'Sources that passed all of the above cuts: {int(s.sum())}\n'

    return s, msg

def sigma_clip_outside_aperture(data,r_eff,clip_sigma=4,
                                aper_size_in_r_eff=1,plot=True):
    # sigma-clip pixels outside r_eff
    mask = sigma_clip(data,sigma=clip_sigma).mask
    aperture = CircularAperture((data.shape[0]/2,data.shape[1]/2),
                                r_eff*aper_size_in_r_eff)
    aperture_mask = aperture.to_mask(method='center')
    aperture_mask_img = aperture_mask.to_image(data.shape).astype(bool)
    mask[aperture_mask_img] = False
    return mask # bad pixels are True

def iterative_psf_fitting(data,psfimg,psf_sigma,psf_oversample,
                          threshold_list,
                          progress=None,
                          progress_text='Running iPSF...',
                        #   ignore_warnings=True,
                          **kwargs):
    ''' Perform iterative PSF fitting.'''
            
    # if resid is None:
    resid = data
    phot_result = None
    
    # repeat PSF subtraction
    if progress is not None:
        progress_psf = progress.add_task(progress_text, 
                                        total=len(threshold_list))
    for th in threshold_list:
        psf_results = do_psf_photometry(resid, psfimg,
                                        psf_sigma= psf_sigma,
                                        psf_oversample=psf_oversample,
                                        th=th,**kwargs)
        if progress is not None:
            progress.update(progress_psf, advance=1, refresh=True)
        if psf_results[0] is None:
            continue
        else:
            _phot_result, _, _, _resid = psf_results
            if np.all(~np.isfinite(_resid)):
                continue
            # append the results
            resid = _resid
            if phot_result is None:
                phot_result = _phot_result
            else:
                phot_result = vstack([phot_result, _phot_result])
    if progress is not None:
        progress.remove_task(progress_psf)
    return phot_result, resid
        