# fitter.py
import numpy as np

from scipy.optimize import minimize, dual_annealing
from scipy.ndimage import zoom

from .config import config

from petrofit import PSFConvolvedModel2D, model_to_image


def _install_cached_convolve():
    """Replace petrofit's module-level `convolve` (scipy.signal.convolve)
    with a wrapper that caches the PSF kernel's FFT.

    The Sersic model image is convolved with the SAME kernel on every
    objective evaluation (~1e4 per base fit); scipy recomputes the
    kernel's transform (and re-runs method selection) each call. Cache
    the padded kernel FFT keyed by (kernel id, image shape) and do the
    image-side transform only. Falls back to scipy for anything but the
    2D real mode='same' case petrofit uses, and reproduces scipy's
    'same' cropping convention exactly (verified for odd and even
    kernels).
    """
    import petrofit.modeling.models as _pmm
    import scipy.signal as _ss
    import scipy.fft as _sfft
    if getattr(_pmm.convolve, '_sphot_cached', False):
        return
    _orig = _pmm.convolve
    _cache = {}

    def _cached_convolve(in1, in2, mode='full', method='auto'):
        if (mode != 'same' or np.iscomplexobj(in1) or np.iscomplexobj(in2)
                or np.ndim(in1) != 2 or np.ndim(in2) != 2):
            return _orig(in1, in2, mode=mode, method=method)
        img = np.asarray(in1, dtype=float)
        ker = np.asarray(in2, dtype=float)
        H, W = img.shape
        kh, kw = ker.shape
        fh, fw = H + kh - 1, W + kw - 1
        sh = _sfft.next_fast_len(fh)
        sw = _sfft.next_fast_len(fw)
        key = (id(in2), ker.shape, sh, sw)
        kfft = _cache.get(key)
        if kfft is None:
            if len(_cache) > 8:
                _cache.clear()
            kfft = _sfft.rfft2(ker, s=(sh, sw))
            _cache[key] = kfft
        full = _sfft.irfft2(_sfft.rfft2(img, s=(sh, sw)) * kfft,
                            s=(sh, sw))[:fh, :fw]
        y0 = (kh - 1) // 2
        x0 = (kw - 1) // 2
        return full[y0:y0 + H, x0:x0 + W]

    _cached_convolve._sphot_cached = True
    _pmm.convolve = _cached_convolve


_install_cached_convolve()
        
def _crop_psf_for_sersic(psf):
    ''' Trim the PSF used for the SERSIC convolution to the smallest
    centred square holding `[core].sersic_psf_crop_flux_fraction` of its
    flux, then renormalise.

    The Sersic model is re-convolved with this PSF on every objective
    evaluation (~1e4 per base fit) and that FFT dominates the Sersic fit;
    its cost scales with the padded image+PSF size, so the PSF's far wings
    are most of the work. A galaxy is far larger than the PSF (r_eff of
    tens of px vs a few-px core), so truncating those wings perturbs a
    smooth extended profile only slightly -- measured 2.4x faster per
    evaluation.

    This is emphatically NOT applied to the PSF used for point-source
    photometry, where the wings carry real flux and truncating them would
    bias fluxes. Set the fraction to 1.0 (or 0) to disable.
    '''
    try:
        frac = float(config['core'].get('sersic_psf_crop_flux_fraction', 0.98))
    except Exception:
        return psf
    if not (0.0 < frac < 1.0):
        return psf
    p = np.asarray(psf, dtype=float)
    H, W = p.shape
    cy, cx = (H - 1) // 2, (W - 1) // 2
    max_half = int(min(cy, cx, H - 1 - cy, W - 1 - cx))
    if max_half < 8:
        return psf
    total = float(p.sum())
    if not np.isfinite(total) or total <= 0:
        return psf
    # enclosed flux in growing centred squares, via an integral image
    I = np.cumsum(np.cumsum(p, axis=0), axis=1)

    def enclosed(h):
        y0, y1 = cy - h - 1, cy + h
        x0, x1 = cx - h - 1, cx + h
        s = I[y1, x1]
        if y0 >= 0:
            s -= I[y0, x1]
        if x0 >= 0:
            s -= I[y1, x0]
        if y0 >= 0 and x0 >= 0:
            s += I[y0, x0]
        return float(s)

    half = max_half
    for h in range(8, max_half + 1):
        if enclosed(h) / total >= frac:
            half = h
            break
    if half >= max_half:
        return psf
    q = p[cy - half:cy + half + 1, cx - half:cx + half + 1].copy()
    s = q.sum()
    if s > 0:
        q /= s
    return q


class SphotModel(PSFConvolvedModel2D):
    def __init__(self,model,cutoutdata,resample_psf=True,**kwargs):
        '''
        A wrapper class for the petrofit model.
        Args:
            model (astropy FittableModel): model to fit.
            cutoutdata (CutoutData): the data to fit.
        '''
        if resample_psf:
            psf = cutoutdata.psf
            psf_oversample = cutoutdata.psf_oversample
            psf = zoom(psf,1/psf_oversample)
            psf /= psf.sum() # normalize
            psf = _crop_psf_for_sersic(psf)
            psf_oversample = 1
        else:
            psf = cutoutdata.psf
            psf_oversample = cutoutdata.psf_oversample

        
        super().__init__(model,
                        psf=psf, 
                        psf_oversample = int(psf_oversample),
                        oversample = int(psf_oversample))
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
                # Strictly positive: downstream EllipticalAperture
                # construction (sigma_clip_outside_aperture) requires
                # r_eff > 0. Sub-pixel r_eff has no physical meaning
                # for galaxy cutouts either way.
                bounds.append([0.5,data.shape[0]])
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
        '''Set condition functions.

        Args:
            list_of_conditions (list of 2-tuple): list of conditions, each as a 2-tuple. Each tuple ``(a, b)`` is evaluated as ``a >= b``. Either entry of the tuple can be a parameter name or a numerical value. For example, ``[('r_eff', 10), ('r_eff_1', 'r_eff_0')]`` returns True iff ``r_eff >= 10`` AND ``r_eff_1 >= r_eff_0``.
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
    '''
    A fitter class to perform Sersic model fitting to data.
    '''
    def __init__(self,model,cutoutdata,**kwargs):
        self.cutoutdata = cutoutdata
        self.shape = self.cutoutdata.data.shape
        self.model = model
        self.param_names = model._param_names

    def standardize_params(self,params):
        '''
        normalize parameters to be between 0 and 1.
        '''
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
        # petrofit.model_to_image takes size as (x_size, y_size); numpy
        # shape is (rows, cols) = (y, x), so reverse before passing.
        ny, nx = self.data.shape
        _img = model_to_image(self.model, size=(nx, ny))
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
        elif method == 'dual_annealing':
            # Iter-1 global escape. Pre-bracket with a short iNM, then run a
            # single scipy.dual_annealing pass. Output is routed through the
            # provided Rich `progress` task so it stays inside the existing
            # progress block instead of streaming raw prints.
            progress = kwargs.pop('progress', None)
            progress_text = kwargs.pop('progress_text',
                                       'dual_annealing global escape')
            da_maxiter = int(kwargs.pop('da_maxiter', 10))

            pre_iNM_kwargs = dict(rtol_init=1e-3, rtol_iter=1e-3,
                                  rtol_convergence=1e-6, xrtol=1, max_iter=5)
            result, _ = iterative_NM(self.calc_chi2, (), self.model.x0,
                                     self.bounds, **pre_iNM_kwargs)

            if progress is not None:
                da_task = progress.add_task(progress_text, total=da_maxiter)

                def _da_callback(x, f, context):
                    progress.update(da_task, advance=1, refresh=True)
                    return False
            else:
                _da_callback = None

            result = dual_annealing(
                self.calc_chi2,
                bounds=self.bounds,
                x0=result.x,
                maxiter=da_maxiter,
                initial_temp=10.0,
                seed=0,
                callback=_da_callback,
                minimizer_kwargs=dict(method='L-BFGS-B', bounds=self.bounds,
                                      options=dict(eps=1e-4, maxfun=1000)),
            )
            if progress is not None:
                progress.update(da_task, completed=da_maxiter, refresh=True)
                progress.remove_task(da_task)
            success = True
        elif method == 'BFGS':
            result = minimize(self.calc_chi2,self.model.x0,
                            bounds=self.bounds,
                            method='L-BFGS-B',
                            options=dict(eps=1e-4,maxfun=1000))
            success = result.success
        elif method == 'lbfgsb_polish':
            # Tight L-BFGS-B polish meant to run after iNM/dual_annealing
            # has placed us inside the basin; tightens chi^2 to gradient ~ 0.
            result = minimize(self.calc_chi2,self.model.x0,
                            bounds=self.bounds,
                            method='L-BFGS-B',
                            options=dict(eps=1e-5,ftol=1e-12,gtol=1e-10,
                                         maxfun=2000))
            success = result.success
        else:
            raise ValueError('method not recognized')
        
        # results
        bestfit_sersic_params_physical = _with_canonical_geometry(
            dict(zip(self.model.free_params,
                     self.unstandardize_params(result.x))))
        bestfit_img = self.eval_model(result.x)
        sersic_residual = self.cutoutdata._rawdata - bestfit_img # always take residual from raw data

        # update the total residual
        psf_modelimg = getattr(self.cutoutdata,'psf_modelimg',0)
        residual_img = self.cutoutdata._rawdata - psf_modelimg - bestfit_img
        residual_masked = residual_img.copy() # no sigma clipping

        # save all information in cutoutdata
        self.cutoutdata.sersic_params_physical = bestfit_sersic_params_physical
        self.cutoutdata.sersic_params = result.x
        self.cutoutdata.sersic_modelimg = bestfit_img
        self.cutoutdata.sersic_residual = sersic_residual
        self.cutoutdata.residual = residual_img
        self.cutoutdata.residual_masked = residual_masked
        # Dynamic galaxy-size attribute consumed by the PSF calibrator's
        # centre mask. Distinct from cd.galaxy_size (the prep-time
        # Gaussian σ guess) so we don't break anything that depends on
        # the static initial value.
        self.cutoutdata.galaxy_size_sersic = float(
            bestfit_sersic_params_physical.get('r_eff',
                                                self.cutoutdata.galaxy_size))
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
        bestfit_sersic_params_physical = _with_canonical_geometry(
            dict(zip(self.model.free_params,
                     self.unstandardize_params(scaled_modelparams))))
        bestfit_img = self.eval_model(scaled_modelparams)
        sersic_residual = self.cutoutdata._rawdata - bestfit_img # always take residual from raw data

        # update the total residual
        psf_modelimg = getattr(self.cutoutdata,'psf_modelimg',0)
        residual_img = self.cutoutdata._rawdata - psf_modelimg - bestfit_img
        residual_masked = residual_img.copy() # no sigma clipping
        
        # save all information in cutoutdata
        self.cutoutdata.sersic_params_physical = bestfit_sersic_params_physical
        self.cutoutdata.sersic_params = scaled_modelparams
        self.cutoutdata.sersic_modelimg = bestfit_img
        self.cutoutdata.sersic_residual = sersic_residual
        self.cutoutdata.residual = residual_img
        self.cutoutdata.residual_masked = residual_masked
        self.cutoutdata.galaxy_size_sersic = float(
            bestfit_sersic_params_physical.get('r_eff',
                                                self.cutoutdata.galaxy_size))
        self.model.x0 = result.x
        save_bestfit_params(self.cutoutdata,bestfit_sersic_params_physical)
        return self.cutoutdata
    
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

def _with_canonical_geometry(params):
    ''' Add canonical `r_eff` / `ellip` / `theta` / `n` / `amplitude` keys
    for multi-component models.

    A 2-component (disk+bulge) model names its parameters `r_eff_0`,
    `ellip_0`, `r_eff_1`, ... but downstream consumers ask for a single
    `sersic_params_physical['r_eff']` to size the sky mask
    (`sigma_clip_outside_aperture`), the calibration centre mask and
    `galaxy_size_sersic`. Alias them to the component with the LARGEST
    r_eff: those consumers all want the galaxy's overall extent, and
    `prep_model` already constrains r_eff_0 >= r_eff_1 so that is the disk.

    `x_0` / `y_0` are shared across components and need no aliasing.
    '''
    if 'r_eff' in params:
        return params
    idx = [k[len('r_eff_'):] for k in params if k.startswith('r_eff_')]
    if not idx:
        return params
    try:
        best = max(idx, key=lambda i: float(params.get(f'r_eff_{i}',
                                                       -np.inf)))
    except Exception:
        best = idx[0]
    out = dict(params)
    for base in ('r_eff', 'ellip', 'theta', 'n', 'amplitude'):
        key = f'{base}_{best}'
        if key in params and base not in out:
            out[base] = params[key]
    return out


def save_bestfit_params(cutoutdata,bestfit_sersic_params_physical,):
    for key,val in bestfit_sersic_params_physical.items():
        setattr(cutoutdata,key,val)
        
        
