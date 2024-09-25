#!python

description = '''run_sphot.py
Y.S.Murakami 2024 @ JHU
Run the basic sphot pipeline on a single galaxy.
This script automatically detects the running environment (e.g., Slurm array job or local machine) and switches the logging output accordingly.

Usage:
    run_sphot data_file.h5  [--out_folder=foldername] # run sphot on a new data file (option 1)
    run_sphot sphot_file.h5 --rerun_all      # rerun both basefit and scalefit (option 2)
    run_sphot sphot_file.h5 --continue       # continue scalefit on existing sphot file if necessary (option 3)
    run_sphot sphot_file.h5 --rerun_basefit  # rerun basefit on existing sphot file (option 4)
    run_sphot sphot_file.h5 --rerun_scalefit # rerun all scalefit on existing sphot file (option 5)
    run_sphot sphot_file.h5 --rerun_scalefit --filter=F555W # rerun specific filter 

Requirements:
    - for running the initial fit, a file PSFdata.h5 is required in the working directory. See documentation for the format.
    - fit parameters can be changed by placing sphot_conf.py in the working directory. If the file is not found, the default settings are used.
    
'''

import sys
import os 
import numpy as np
from sphot.utils import load_and_crop
from sphot.core import run_basefit, run_scalefit, logger
from sphot.data import read_sphot_h5
from .config import config

filters = config['core']['filters']
PSF_file = config['core']['PSF_file']
base_filter = config['core']['base_filter']
blur_psf = config['core']['blur_psf']
iter_basefit = config['core']['iter_basefit']
iter_scalefit = config['core']['iter_scalefit']
fit_complex_model = config['core']['fit_complex_model']
allow_refit = config['core']['allow_refit']
custom_initial_crop = config['core']['custom_initial_crop']
sigma_guess = config['core']['sigma_guess']
filters_to_fit = filters.copy()
    
def argv_to_kwargs(args):
    # default options
    initial_run = True
    rerun_basefit = False
    continue_scalefit = False
    rerun_scalefit = False
    out_folder = './'

    # parse command line arguments
    datafile = args[1]
    if len(args) > 2:
        for arg in args[2:]:
            if arg == '--rerun_all':
                logger.info('rerun all option detected')
                rerun_basefit = True
                rerun_scalefit = True
                initial_run = False
            elif arg == '--rerun_basefit':
                logger.info('basefit rerun option detected')
                rerun_basefit = True
                initial_run = False
            elif arg == '--continue':
                logger.info('continue option detected. Running scalefit if necessary.')
                continue_scalefit = True
                initial_run = False
            elif arg == '--rerun_scalefit':
                logger.info('rerun option detected')
                scalefit_only = True
                rerun_scalefit = True
                initial_run = False
            elif arg.startswith('--out_folder'):
                out_folder = arg.split('=')[1]
                logger.info(f'output folder specified: {out_folder}')
            elif arg.startswith('--filter'):
                filters_to_fit = np.atleast_1d(arg.split('=')[1].split(','))
                logger.info(f'filters to fit: {filters_to_fit}')
            else:
                logger.info(f'unknown options: {arg}')   
                
    kwargs = dict(datafile=datafile,
                  initial_run = initial_run,
                  continue_scalefit = continue_scalefit,
                  rerun_basefit=rerun_basefit,
                  rerun_scalefit=rerun_scalefit,
                  out_folder=out_folder,
                  filters_to_fit=filters_to_fit)
    return kwargs

def run_sphot(datafile,
              initial_run=True,continue_scalefit=False,
              rerun_basefit=False,rerun_scalefit=False,
              out_folder='.',filters_to_fit=[],**kwargs):
    ''' main commands are put in this dummy function so that the rich output can be forwarded to a log file when running in slurm'''

    # 1. load data
    if initial_run:
        galaxy = load_and_crop(datafile,filters,PSF_file,
                            base_filter = base_filter,
                            plot = False,
                            custom_initial_crop = custom_initial_crop,
                            sigma_guess = sigma_guess)
        out_path = os.path.join(out_folder,f'{galaxy.name}_sphot.h5')
        logger.info(f'Galaxy data loaded: sphot file will be saved as {out_path}')
        rerun_basefit = True
        rerun_scalefit = True
    else:
        logger.info(f'Loading an existing sphot filt: {datafile}')
        galaxy = read_sphot_h5(datafile)
        out_path=datafile

    # 2. fit Sersic model using the base filter
    if rerun_basefit:
        logger.info('----- Starting base fit -----')
        run_basefit(galaxy,
                    base_filter = base_filter,
                    fit_complex_model = fit_complex_model,
                    blur_psf = blur_psf[base_filter],
                    N_mainloop_iter = iter_scalefit,
                    **kwargs)
        galaxy.save(out_path)
    
    # 3. Scale Sersic model (if necessary)
    if rerun_scalefit or continue_scalefit:
        logger.info('----- Starting Scale fit -----')
        logger.info(f'Filters to fit: {filters_to_fit}')
        base_params = galaxy.images[base_filter].sersic_params
        for filt in filters_to_fit:
            # check if we should skip scalefit
            if hasattr(galaxy.images[filt],'psf_sub_data') and not rerun_scalefit:
                # skip scalefit if data exists and not rerun_scalefit==True
                if not np.any(np.isfinite(getattr(galaxy.images[filt],'psf_sub_data'))):
                    logger.info(f'Filter {filt} already has PSF-subtracted data, but it is all-NaN. running Sphot again...')
                    pass
                else:
                    logger.info(f'Filter {filt} already has PSF-subtracted data')
                    continue
                
            # run scalefit
            try:
                run_scalefit(galaxy,filt,base_params,
                            allow_refit=allow_refit,
                            fit_complex_model=fit_complex_model,
                            N_mainloop_iter=7,
                            blur_psf=blur_psf[filt],
                            **kwargs)
                galaxy.save(out_path)
            except Exception as e:
                logger.info(f'Filter {filt} failed: {str(e)}')
                continue
            
    logger.info('Completed Sphot')

def prep_console_wrapper():
    if "SLURM_JOB_ID" in os.environ:
        from rich.console import Console
        def console_wrapper(func,*args,**kwargs):
            slurm_jobid = os.environ.get("SLURM_ARRAY_JOB_ID")
            slurm_taskid = os.environ.get("SLURM_ARRAY_TASK_ID")
            logfile = f'logs/{slurm_jobid}_{slurm_taskid}.log'
            logger.info(f"Running in Slurm (jobid={slurm_jobid}, taskid={slurm_taskid})")
            logger.info(f'Saving the progress in the log file: {logfile}')
            print(f'Saving the progress in the log file: {logfile}',flush=True)
            with open(logfile, 'w') as log_file:
                # Create a Console instance that writes to the log file
                console = Console(file=log_file, force_terminal=True, force_interactive=True)   
                # console.print('test: this should be written to the log file '+logfile)
                kwargs.update(dict(console=console))
                return func(*args,**kwargs)
    else:
        def console_wrapper(func,*args,**kwargs):
            return func(*args,**kwargs)
    return console_wrapper

def main():
    if sys.argv[1] == '--help':
        print(description)
        sys.exit()
    
    # parse command line arguments
    kwargs = argv_to_kwargs(sys.argv)
                
    # switch logging option based on how this file is running
    console_wrapper = prep_console_wrapper()
            
    # run sphot
    kwargs['plot'] = False
    console_wrapper(run_sphot,**kwargs)
    

if __name__ == '__main__':
    main()