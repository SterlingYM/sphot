import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import os
import astropy.units as u
from astropy.modeling import models
from astropy.table import Table, vstack
from tqdm.auto import tqdm
from termcolor import colored

from .fitting import SphotModel
from .data import (CutoutData, MultiBandCutout, 
                        load_h5data, get_data_annulus)

def load_and_crop(datafile,filters,psffile=None,
                  base_filter='F150W',plot=True,custom_initial_crop=False,
                  auto_crop=True,auto_crop_factor=8,**kwargs):
            
    # load data
    galaxy_ID = os.path.splitext(os.path.split(datafile)[-1])[0]
    galaxy = load_h5data(datafile, galaxy_ID, filters, psffile=psffile)
    if plot:
        galaxy.plot()
        
    # custom crop-in (run this if galaxy seems too small)
    if custom_initial_crop:
        shape = galaxy.images[base_filter].data.shape
        x0, y0 = shape[1]/2, shape[0]/2
        cutout_size = shape[0] * custom_initial_crop
        galaxy.crop_in(x0, y0, cutout_size)

    # estimate size of the galaxy
    cutoutdata = galaxy.images[base_filter]
    size_guess_kwargs = dict(plot=plot, sigma_guess=10, center_slack = 0.20, sigma_kernel=5)
    size_guess_kwargs.update(kwargs)
    cutoutdata.init_size_guess(**size_guess_kwargs)

    # determine cutout size based on the initial fit
    galaxy_size = cutoutdata.size_guess
    x0, y0 = cutoutdata.x0_guess, cutoutdata.y0_guess

    if auto_crop:
        cutout_size = galaxy_size * auto_crop_factor * 2 # number of pixels in each axis (hence x2)
        galaxy.crop_in(x0, y0, cutout_size)
    if plot:
        galaxy.plot()
        plt.show()
    for cutoutdata in galaxy.image_list:
        cutoutdata.galaxy_size = galaxy_size
    return galaxy


def prep_model(cutoutdata,simple=False,fixed_params={}):
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
    model.set_fixed_params(fixed_params)
    return model

from termcolor import colored

def print_h5_structure(name, obj, prefix='',
                       pattern=None,terminate=False,show_attrs=True):
    """Recursively prints the HDF5 file structure."""
    # exception for the top level
    
    # If the item is a group, recurse into it.
    if isinstance(obj, h5py.Group):
        Nobj = len(list(obj.keys()))
        Nattr = len(list(obj.attrs.keys()))
        
        # first print all attributes
        if show_attrs:
            for i, (key, val) in enumerate(obj.attrs.items()):
                joint = '└── ' if i == Nattr-1 and Nobj == 0 else '├── '
                text = prefix + joint + colored(f'{key}: {val}', 'magenta')
                print(text)        
        
        # group header
        if name != '/': # show group name unless it's the root
            joint = '└── ' if terminate else '├── '
            text = prefix + joint + colored(name + f' ({Nobj} obj, {Nattr} attr)', 'green')
            print(text)
            next_prefix = prefix + '    ' if terminate else prefix +  '│   '
        else:
            next_prefix = ''
            
        # make a list of items to recurse into (pattern match filter)
        recurse_items = {}
        for key, item in obj.items():
            if isinstance(item, h5py.Group):
                recurse_items[key] = item
            else:
                if pattern is not None:
                    if pattern in key:
                        recurse_items[key] = item
                else:
                    recurse_items[key] = item
        for i, (key, item) in enumerate(recurse_items.items()):
            terminate = i == len(recurse_items) - 1
            print_h5_structure(key, item, next_prefix, pattern, terminate, show_attrs)
            
    # if the item is dataset, print info
    else:
        if pattern is not None and pattern not in name:
            return
        joint = '└── ' if terminate else '├── '
        text = prefix + joint + colored(name + f' {obj.shape}, {obj.dtype}', 'yellow')
        print(text)

def h5tree(file_path,pattern=None,show_attrs=True):
    ''' a wrapper for print_h5_structure '''
    filename = os.path.basename(file_path)
    with h5py.File(file_path, 'r') as hdf:
        Nobj = len(list(hdf.keys()))
        Nattr = len(list(hdf.attrs.keys()))
        print(colored(f'{filename} ({Nobj} obj, {Nattr} attr)','green'))
        print_h5_structure('/', hdf,pattern=pattern,show_attrs=show_attrs)