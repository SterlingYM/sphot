import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import os
import astropy.units as u
from astropy.modeling import models
from astropy.table import Table, vstack
from tqdm.auto import tqdm

from .fitting import SphotModel
from .data import (CutoutData, MultiBandCutout, 
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