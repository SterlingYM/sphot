# plots.py

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import logging
from scipy import stats

from photutils import CircularAperture

import astropy.units as u
from astropy.stats import sigma_clip
from astropy.nddata import Cutout2D
from astropy.modeling import models

def astroplot(data,percentiles=[1,99.9],cmap='viridis',
              ax=None,
              offset=0,norm=None,figsize=(5,5),title=None,set_bad='r',**kwargs):
    ''' plot 2d data in the log scale. Automatically takes care of the negative values. '''
    if (data is None) or np.isnan(data).all():
        raise ValueError('Data is empty!')
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)

    # auto-normalize data in log scale 
    # (and take care of the negative values if needed)       
    if norm is None:
        vmin,vmax = np.nanpercentile(data,percentiles)
        if vmin <= 0:
            offset = -vmin + 1e-1 # never make it zero
        else:
            offset = 0
        vmin += offset 
        vmax += offset
        norm = LogNorm(vmin=vmin,vmax=vmax)
    else:
        assert offset is not None, 'offset has to be provided if norm is provided'

    # plot
    clipped_data = data.copy() + offset
    clipped_data[clipped_data<=norm.vmin] = norm.vmin
    clipped_data[clipped_data>=norm.vmax] = norm.vmax
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
        cmap.set_bad(set_bad)
    ax.imshow(clipped_data,norm=norm,cmap=cmap,origin='lower')
    ax.set_yticks([])
    ax.set_xticks([])
    if title is not None:
        ax.set_title(title,fontsize=13)
    return norm,offset

def plot_sersicfit_result(data,data_annulus,_img):
    ''' 
    Plot the sersic profile fit result.
    
    Parameters:
        data (2d array): the original data
        _img (2d array): the fitted model image
    Returns:
        None
    '''
    _res = data - _img
    fig = plt.figure(figsize=(10,10))
    ax_img = fig.add_axes([0.1,0.1,0.4,0.4])
    ax_top = fig.add_axes([0.1,0.5,0.4,0.25])
    ax_right = fig.add_axes([0.5,0.1,0.25,0.4])
    ax_hist = fig.add_axes([0.4,0.6,0.6,0.2])
    ax_model = fig.add_axes([0.62,0.3,0.12,0.3])

    ax_res = fig.add_axes([0.85,0.1,0.4,0.4])
    ax_top2 = fig.add_axes([0.85,0.5,0.4,0.25])
    ax_left2 = fig.add_axes([0.6,0.1,0.25,0.4])

    norm,offset = astroplot(data,ax=ax_img,percentiles=[0.01,99.9],cmap='viridis')
    astroplot(_res,ax=ax_res,norm=norm,offset=offset,cmap='viridis')
    astroplot(_img,ax=ax_model,norm=norm,offset=offset,cmap='viridis')

    for row in data:
        ax_top.plot(row+1,c='k',alpha=0.2)
    ax_top.plot(np.nanmax(_img+1,axis=0),color='r',lw=2,ls=':')
    ax_top.set_xlim(0,data.shape[0])
    ax_top.set_ylim(np.nanpercentile(data,2)+1,)
    ax_top.set_yscale('log')
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.axis('off')

    for row in data.T:
        ax_right.plot(row+1,np.arange(len(row)),c='k',alpha=0.2)
    ax_right.plot(np.nanmax(_img+1,axis=1),np.arange(len(row)),color='r',lw=2,ls=':')
    ax_right.set_ylim(0,data.shape[0])
    ax_right.set_xlim(np.nanpercentile(data,2)+1,)
    ax_right.set_xscale('log')
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.axis('off')

    count_bins = np.linspace(*np.nanpercentile(_res,[0.1,99.9]),100)
    ax_hist.hist(data_annulus-np.nanmean(data_annulus),bins=count_bins,color='lightgray',lw=2,label='data (annulus)',density=True)
    # ax_hist.plot(count_bins,lik_func(count_bins),c='gray',lw=2,label='likelihood model')
    ax_hist.hist(data.flatten(),bins=count_bins,color='k',histtype='step',lw=2,density=True,label='data (total)');
    ax_hist.hist(_res.flatten(),bins=count_bins,color='tomato',histtype='step',lw=2,density=True,label='fit residual');
    ax_hist.legend(frameon=False,fontsize=13)
    ax_hist.set_yscale('log')
    # ax_hist.set_ylim(1e-3,lik_func(count_bins).max()*1.3)
    ax_hist.tick_params(direction='in')
    ax_hist.set_yticks([])
    ax_hist.tick_params(direction='in',which='both',left=False)
    ax_hist.set_ylabel('count density',fontsize=13)
    ax_hist.set_xlabel('pixel value [MJy/sr]',fontsize=13)

    for row in _res:
        ax_top2.plot(row+1,c='k',alpha=0.2)
    ax_top2.set_xlim(ax_top.get_xlim())
    ax_top2.set_ylim(ax_top.get_ylim())
    ax_top2.set_yscale('log')
    ax_top2.set_xticks([])
    ax_top2.set_yticks([])
    ax_top2.axis('off')

    for row in _res.T:
        ax_left2.plot(row+1,np.arange(len(row)),c='k',alpha=0.2)
    ax_left2.set_ylim(ax_right.get_ylim())
    ax_left2.set_xlim(ax_right.get_xlim())
    ax_left2.invert_xaxis()
    ax_left2.set_xscale('log')
    ax_left2.set_xticks([])
    ax_left2.set_yticks([])
    ax_left2.axis('off')
    
def plot_profile2d(data,ax=None,fig=None,lower_limit_percentile=20,
                   left=False,**kwargs):
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(8,8))
    elif fig is None:
        fig = plt.gcf()
    norm,offset = astroplot(data,ax=ax,**kwargs)
    
    pos = ax.get_position()
    ax_top = fig.add_axes([pos.x0,pos.y1,pos.width,pos.height/4])
    
    if left:
        ax_side = fig.add_axes([pos.x0-pos.width/4,pos.y0,pos.width/4,pos.height])
    else:
        ax_side = fig.add_axes([pos.x1,pos.y0,pos.width/4,pos.height])
    
    upper_limit = norm((np.nanmax(data)+offset)*1.5)
    lower_limit = np.nanpercentile(norm(data+offset).data,lower_limit_percentile)
    for row in data:
        if np.isfinite(row).sum() == 0:
            continue
        val = norm(row+offset)
        val_norm = np.clip(np.nanmax(val),0,None)
        alpha = np.clip(0.3*(val_norm-lower_limit)/(upper_limit-lower_limit),0,1)
        ax_top.plot(val,c='k',
                    alpha=alpha,
                    zorder=val_norm)
    ax_top.set_xlim(0,data.shape[0])
    ax_top.set_ylim(lower_limit,upper_limit)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.axis('off')

    for row in data.T:
        val = norm(row+offset)
        val_norm = np.clip(np.nanmax(val),0,None)
        alpha = np.clip(0.3*(val_norm-lower_limit)/(upper_limit-lower_limit),0,1)
        ax_side.plot(val,np.arange(len(row)),c='k',
                      alpha=alpha,
                      zorder=val_norm)
    ax_side.set_ylim(0,data.shape[0])
    ax_side.set_xlim(lower_limit,upper_limit)
    ax_side.set_xticks([])
    ax_side.set_yticks([])
    ax_side.axis('off')
    if left:
        ax_side.invert_xaxis()
    return norm,offset
        
def plot_sphot_results(cutoutdata,right_attr='psf_sub_data',dpi=100,**kwargs):
    sky_model = getattr(cutoutdata,'sky_model',None)
    if sky_model is None:
        sky_model = cutoutdata._bkg_level
    rawdata_bksub = cutoutdata._rawdata - sky_model.mean() # cutoutdata._bkg_level
    bestfit_sersic_img = cutoutdata.sersic_modelimg
    sersic_residual = cutoutdata.sersic_residual
    psf_model_total = cutoutdata.psf_modelimg
    psf_subtracted_data_bksub = getattr(cutoutdata,right_attr)
                  
    fig = plt.figure(figsize=(10,10),dpi=dpi)
    ax0 = fig.add_axes([-0.45,0.55,0.55,0.55])
    ax4 = fig.add_axes([0.14,0.66,0.25,0.25])
    ax3 = fig.add_axes([0.4,0.4,0.25,0.25])
    ax2 = fig.add_axes([0.66,0.66,0.25,0.25])
    ax1 = fig.add_axes([0.4,0.92,0.25,0.25])
    ax5 = fig.add_axes([0.95,0.55,0.55,0.55])
    ax_name = fig.add_axes([0.4,0.66,0.25,0.25])
    ax_name.axis('off')
    ax_name.text(0.5,0.5,cutoutdata.filtername,
                 transform=ax_name.transAxes,
                 ha='center',va='center',fontsize=28,color='k')
    
    cmap = kwargs.get('cmap','viridis')
    norm,offset = astroplot(rawdata_bksub,ax=ax1,percentiles=[0.1,99.9],cmap=cmap)
    kwargs['norm'] = norm
    kwargs['offset'] = offset
    kwargs['cmap'] = cmap
    plot_profile2d(rawdata_bksub,ax0,fig,left=True,**kwargs)
    astroplot(bestfit_sersic_img,ax=ax2,**kwargs)
    astroplot(sersic_residual,ax=ax3,**kwargs)
    astroplot(psf_model_total,ax=ax4,**kwargs)
    plot_profile2d(psf_subtracted_data_bksub,ax5,fig,**kwargs)

    arrowprops0 = dict(arrowstyle="simple",lw=0,fc='dodgerblue')
    arrowprops1 = dict(arrowstyle="simple",connectionstyle="arc3,rad=-0.3",
                    lw=0,fc='yellowgreen')
    arrowprops2 = dict(arrowstyle="simple",connectionstyle="arc3,rad=-0.55",
                    lw=0,fc='orange')
    arrowprops3 = dict(arrowstyle="simple",lw=0,fc='orange')
    arrow_kwargs = dict(
        xycoords='axes fraction',
        textcoords='axes fraction',
        size=50, va="center", ha="center",
    )
    ax4.annotate("",xy=(1.05,1.3), xytext=(-0.17,1.3), arrowprops=arrowprops0,zorder=0,**arrow_kwargs)
    ax1.annotate("",xy=(0,0.5), xytext=(-0.55,-0.05), arrowprops=arrowprops1,**arrow_kwargs)
    ax1.annotate("",xy=(1.55,-0.05), xytext=(1.0,0.5), arrowprops=arrowprops1,**arrow_kwargs)
    ax3.annotate("",xy=(1.0,0.5), xytext=(1.55,1.05), arrowprops=arrowprops1,**arrow_kwargs)
    ax3.annotate("",xy=(-0.55,1.05), xytext=(0,0.5), arrowprops=arrowprops1,**arrow_kwargs)
    ax4.annotate("",xy=(1.7,1.55), xytext=(0.33,0.99), arrowprops=arrowprops2, **arrow_kwargs)
    ax1.annotate("",xy=(2.2,0.67), xytext=(1.0,0.67), arrowprops=arrowprops3, zorder=10,**arrow_kwargs)
    ax2.annotate('Sersic\nfit',xy=(0.55,1.4), xycoords='axes fraction', ha='center', va='center', fontsize=25)
    ax2.annotate('remove\nprofile',xy=(0.57,-0.55), xycoords='axes fraction', ha='center', va='center', fontsize=25)
    ax3.annotate('PSF fit',xy=(-0.55,0.5), xycoords='axes fraction', ha='center', va='center', fontsize=25)
    ax4.annotate('remove\nPSF',xy=(0.5,1.65), xycoords='axes fraction', ha='center', va='center', fontsize=25)

    ax0.text(0.1,0.88,'A',transform=ax0.transAxes,ha='center',va='center',fontsize=30,color='w')
    ax1.text(0.1,0.88,'A\'',transform=ax1.transAxes,ha='center',va='center',fontsize=30,color='w')
    ax2.text(0.1,0.88,'B',transform=ax2.transAxes,ha='center',va='center',fontsize=30,color='w')
    ax3.text(0.3,0.88,'C: A-B',transform=ax3.transAxes,ha='center',va='center',fontsize=30,color='w')
    ax4.text(0.1,0.88,'D',transform=ax4.transAxes,ha='center',va='center',fontsize=30,color='w')
    ax5.text(0.2,0.88,'S: A-D',transform=ax5.transAxes,ha='center',va='center',fontsize=30,color='w')