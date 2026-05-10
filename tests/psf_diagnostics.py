"""PSF photometry diagnostics for an existing sphot save file.

Investigates two specific failure modes:
    (A) bright stars leave bright residuals after PSF subtraction
    (B) dim stars are not captured by the photometry

Usage from this folder:

    python psf_diagnostics.py [filter]              # CLI: run all checks
    python -i psf_diagnostics.py F150W              # interactive

Or in a notebook (cells):

    from psf_diagnostics import *
    galaxy = load_galaxy('g260_sphot.h5')
    cd = galaxy.images['F150W']
    show_psf(cd); plt.show()
    threshold_scan(cd)                              # (B) detection coverage
    pr, resid, bkg = run_psf_photometry(cd, th=2.0)
    pass_mask = quality_cut_breakdown(pr, cd, bkg)  # (B) which cut kicks
    show_brightest_stars(cd, pr, n=8)               # (A) bright residuals
    quality_histograms(pr)
    show_detections_overlay(cd, pr, pass_mask)
"""
from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# allow running from inside tests/ folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

from sphot.config import config
from sphot.data import read_sphot_h5
from sphot.psf import (
    PSFFitter,
    do_psf_photometry,
    filter_psfphot_results,
    subtract_background,
)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_galaxy(filename='g260_sphot.h5'):
    if not os.path.exists(filename):
        # try relative to script dir
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(here, filename)
        if os.path.exists(cand):
            filename = cand
    return read_sphot_h5(filename)


# --------------------------------------------------------------------------
# (A) PSF model + input data inspection
# --------------------------------------------------------------------------

def show_psf(cd, ax=None):
    """Show the (blurred) PSF model that the fitter is using."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    psf = cd.psf
    pos = psf[psf > 0]
    vmin = np.percentile(pos, 1) if pos.size else psf.min()
    vmax = psf.max()
    ax.imshow(psf, norm=LogNorm(vmin=vmin, vmax=vmax),
              cmap='inferno', origin='lower')
    ax.set_title(f'{cd.filtername} PSF '
                 f'(σ_blur={cd.psf_blurring:.2f}, σ_eq={cd.psf_sigma:.2f})')
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def show_input_and_residual(cd, percentile=(1, 99.5)):
    """Plot the input image, saved psf_modelimg, and the saved residual map.

    Useful to eyeball whether the saved fit already shows the failure
    modes (A) and (B).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    images = [cd.sersic_residual, cd.psf_modelimg,
              cd.sersic_residual - cd.psf_modelimg]
    titles = ['input: sersic_residual',
              'saved psf_modelimg',
              'data - psf_modelimg']
    vmin, vmax = np.nanpercentile(cd.sersic_residual, percentile)
    offset = max(0.0, -vmin) + 1e-3
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img + offset, norm=LogNorm(vmin=vmin + offset, vmax=vmax + offset),
                  cmap='gray_r', origin='lower')
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    return fig, axes


# --------------------------------------------------------------------------
# (B) detection coverage
# --------------------------------------------------------------------------

def threshold_scan(cd, thresholds=None):
    """Count DAOStarFinder detections at a sweep of thresholds.

    Tells you whether dim stars are missed because the threshold is too
    high. The ladder runs *only* the detector (no PSFPhotometry), so it's
    fast.
    """
    data = cd.sersic_residual
    data_bksub, bkg_std, _ = subtract_background(data)

    if thresholds is None:
        thresholds = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0])

    finder_kwargs = config['psf']['finder_kwargs']
    psf_sigma = cd.psf_sigma

    print(f'\n[threshold scan] {cd.filtername}'
          f'   bkg_std={bkg_std:.4f}  psf_sigma={psf_sigma:.2f}  '
          f'fwhm_used={psf_sigma*2.33:.2f}')
    print(f'{"th":>6} {"th*bkg":>10} {"N":>6}')
    counts = []
    for th in thresholds:
        finder = DAOStarFinder(threshold=th * bkg_std,
                               fwhm=psf_sigma * 2.33,
                               **finder_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tab = finder(data_bksub)
        n = 0 if tab is None else len(tab)
        counts.append(n)
        print(f'{th:>6.1f} {th*bkg_std:>10.4f} {n:>6d}')
    return np.asarray(thresholds), np.asarray(counts), bkg_std


# --------------------------------------------------------------------------
# full PSFPhotometry pass + quality cut breakdown
# --------------------------------------------------------------------------

def run_psf_photometry(cd, th=None):
    """Run a single threshold pass at the saved blur. Returns
    (phot_result, residual_image, bkg_std).
    """
    if th is None:
        th = config['psf'].get('th_min', 2.0)

    psf_fitter = PSFFitter(cd)
    psf_fitter.update_psf_blur(cd.psf_blurring)

    data = cd.sersic_residual
    data_bksub, bkg_std, data_error = subtract_background(data)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        phot, resid = do_psf_photometry(
            data_bksub, data_error, bkg_std,
            psf_fitter.psf_model, psf_fitter.psf_sigma,
            th=th, plot=False,
        )
    n = 0 if phot is None else len(phot)
    print(f'\n[psf phot] th={th}: detected {n} sources')
    return phot, resid, bkg_std


def quality_cut_breakdown(phot, cd, bkg_std):
    """Run the same cuts filter_psfphot_results applies, but report the
    per-cut survivor count so we can see which one is doing the most damage.
    """
    if phot is None or len(phot) == 0:
        print('  no sources')
        return None

    cuts = config['psf']
    res_cen = phot['cfit'] * phot['flux_fit']
    cfit = phot['cfit']
    qfit = phot['qfit']
    npixfit = phot['n_pixels_fit']
    xerr = phot['x_err']
    yerr = phot['y_err']
    x_diff = phot['x_fit'] - phot['x_init']
    y_diff = phot['y_fit'] - phot['y_init']
    pos_diff = np.sqrt(x_diff**2 + y_diff**2)
    flux_snr = phot['flux_fit'] / phot['flux_err']

    # photutils flag bits (same mask as filter_psfphot_results)
    s_flags = ~(phot['flags'].value & (2 + 4 + 32 + 64 + 128 + 256)).astype(bool)
    s_poserr = (xerr <= cuts['cuts_pos_err_max']) & (yerr <= cuts['cuts_pos_err_max'])
    N = cuts['cuts_res_cen_sigma_clip']
    s_res = ~((res_cen < -3 * N * bkg_std)
              | ((res_cen < -N * bkg_std)
                 & (qfit / np.sqrt(npixfit) > N * bkg_std)))
    Nc = cuts['cuts_cfit_sigma_clip']
    _, cfit_med, cfit_std = sigma_clipped_stats(cfit[s_flags], sigma=Nc)
    cfit_std = max(cfit_std, 0.01)
    s_cfit = cfit >= (cfit_med - Nc * cfit_std)
    s_pos_drift = pos_diff <= np.nanmedian(pos_diff[s_flags]) * cuts['cuts_pos_diff_median_factor']
    s_snr = flux_snr >= cuts['cuts_flux_SNR_min']

    x0 = cd.sersic_params_physical['x_0']
    y0 = cd.sersic_params_physical['y_0']
    s_centermask = ((phot['x_fit'] - x0)**2
                    + (phot['y_fit'] - y0)**2
                    > (cd.psf_sigma * 2)**2)

    n = len(phot)
    print(f'\n[quality cuts] starting from {n} sources')
    print(f'  flag mask           : {int(s_flags.sum()):>5d}  ({100*s_flags.mean():.1f}%)')
    print(f'  pos_err <= {cuts["cuts_pos_err_max"]} px       : {int(s_poserr.sum()):>5d}  ({100*s_poserr.mean():.1f}%)')
    print(f'  res-centre OK       : {int(s_res.sum()):>5d}  ({100*s_res.mean():.1f}%)')
    print(f'  cfit > {cfit_med:+.4f} - {Nc}σ : {int(s_cfit.sum()):>5d}  ({100*s_cfit.mean():.1f}%)'
          f'   (median={cfit_med:+.4f}, std={cfit_std:.4f})')
    print(f'  pos drift OK        : {int(s_pos_drift.sum()):>5d}  ({100*s_pos_drift.mean():.1f}%)')
    print(f'  flux SNR >= {cuts["cuts_flux_SNR_min"]}      : {int(s_snr.sum()):>5d}  ({100*s_snr.mean():.1f}%)')
    print(f'  outside galaxy core : {int(s_centermask.sum()):>5d}  ({100*s_centermask.mean():.1f}%)')

    # Use the canonical filter to confirm the count
    s_all, _ = filter_psfphot_results(
        phot,
        center_mask_params=[x0, y0, cd.psf_sigma * 2],
        bkg_std=bkg_std)
    print(f'  [combined]          : {int(s_all.sum()):>5d}  ({100*s_all.mean():.1f}%)')
    return s_all


# --------------------------------------------------------------------------
# (A) bright-star residual inspection
# --------------------------------------------------------------------------

def show_brightest_stars(cd, phot, n=6, half_size=10, only_passing=False, bkg_std=None):
    """For the N brightest sources, plot
       data | model | residual  cutouts (centred on the fitted position).

    Asymmetric / dipole / central-spike residuals = wrong PSF width or
    miscentring. Uses the SAVED psf_modelimg/residual maps.
    """
    if phot is None or len(phot) == 0:
        print('no sources')
        return None

    if only_passing:
        if bkg_std is None:
            _, bkg_std, _ = subtract_background(cd.sersic_residual)
        x0 = cd.sersic_params_physical['x_0']
        y0 = cd.sersic_params_physical['y_0']
        mask, _ = filter_psfphot_results(
            phot, center_mask_params=[x0, y0, cd.psf_sigma * 2],
            bkg_std=bkg_std)
        phot = phot[mask]
        if len(phot) == 0:
            print('no passing sources')
            return None

    data = cd.sersic_residual
    model = cd.psf_modelimg
    resid = data - model

    order = np.argsort(np.asarray(phot['flux_fit']))[::-1][:n]
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for row, idx in enumerate(order):
        x = int(round(float(phot['x_fit'][idx])))
        y = int(round(float(phot['y_fit'][idx])))
        sy = slice(max(0, y - half_size), y + half_size + 1)
        sx = slice(max(0, x - half_size), x + half_size + 1)
        d = data[sy, sx]
        m = model[sy, sx]
        r = d - m
        vmin = float(np.nanpercentile(d, 5))
        vmax = float(np.nanpercentile(d, 99))
        rmax = float(np.nanmax(np.abs(r))) or 1e-6

        axes[row, 0].imshow(d, vmin=vmin, vmax=vmax, cmap='inferno', origin='lower')
        axes[row, 1].imshow(m, vmin=vmin, vmax=vmax, cmap='inferno', origin='lower')
        axes[row, 2].imshow(r, vmin=-rmax, vmax=rmax, cmap='RdBu_r', origin='lower')

        axes[row, 0].set_title(
            f'#{row+1}: flux={float(phot["flux_fit"][idx]):.2f}')
        axes[row, 1].set_title(
            f'cfit={float(phot["cfit"][idx]):+.4f}  qfit={float(phot["qfit"][idx]):.4f}')
        axes[row, 2].set_title(
            f'resid range [{r.min():+.2f}, {r.max():+.2f}]')
        for a in axes[row]:
            a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------
# distributions + overlay
# --------------------------------------------------------------------------

def quality_histograms(phot):
    if phot is None or len(phot) == 0:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    cfit = np.asarray(phot['cfit'])
    qfit = np.asarray(phot['qfit'])
    snr = np.asarray(phot['flux_fit']) / np.asarray(phot['flux_err'])
    pos_err = np.sqrt(np.asarray(phot['x_err'])**2 + np.asarray(phot['y_err'])**2)

    axes[0, 0].hist(cfit, bins=50, color='steelblue')
    axes[0, 0].axvline(0, c='k', ls=':')
    axes[0, 0].set_title('cfit  (residual at centre / flux)')

    axes[0, 1].hist(qfit, bins=50, color='steelblue')
    axes[0, 1].set_title('qfit  (sum |resid| / flux)')

    axes[1, 0].hist(np.log10(np.maximum(snr, 0.1)), bins=50, color='steelblue')
    axes[1, 0].axvline(0, c='r', ls=':', label='SNR=1')
    axes[1, 0].set_title('log10(flux SNR)')
    axes[1, 0].legend()

    axes[1, 1].hist(pos_err, bins=50, color='steelblue')
    axes[1, 1].axvline(config['psf']['cuts_pos_err_max'], c='r', ls=':',
                       label=f'cut={config["psf"]["cuts_pos_err_max"]}')
    axes[1, 1].set_title('pos_err  (px)')
    axes[1, 1].legend()
    plt.tight_layout()
    return fig


def show_detections_overlay(cd, phot, pass_mask=None):
    """Overlay all detections on the input data, green=pass, red=fail."""
    fig, ax = plt.subplots(figsize=(9, 9))
    data = cd.sersic_residual
    vmin, vmax = np.nanpercentile(data, [5, 99])
    offset = max(0.0, -vmin) + 1e-3
    ax.imshow(data + offset,
              norm=LogNorm(vmin=vmin + offset, vmax=vmax + offset),
              cmap='gray_r', origin='lower')
    if phot is not None and len(phot) > 0:
        x = np.asarray(phot['x_fit'])
        y = np.asarray(phot['y_fit'])
        if pass_mask is None:
            ax.scatter(x, y, facecolors='none', edgecolors='cyan',
                       s=80, lw=1.2, label=f'detected ({len(phot)})')
        else:
            ax.scatter(x[pass_mask], y[pass_mask],
                       facecolors='none', edgecolors='lime',
                       s=80, lw=1.2, label=f'pass ({pass_mask.sum()})')
            ax.scatter(x[~pass_mask], y[~pass_mask],
                       facecolors='none', edgecolors='red',
                       s=80, lw=1.2, label=f'fail ({(~pass_mask).sum()})')
    ax.legend(loc='upper right')
    ax.set_title(f'{cd.filtername}: PSF detections (sersic_residual background)')
    return fig


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def diagnose(filtername='F150W', sphot_file='g260_sphot.h5', th=None,
             show=True, save_dir=None):
    """Run all diagnostics for one filter. If `show=False` and `save_dir` is
    given, write each figure to disk instead of displaying.
    """
    galaxy = load_galaxy(sphot_file)
    cd = galaxy.images[filtername]

    print(f'\n========= {filtername} =========')
    print(f'PSF blur (σ)    : {cd.psf_blurring:.3f} px')
    print(f'PSF size (σ_eq) : {cd.psf_sigma:.3f} px')
    print(f'galaxy_size     : {cd.galaxy_size:.1f} px')
    print(f'data shape      : {cd.data.shape}')

    figs = {}
    figs['psf'] = plt.figure(figsize=(4, 4))
    show_psf(cd, ax=figs['psf'].gca())
    figs['input_resid'], _ = show_input_and_residual(cd)

    threshold_scan(cd)

    phot, _resid, bkg_std = run_psf_photometry(cd, th=th)
    pass_mask = quality_cut_breakdown(phot, cd, bkg_std)

    figs['hist'] = quality_histograms(phot)
    figs['bright'] = show_brightest_stars(cd, phot, n=6)
    figs['overlay'] = show_detections_overlay(cd, phot, pass_mask)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figs.items():
            if fig is None:
                continue
            fig.savefig(os.path.join(save_dir, f'{filtername}_{name}.png'),
                        dpi=120, bbox_inches='tight')
            plt.close(fig)
    elif show:
        plt.show()

    return galaxy, cd, phot, pass_mask


if __name__ == '__main__':
    filt = sys.argv[1] if len(sys.argv) > 1 else 'F150W'
    sphot_file = sys.argv[2] if len(sys.argv) > 2 else 'g260_sphot.h5'
    diagnose(filt, sphot_file)
