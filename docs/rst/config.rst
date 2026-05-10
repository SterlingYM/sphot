User config with ``sphot_config.toml``
======================================

Sphot is configured via a TOML file (``sphot_config.toml``) placed in the working directory. If a key is omitted, the value from the bundled default (``sphot/default_config.toml``) is used. Below is the full default config, followed by per-block reference tables.

.. code-block:: toml

    [prep]
    filters = ['F555W', 'F814W', 'F090W', 'F150W', 'F160W', 'F277W']
    PSF_file = 'PSFdata.h5'
    blur_psf = {}                          # empty: calibrator bootstrap picks per-filter blur
    custom_initial_crop = 1
    sigma_guess = 10
    auto_crop = true
    auto_crop_factor = 8                   # cutout half-axis in σ of best-fit Gaussian

    [core]
    base_filter = 'F150W'
    iter_basefit = 10
    iter_scalefit = 3
    fit_complex_model = false
    allow_refit = false
    mainloop_convergence_atol = 1e-3
    mainloop_convergence_patience = 2
    mainloop_min_iter = 2
    use_dual_annealing = true
    use_final_polish = true

    [psf-calib]
    in_mainloop = false
    kernel_family = 'gaussian'
    kernel_anchor_K = 30
    kernel_fit_skip_on_convergence = false
    kernel_skip_threshold = 1e-3
    kernel_fit_objective = 'multi_source'
    kernel_fit_stamp_half_min = 8
    kernel_fit_stamp_half_in_sigmaeff = 3.0
    kernel_fit_max_neighbors = 15
    kernel_fit_neighbor_pad_pix = 4.0
    center_mask_r_in_fwhm = 2.5
    scan_method = 'lm'
    scan_metric = 'sum_snr'
    scan_top_k = 100
    fwhm_method = 'empirical'

    [psf]
    th_min = 5
    th_max = 30
    th_iter = 10
    th_max_consec_empty = 3
    bkg_refit_per_iteration = true
    bkg_floor_factor = 0.3
    ladder_dedup_psfsigma = 0.5
    final_refit_method = 'iterative'
    final_joint_refit = true
    final_refit_xy_bounds = 0.0
    final_refit_iterations = 5
    final_refit_detect_th = 4.0
    final_refit_dedup_psfsigma = 2.0
    final_refit_residual_tol = 0.01
    final_refit_catastrophic_flux_factor = 20.0
    final_refit_fit_shape = [11, 11]
    leftover_max_iterations = 3
    leftover_detect_k_sigma = 5.0
    leftover_dedup_psfsigma = 1.5
    leftover_box_size = 3
    mask_aper_size_in_r_eff = 3.5
    raise_error = false
    residual_clip_sigma = 15.0
    finder_kwargs = {roundness_range=[-1.0, 1.0], sharpness_range=[0.20, 1.0]}
    grouper_separation_in_psfsigma = 2
    bkg_sigma_clip = 3.0
    bkg_box_size = [10, 10]
    bkg_filter_size = [5, 5]
    localbkg_bounds_in_psfsigma = [2, 5]
    center_mask_r_in_fwhm = 1.5
    PSFPhotometry_aperture_radius = 3
    PSFPhotometry_fitter_maxiters = 300
    PSFPhotometry_fit_shape_in_fwhm = 1.5
    modelimg_render_shape = [25, 25]
    PSFPhotometry_group_warning_threshold = 10
    cuts_pos_err_max = 2
    cuts_res_cen_sigma_clip = 3.0
    cuts_cfit_sigma_clip = 3.0
    cuts_chi2dof_median_factor = 3.0
    cuts_pos_diff_median_factor = 2.0
    cuts_flux_SNR_min = 1.0

    [aperture]
    petro = 0.5
    center_mask = 4.0
    plot = true
    isophot_base_filter = 'F150W'
    isophot_frac_min = 0.05
    isophot_frac_max = 0.95
    fit_isophot_to = 'sersic_modelimg'
    measure_on = 'psf_sub_data'
    error_on = 'psf_sub_data_error'
    measure_sky_on = 'residual_masked'
    fill_max_nan_frac = 0.7
    fill_replace_with = 'median'
    PSF_corr_base_filter = 'F090W'


``[prep]`` block — input data and pre-processing
------------------------------------------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - ``filters``
     - list[str]
     - Filter names to load from the input HDF5 cube (e.g. ``['F555W', 'F814W', 'F090W', 'F150W', 'F160W', 'F277W']``).
   * - ``PSF_file``
     - str
     - Path to the PSF HDF5 file. Per-filter PSFs must be stored at integer-multiple pixel scale relative to the science image.
   * - ``blur_psf``
     - dict[str, float]
     - Per-filter Gaussian blur σ (oversampled px) applied to the library PSF. ``{}`` lets the calibrator bootstrap a per-filter blur from the data.
   * - ``custom_initial_crop``
     - float
     - Fraction of the input image to crop to before any other prep step. Range ``(0, 1]``; ``1`` disables.
   * - ``sigma_guess``
     - float
     - Initial guess for the galaxy Gaussian σ (px), used to seed the auto-crop sizing fit.
   * - ``auto_crop``
     - bool
     - If true, fit a coarse Gaussian to the galaxy and re-crop the cutout to ``auto_crop_factor × σ`` per axis.
   * - ``auto_crop_factor``
     - float
     - Cutout half-axis size in units of the best-fit Gaussian σ.


``[core]`` block — top-level pipeline behaviour
-----------------------------------------------

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - ``base_filter``
     - str
     - Filter used for the base 8-parameter Sersic fit. Other filters are scale-fitted against this template.
   * - ``iter_basefit``
     - int
     - Max iterations of the basefit main loop (Sersic fit ↔ PSF photometry alternation).
   * - ``iter_scalefit``
     - int
     - Max iterations of the scalefit main loop for non-base filters.
   * - ``fit_complex_model``
     - bool
     - If true, allow the optional Sersic + secondary-component complex model. Default ``false`` (single Sersic).
   * - ``allow_refit``
     - bool
     - If true, re-run the basefit Sersic from a perturbed init when the first pass looks degenerate.
   * - ``mainloop_convergence_atol``
     - float
     - Early-stop threshold on the L∞ norm of standardized Sersic-parameter changes between iterations.
   * - ``mainloop_convergence_patience``
     - int
     - Number of consecutive within-atol iterations required before early-stopping.
   * - ``mainloop_min_iter``
     - int
     - Hard floor; never early-stop before this many main-loop iterations.
   * - ``use_dual_annealing``
     - bool
     - If true, the iter-1 main-loop Sersic fit runs a short iNM bracket followed by ``scipy.optimize.dual_annealing`` (single pass, ``maxiter=10``) to escape false local basins. Adds runtime cost but markedly improves robustness when the iter-0 fit lands in a wrong-``n`` basin.
   * - ``use_final_polish``
     - bool
     - If true, run an L-BFGS-B polish on the final Sersic params after the main loop, followed by one PSF photometry pass.


``[psf-calib]`` block — per-iteration PSF kernel recalibration
--------------------------------------------------------------

.. list-table::
   :widths: 35 15 50
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - ``in_mainloop``
     - bool
     - If true, recalibrate the kernel + matched-filter FWHM on every main-loop iteration. If false, only an initial calibration is performed.
   * - ``kernel_family``
     - str
     - PSF blur kernel family: ``'gaussian'``, ``'moffat'``, or ``'drizzle'``.
   * - ``kernel_anchor_K``
     - int
     - Number of brightest anchor sources used in the kernel fit.
   * - ``kernel_fit_skip_on_convergence``
     - bool
     - If true, skip the FWHM scan once the kernel parameter L2 distance between iterations falls below ``kernel_skip_threshold``.
   * - ``kernel_skip_threshold``
     - float
     - L2 threshold for the above shortcut.
   * - ``kernel_fit_objective``
     - str
     - ``'multi_source'`` (NNLS lstsq joint over neighbours) or ``'single_source'`` (legacy LM, one source at a time).
   * - ``kernel_fit_stamp_half_min``
     - int
     - Minimum stamp half-size (px) when fitting per-source kernel residuals.
   * - ``kernel_fit_stamp_half_in_sigmaeff``
     - float
     - Adaptive stamp half-size in units of √(``psf_sigma²`` + ``σ_kernel²``).
   * - ``kernel_fit_max_neighbors``
     - int
     - Max number of neighbour columns folded into each multi-source NNLS stamp.
   * - ``kernel_fit_neighbor_pad_pix``
     - float
     - Search radius beyond the stamp (px) for neighbour candidates.
   * - ``center_mask_r_in_fwhm``
     - float
     - Calibration centre-mask radius in units of empirical effective PSF FWHM. ``0`` disables. The mask hides Sersic central residuals from the DAO finder during kernel calibration.
   * - ``scan_method``
     - str
     - FWHM-scan inner photometry: ``'lm'`` (full ``photutils.PSFPhotometry``) or ``'nnls'`` (forced photometry at DAO positions, no SourceGrouper, no LM iters). ``'nnls'`` is much faster and roughly equivalent in quality.
   * - ``scan_metric``
     - str
     - Bracket-candidate ranking when ``scan_method='nnls'``: ``'sum_snr'`` (max ``Σ(flux/flux_err)``), ``'resid_mad'`` (min residual MAD), or ``'count'`` (max # of sources passing ``cuts_flux_SNR_min``; saturates with ``scan_top_k``).
   * - ``scan_top_k``
     - int
     - Cap DAO detections to the brightest ``K`` before scoring. ``0`` disables the cap.
   * - ``fwhm_method``
     - str
     - How to set the matched-filter ``dao_fwhm_factor``: ``'scan'`` runs the bracket scan; ``'empirical'`` skips the scan entirely and sets the factor to ``fit_fwhm(cd.psf) / psf_sigma`` (matched-filter theorem). ``'empirical'`` is significantly faster and is the recommended default.


``[psf]`` block — PSF photometry, threshold ladder, refit
---------------------------------------------------------

Detection ladder and background re-estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 35 15 50
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - ``th_min``
     - float
     - Minimum detection threshold (× background σ) for the bottom of the ladder.
   * - ``th_max``
     - float
     - Maximum detection threshold; the top of the ladder.
   * - ``th_iter``
     - int
     - Number of ladder steps from ``th_max`` down to ``th_min``.
   * - ``th_max_consec_empty``
     - int
     - Stop the ladder early after ``N`` consecutive passes find no new sources.
   * - ``bkg_refit_per_iteration``
     - bool
     - If true, re-estimate the background σ after each successful ladder pass.
   * - ``bkg_floor_factor``
     - float
     - Floor on the re-estimated ``bkg_std`` as a fraction of the initial estimate (prevents collapse).
   * - ``ladder_dedup_psfsigma``
     - float
     - In-ladder duplicate-source cut radius (× ``psf_sigma``). ``0`` disables.

Final refit
~~~~~~~~~~~

.. list-table::
   :widths: 40 15 45
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - ``final_refit_method``
     - str
     - ``'iterative'`` (joint LM refit + leftover loop), ``'nnls'`` (NNLS forced refit + ``find_peaks`` leftover loop), or ``'none'``.
   * - ``final_joint_refit``
     - bool
     - For ``iterative``: jointly refit all sources rather than per-group.
   * - ``final_refit_xy_bounds``
     - float
     - Position bounds (px) for the joint refit. ``0`` pins positions.
   * - ``final_refit_iterations``
     - int
     - Max joint-refit iterations.
   * - ``final_refit_detect_th``
     - float
     - Leftover-detection threshold (× residual MAD) inside the iterative refit.
   * - ``final_refit_dedup_psfsigma``
     - float
     - Leftover-detection dedup radius (× ``psf_sigma``).
   * - ``final_refit_residual_tol``
     - float
     - Stop refit iterations when residual MAD improves by less than this fraction.
   * - ``final_refit_catastrophic_flux_factor``
     - float
     - If a source's flux changes by more than this factor between iterations, mark the refit catastrophic and roll back.
   * - ``final_refit_fit_shape``
     - [int, int]
     - Joint-refit fit window (px).
   * - ``leftover_max_iterations``
     - int
     - For NNLS refit: max number of post-NNLS leftover detection passes. ``0`` disables.
   * - ``leftover_detect_k_sigma``
     - float
     - Leftover detection threshold (× residual σ) for the NNLS leftover loop.
   * - ``leftover_dedup_psfsigma``
     - float
     - Leftover dedup radius (× ``psf_sigma``).
   * - ``leftover_box_size``
     - int
     - ``find_peaks`` box size for leftover detection.

Masks, background, and detection knobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 35 15 50
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - ``mask_aper_size_in_r_eff``
     - float
     - Sky-mask aperture size in units of the Sersic ``r_eff``.
   * - ``raise_error``
     - bool
     - If true, raise on PSF-photometry errors instead of warning and continuing.
   * - ``residual_clip_sigma``
     - float
     - Sigma used to clip residual outliers; high default avoids over-masking the Sersic core.
   * - ``finder_kwargs``
     - dict
     - Passed to ``DAOStarFinder``: e.g. ``roundness_range`` and ``sharpness_range`` cuts.
   * - ``grouper_separation_in_psfsigma``
     - float
     - ``SourceGrouper`` minimum separation (× ``psf_sigma``).
   * - ``bkg_sigma_clip``
     - float
     - Sigma used by the 2D ``Background2D`` estimator.
   * - ``bkg_box_size``
     - [int, int]
     - 2D background mesh box size.
   * - ``bkg_filter_size``
     - [int, int]
     - 2D background median-filter size.
   * - ``localbkg_bounds_in_psfsigma``
     - [float, float]
     - Inner/outer annulus radii for ``LocalBackground``, in units of ``psf_sigma``.
   * - ``center_mask_r_in_fwhm``
     - float
     - PSF-photometry centre-mask radius (× empirical effective PSF FWHM); hides the Sersic core from PSF detection. ``0`` disables.

PSFPhotometry knobs
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 15 45
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - ``PSFPhotometry_aperture_radius``
     - float
     - Aperture radius (px) used for the initial-flux estimate inside ``PSFPhotometry``.
   * - ``PSFPhotometry_fitter_maxiters``
     - int
     - Max LM iterations per source/group.
   * - ``PSFPhotometry_fit_shape_in_fwhm``
     - float
     - Fit-window size in units of FWHM. Internally clamped to ``max(3, next_odd(2.355·psf_sigma·frac))``.
   * - ``modelimg_render_shape``
     - [int, int]
     - Render shape (px) for the per-source PSF model image used in the final residual.
   * - ``PSFPhotometry_group_warning_threshold``
     - int
     - Warn when a single ``SourceGrouper`` group exceeds this many members (LM degrades on large groups).

Quality cuts (post-fit)
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 35 15 50
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - ``cuts_pos_err_max``
     - float
     - Max fitted x/y error (px). Sources beyond this are discarded.
   * - ``cuts_res_cen_sigma_clip``
     - float
     - Sigma clip on residual values at fitted source centres.
   * - ``cuts_cfit_sigma_clip``
     - float
     - Sigma clip on the ``cfit`` quality metric.
   * - ``cuts_chi2dof_median_factor``
     - float
     - Drop sources with ``chi2dof`` exceeding this factor × the median ``chi2dof``.
   * - ``cuts_pos_diff_median_factor``
     - float
     - Drop sources whose positional drift between iterations exceeds this factor × the median.
   * - ``cuts_flux_SNR_min``
     - float
     - Minimum ``flux / flux_err``. Sources below this are discarded.


``[aperture]`` block — aperture photometry
------------------------------------------

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - ``petro``
     - float
     - Petrosian fraction used to define the aperture radius.
   * - ``center_mask``
     - float
     - Aperture centre-mask radius (px); hides the saturated/PSF core from aperture sums.
   * - ``plot``
     - bool
     - If true, generate diagnostic aperture plots.
   * - ``isophot_base_filter``
     - str
     - Filter used as the reference for isophote fitting.
   * - ``isophot_frac_min``
     - float
     - Minimum flux-fraction level fit by ``photutils.isophote``.
   * - ``isophot_frac_max``
     - float
     - Maximum flux-fraction level.
   * - ``fit_isophot_to``
     - str
     - Image type the isophotes are fit to; usually ``'sersic_modelimg'``.
   * - ``measure_on``
     - str
     - Image type aperture flux is measured on, typically ``'psf_sub_data'`` (raw − PSF model).
   * - ``error_on``
     - str
     - Error image used for aperture-flux uncertainties.
   * - ``measure_sky_on``
     - str
     - Image type used to estimate the local sky inside the aperture annulus.
   * - ``fill_max_nan_frac``
     - float
     - Max fraction of NaN pixels permitted inside the aperture before rejecting the measurement.
   * - ``fill_replace_with``
     - str
     - How to fill NaNs prior to aperture sum: ``'median'`` (annulus median) or ``'sersic_modelimg'``.
   * - ``PSF_corr_base_filter``
     - str
     - Filter used as reference for cross-filter PSF-aperture corrections.
