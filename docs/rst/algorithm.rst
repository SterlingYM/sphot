==================
Algorithm overview
==================

Sphot is a **joint Sersic + multi-source PSF + sky** fitter that alternates
three sub-problems until they stop fighting each other:

1. fit a smooth Sersic galaxy model to the image **after** point-source light
   has been removed,
2. fit a forest of point-source PSFs to the image **after** the Sersic galaxy
   has been removed,
3. estimate and subtract a residual sky from what's left.

Each step is straightforward in isolation. The trick is that they share the
same pixels, so getting any one of them wrong biases the other two. Sphot
solves the deadlock by **iterating** them — each pass uses the cleanest
estimate of the *other* components currently available — and by **calibrating
the PSF kernel itself** as it goes (because the input library PSF is rarely an
exact match for the data).

The rest of this page walks through the structure of that iteration, from the
top-level pipeline down to a single PSF photometry pass.

------------------------------------------------------------------

Top-level pipeline
==================

The command-line entry point ``run_sphot`` (or its programmatic equivalent in
``sphot.run_sphot.run_sphot``) drives a galaxy through four stages:

.. mermaid::

    flowchart TB
        H[("input HDF5<br/>(galaxy cutouts + library PSF)")]:::io
        H --> L["load_and_crop<br/><i>per-filter cutouts, auto-crop, blur</i>"]:::step
        L --> G["Galaxy<br/><i>cd[filter] = CutoutData</i>"]:::data
        G --> B["run_basefit<br/><b>base filter only</b>"]:::heavy
        B --> P["base_params + psf_table<br/><i>(Sersic shape + source list)</i>"]:::data
        P --> S["run_scalefit<br/><b>parallel, all other filters</b>"]:::heavy
        S --> A["run_aperphot<br/><i>(optional)</i>"]:::step
        A --> O[("output sphot.h5<br/>+ results CSV")]:::io

        click L "../api/utils.html#sphot.utils.load_and_crop" "load_and_crop in sphot.utils"
        click B "#run-basefit" "Iterative Sersic + PSF fit on the base filter"
        click S "#run-scalefit" "Pin Sersic shape, fit per-filter scale + PSF photometry"
        click A "#run-aperphot" "Aperture photometry on psf_sub_data"

        classDef io      fill:#1f2a44,stroke:#1f2a44,color:#fff,rx:8,ry:8;
        classDef data    fill:#eef3fb,stroke:#5b7fb8,color:#1f2a44,rx:8,ry:8;
        classDef step    fill:#e8f3ec,stroke:#3b8a5a,color:#143824,rx:8,ry:8;
        classDef heavy   fill:#ffe8c2,stroke:#c97a13,color:#5a3604,rx:8,ry:8,stroke-width:2px;

The two heavy boxes — ``run_basefit`` and ``run_scalefit`` — share most of
their structure. The base fit is run once and produces both a *Sersic shape*
(``sersic_params``) and a *source catalogue* (``psf_table``). The scale fit
then runs in parallel across the remaining filters with the Sersic shape
**pinned** to the base solution, fitting only an overall brightness scale (and
optionally a small re-fit) plus per-filter PSF photometry.

.. note::

   For very blurry IR filters where source detection is unreliable, the
   variant :func:`sphot.core.run_scalefit_forced` skips detection entirely and
   force-extracts flux at the base filter's source positions. See
   :ref:`forced-scalefit` below.

------------------------------------------------------------------

.. _run-basefit:

The base fit (``run_basefit``)
==============================

This is where almost all of the work happens. It loads the base filter,
builds an initial Sersic model and PSF photometry fitter, runs a primer
fit on the raw data, then enters the **main loop** — alternating Sersic,
PSF, sky, and (optionally) PSF-kernel calibration until the Sersic
parameters stop moving.

.. container:: sphot-grid sphot-grid-3

   .. container:: sphot-col

      1. Init + primer fit

      .. mermaid::

         flowchart TB
             I["init<br/>perform_bkg_stats · blur_psf<br/>seed dao_fwhm_factor<br/>build fitter_1, fitter_2, fitter_psf"]:::init
             I --> S0["fitter_1.fit<br/><b>Sersic on RAW data</b><br/><i>fit_to = data</i>"]:::sersic
             S0 --> R0["remove_sky · sersic"]:::sky
             R0 --> P0["fitter_psf.fit<br/><i>fit_to = sersic_residual</i>"]:::psf
             P0 --> Q0["remove_sky · psf"]:::sky
             Q0 --> C0["recalibrate_psf<br/><i>(if psf-calib.in_mainloop)</i>"]:::cal
             C0 --> CONT(["→ continues in column 2:<br/>Main loop"]):::cont
             click P0 "#fitter-psf-fit" "PSF photometry sub-step"
             click C0 "#recalibrate-psf" "Kernel recalibration sub-step"
             classDef init   fill:#f4ecff,stroke:#6f4cc2,color:#2c1a55,rx:8,ry:8;
             classDef sersic fill:#fff1d6,stroke:#c97a13,color:#5a3604,rx:8,ry:8;
             classDef psf    fill:#e1efff,stroke:#2b69b3,color:#0d2748,rx:8,ry:8;
             classDef sky    fill:#e8f3ec,stroke:#3b8a5a,color:#143824,rx:8,ry:8;
             classDef cal    fill:#ffe1ec,stroke:#b53274,color:#451026,rx:8,ry:8;
             classDef cont   fill:#444,stroke:#222,color:#fff,rx:8,ry:8;

   .. container:: sphot-col

      2. Main loop

      .. mermaid::

         flowchart TB
             PREV(["← from column 1:<br/>Init + primer fit"]):::cont
             PREV --> LOOP{{"<b>Main loop</b><br/>i = 0 to N_mainloop_iter"}}
             LOOP --> BR{"iter 0 and<br/>use_dual_annealing?"}
             BR -->|yes| DA["fitter_2.fit<br/><b>method = dual_annealing</b><br/><i>fit_to = psf_sub_data</i>"]:::sersic
             BR -->|no| NM["fitter_2.fit<br/><b>method = iterative_NM</b><br/><i>fit_to = psf_sub_data</i>"]:::sersic
             DA --> R1["remove_sky · sersic"]:::sky
             NM --> R1
             R1 --> P1["fitter_psf.fit<br/><i>threshold ladder + final refit</i>"]:::psf
             P1 --> Q1["remove_sky · psf"]:::sky
             Q1 --> C1["recalibrate_psf<br/><i>(if psf-calib.in_mainloop)</i>"]:::cal
             C1 --> CONV{"sersic params change<br/>under atol for<br/>patience iters?"}
             CONV -->|"keep going"| LOOP
             CONV -->|"converged or i = N"| NEXT(["→ continues in column 3:<br/>Polish + done"]):::cont
             click P1 "#fitter-psf-fit" "PSF photometry sub-step"
             click C1 "#recalibrate-psf" "Kernel recalibration sub-step"
             click DA "../api/fitting.html#sphot.fitting.ModelFitter.fit" "ModelFitter.fit (method='dual_annealing')"
             click NM "../api/fitting.html#sphot.fitting.ModelFitter.fit" "ModelFitter.fit (method='iterative_NM')"
             classDef sersic fill:#fff1d6,stroke:#c97a13,color:#5a3604,rx:8,ry:8;
             classDef psf    fill:#e1efff,stroke:#2b69b3,color:#0d2748,rx:8,ry:8;
             classDef sky    fill:#e8f3ec,stroke:#3b8a5a,color:#143824,rx:8,ry:8;
             classDef cal    fill:#ffe1ec,stroke:#b53274,color:#451026,rx:8,ry:8;
             classDef cont   fill:#444,stroke:#222,color:#fff,rx:8,ry:8;

   .. container:: sphot-col

      3. Polish + done

      .. mermaid::

         flowchart TB
             PREV(["← from column 2:<br/>Main loop"]):::cont
             PREV --> EXIT["exit loop"]
             EXIT --> POL{"use_final_polish?"}
             POL -->|yes| LB["fitter_2.fit<br/><b>method = lbfgsb_polish</b>"]:::sersic
             LB --> RF["remove_sky · sersic"]:::sky
             RF --> PF["fitter_psf.fit<br/><i>final pass</i>"]:::psf
             PF --> QF["remove_sky · psf"]:::sky
             POL -->|no| QF
             QF --> DONE(["cd.sersic_params<br/>cd.psf_table<br/>cd.psf_sub_data<br/>cd.psf_modelimg<br/>cd.residual_masked"]):::data
             click PF "#fitter-psf-fit" "PSF photometry sub-step"
             click LB "../api/fitting.html#sphot.fitting.ModelFitter.fit" "ModelFitter.fit (method='lbfgsb_polish')"
             classDef sersic fill:#fff1d6,stroke:#c97a13,color:#5a3604,rx:8,ry:8;
             classDef psf    fill:#e1efff,stroke:#2b69b3,color:#0d2748,rx:8,ry:8;
             classDef sky    fill:#e8f3ec,stroke:#3b8a5a,color:#143824,rx:8,ry:8;
             classDef data   fill:#eef3fb,stroke:#5b7fb8,color:#1f2a44,rx:8,ry:8;
             classDef cont   fill:#444,stroke:#222,color:#fff,rx:8,ry:8;

Why the structure looks like this
---------------------------------

* **The first Sersic fit uses raw data.** Before any PSF photometry has run,
  there is no ``psf_sub_data`` to fit against — and in any case, the dominant
  signal at this stage is the galaxy itself. ``fitter_1`` is the *simple*
  Sersic model and is enough to seed everything that follows.

* **Iter 0 of the main loop uses dual annealing.** By then, ``psf_sub_data``
  is clean enough that the chi² landscape has only a handful of basins, and
  global escape is cheap. Subsequent iterations use ``iterative_NM`` because
  Sersic params are already in the right basin and we just need them to
  converge. (Toggle: ``[core].use_dual_annealing``.)

* **Sky is removed twice per iteration.** The first call subtracts a sky that
  best fits ``residual_masked`` and applies it to ``sersic_residual``; the
  second applies a sky to ``psf_sub_data``. Decoupling the two prevents the
  PSF-fit residual from biasing the Sersic-fit sky and vice versa.

* **Kernel recalibration runs at the end of each iteration**, never in the
  middle. After every recal, ``cd.sersic_modelimg`` is *re-rendered* (not
  re-fit) against the new effective PSF so downstream code sees a
  self-consistent state. Toggle: ``[psf-calib].in_mainloop`` — defaults to
  ``false`` in the library default config and ``true`` in the test config.

* **The L-BFGS-B polish is opt-in** and runs *outside* the main loop. It buys
  the last few percent of chi² that ``iterative_NM`` and ``dual_annealing``
  leave on the table. The polish is followed by one more PSF photometry +
  sky pass so the saved ``psf_modelimg`` matches the polished Sersic.

Convergence and early-exit
--------------------------

Three knobs in ``[core]`` control when the loop stops:

* ``mainloop_convergence_atol`` — the L∞ tolerance on standardized Sersic
  parameter changes between iterations.
* ``mainloop_convergence_patience`` — how many *consecutive* within-tolerance
  iterations are required before bailing out.
* ``mainloop_min_iter`` — a hard floor on the number of iterations regardless
  of convergence.

Together these make early-exit conservative: a single quiet iteration is
never enough.

References
----------

* :func:`sphot.core.run_basefit` — the function this section describes.
* :class:`sphot.fitting.ModelFitter` — Sersic fitter; see :doc:`/api/fitting`
  for the available ``method`` values and what they do.
* :doc:`/rst/config` — every knob mentioned above is documented under
  ``[core]`` and ``[psf-calib]``.

------------------------------------------------------------------

.. _run-scalefit:

The scale fit (``run_scalefit``)
================================

Once the base fit has produced a Sersic shape, the *shape* is held fixed and
only an overall brightness scale (plus a small free offset) is fit per
filter. This is much cheaper than a full Sersic re-fit and avoids the band
ambiguities that plague free joint fits.

.. mermaid::

    flowchart TB
        IN[("base_params<br/>(from run_basefit)")]:::data
        IN --> SI["init<br/>perform_bkg_stats · blur_psf · seed dao_fwhm<br/>build fitter_scale (+ optional fitter_2), fitter_psf"]:::init
        SI --> S0["fitter_scale.fit<br/><i>fit_to = data</i>"]:::sersic
        S0 --> X0["remove_sky · sersic →<br/>fitter_psf.fit · remove_sky · psf →<br/>recalibrate_psf"]:::cycle
        X0 --> AR{"allow_refit?"}
        AR -->|yes| RE["fitter_2.fit (free Sersic refit)<br/>+ remove_sky + fitter_psf.fit + remove_sky"]:::sersic
        AR -->|no| LOOP{{"<b>Main loop</b><br/>i = 0 to N_mainloop_iter"}}
        RE --> LOOP

        LOOP --> ST["fitter_scale.fit (or fitter_2 if allow_refit)<br/><i>fit_to = psf_sub_data</i>"]:::sersic
        ST --> CYC["remove_sky · sersic →<br/>fitter_psf.fit · remove_sky · psf →<br/>recalibrate_psf"]:::cycle
        CYC --> CONV{"converged?"}
        CONV -->|no| LOOP
        CONV -->|yes| EXIT["exit loop"]
        EXIT --> CLN["cleanup: remove_sky · sersic →<br/>fitter_psf.fit · remove_sky · psf<br/><i>(if calibrated in-loop)</i>"]:::cycle
        CLN --> DONE(["cd.sersic_params (scaled)<br/>cd.psf_table<br/>cd.psf_sub_data"]):::data

        click ST "../api/fitting.html#sphot.fitting.ModelScaleFitter" "ModelScaleFitter — pins Sersic shape, fits scale + small offset"
        click RE "../api/fitting.html#sphot.fitting.ModelFitter" "ModelFitter — free Sersic re-fit, used only if core.allow_refit=true"

        classDef data   fill:#eef3fb,stroke:#5b7fb8,color:#1f2a44,rx:8,ry:8;
        classDef init   fill:#f4ecff,stroke:#6f4cc2,color:#2c1a55,rx:8,ry:8;
        classDef sersic fill:#fff1d6,stroke:#c97a13,color:#5a3604,rx:8,ry:8;
        classDef cycle  fill:#e8f3ec,stroke:#3b8a5a,color:#143824,rx:8,ry:8;

There is no ``dual_annealing`` or L-BFGS-B polish in scalefit: the
fixed-shape problem is convex enough in ``ModelScaleFitter``'s reduced
parameter space (typically 1–3 free params) that ``iterative_NM`` finds the
optimum directly.

In ``run_sphot`` (the CLI orchestrator), all non-base filters are scalefit
**in parallel** via :func:`sphot.parallel.parallel_scalefit`, with each
worker writing its progress to a per-filter log and merging the results back
into the shared HDF5 at the end.

References
----------

* :func:`sphot.core.run_scalefit` — the loop above.
* :class:`sphot.fitting.ModelScaleFitter` — the pinned-shape fitter.
* :func:`sphot.parallel.parallel_scalefit` — the parallel driver.

------------------------------------------------------------------

.. _forced-scalefit:

Forced-position scalefit (``run_scalefit_forced``)
==================================================

For filters where DAO source detection is unreliable (typically wide IR
bands where the PSF is too blurry to resolve crowded fields), this variant
**skips detection entirely** and force-extracts the flux at every
quality-passing position from the base filter's ``psf_table``:

.. mermaid::

    flowchart LR
        BPHOT[("base_filter.psf_table")]:::data
        BPHOT --> FLT["filter rows by photutils flag bits<br/>(2|4|32|64|128|256) · finite + flux>0"]:::step
        FLT --> POS["base_x, base_y, base_f"]:::data
        POS --> LOOP["main loop:<br/>fitter_scale.fit →<br/>remove_sky · sersic →<br/>run_forced_photometry_on_cutout(base_x, base_y, flux_init=base_f) →<br/>remove_sky · psf →<br/>(optional) recalibrate_psf"]:::cycle
        LOOP --> DONE(["cd.psf_table (forced positions, refit fluxes)"]):::data

        click LOOP "../api/psf.html#sphot.psf.run_forced_photometry_on_cutout" "Forced PSF photometry: pinned positions, NNLS-style flux solve"

        classDef data   fill:#eef3fb,stroke:#5b7fb8,color:#1f2a44,rx:8,ry:8;
        classDef step   fill:#e8f3ec,stroke:#3b8a5a,color:#143824,rx:8,ry:8;
        classDef cycle  fill:#e1efff,stroke:#2b69b3,color:#0d2748,rx:8,ry:8;

This bypasses the threshold ladder and the leftover-detection loop; PSF
positions are completely pinned. Use it when the goal is **consistency
across filters**, not maximum sensitivity within a single filter.

References
----------

* :func:`sphot.core.run_scalefit_forced`
* :func:`sphot.psf.run_forced_photometry_on_cutout`

------------------------------------------------------------------

.. _fitter-psf-fit:

PSF photometry sub-step (``fitter_psf.fit``)
============================================

This is the inner workhorse called from every Sersic ↔ PSF iteration in both
basefit and scalefit. It implements **threshold-ladder PSF photometry**
followed by an optional **simultaneous refit** that re-solves all source
fluxes at once (and optionally hunts for leftover sources missed by the
ladder).

.. container:: sphot-grid sphot-grid-2

   .. container:: sphot-col

      1. Threshold ladder

      .. mermaid::

         flowchart TB
             IN[("inputs:<br/>cd.sersic_residual · cd.psf · cd.bkg_std")]:::data
             IN --> SETUP["build psf_model from cd.psf<br/>build threshold list (geomspace<br/>th_max → th_min, th_iter steps)<br/>build galaxy-centre mask"]:::step
             SETUP --> LADDER{{"Threshold ladder<br/><b>for th in threshold_list:</b>"}}
             LADDER --> DAO["DAOStarFinder detect at<br/>threshold·bkg_std (matched filter<br/>via dao_fwhm_factor·psf_sigma)"]:::detect
             DAO --> EMPTY{"any detections?"}
             EMPTY -->|"no, N consec"| LSTOP["early-stop ladder<br/>(th_max_consec_empty)"]:::detect
             EMPTY -->|yes| FIT["photutils PSFPhotometry (LM)<br/>local-bg, fit_shape,<br/>finder_kwargs"]:::detect
             FIT --> DEDUP["dedup vs accumulated sources<br/>(ladder_dedup_psfsigma)"]:::detect
             DEDUP --> ACC["append to phot_result<br/>subtract psf_modelimg from resid"]:::detect
             ACC --> BKR["re-estimate bkg_std<br/>(bkg_floor_factor floor)"]:::detect
             BKR --> LADDER
             LSTOP --> NEXT(["→ continues in column 2:<br/>Final refit + write"]):::cont
             LADDER -.->|done| NEXT
             click LADDER "../api/psf.html#sphot.psf.iterative_psf_fitting" "iterative_psf_fitting wraps the ladder"
             click DAO "../api/psf.html#sphot.psf.do_psf_photometry" "do_psf_photometry: one ladder pass"
             classDef data   fill:#eef3fb,stroke:#5b7fb8,color:#1f2a44,rx:8,ry:8;
             classDef step   fill:#f4ecff,stroke:#6f4cc2,color:#2c1a55,rx:8,ry:8;
             classDef detect fill:#e1efff,stroke:#2b69b3,color:#0d2748,rx:8,ry:8;
             classDef cont   fill:#444,stroke:#222,color:#fff,rx:8,ry:8;

   .. container:: sphot-col

      2. Final refit + write

      .. mermaid::

         flowchart TB
             PREV(["← from column 1:<br/>Threshold ladder"]):::cont
             PREV --> REFIT{"final_refit_method?"}
             REFIT -->|"'iterative'"| JOINT["_final_joint_refit<br/>simultaneous LM with<br/>leftover detection<br/>(repeat until residual MAD<br/>stops improving)"]:::refit
             REFIT -->|"'nnls'"| NNLS["_final_nnls_refit<br/>Cholesky-Gram NNLS on<br/>quality-passing rows<br/>+ find_peaks leftover loop"]:::refit
             REFIT -->|"'none'"| SKIP["no refit"]:::refit
             JOINT --> OUT[("phot_table, residual")]:::data
             NNLS  --> OUT
             SKIP  --> OUT
             OUT --> WRITE["PSFFitter.fit writes onto cd:<br/>· psf_table<br/>· psf_modelimg = data − resid<br/>· psf_sub_data<br/>· residual<br/>· residual_masked"]:::write
             click NNLS "#nnls-refit" "How NNLS refit results flow back to cutoutdata"
             click JOINT "#nnls-refit" "How the iterative joint refit flows back to cutoutdata"
             classDef data   fill:#eef3fb,stroke:#5b7fb8,color:#1f2a44,rx:8,ry:8;
             classDef refit  fill:#fff1d6,stroke:#c97a13,color:#5a3604,rx:8,ry:8;
             classDef write  fill:#e8f3ec,stroke:#3b8a5a,color:#143824,rx:8,ry:8;
             classDef cont   fill:#444,stroke:#222,color:#fff,rx:8,ry:8;

Why the ladder
--------------

A single ``DAOStarFinder.find_peaks`` + ``PSFPhotometry.fit`` call would, in
a crowded field, hand the LM optimizer hundreds of overlapping sources at
once. LM is a local optimizer: with that many highly correlated parameters,
it routinely settles into compensating-pair pathologies (one source goes
hugely positive, a neighbour absorbs it as a large negative flux), leaves
residuals that look fine but a ``psf_modelimg`` that biases the next Sersic
fit.

The ladder sidesteps the failure mode by extracting sources **brightest
first**. At threshold ``th_max``, only the brightest dozen sources survive;
LM fits them cleanly because they don't overlap much. Their model is
subtracted from the residual, and the next pass operates at a slightly
lower threshold on a cleaner image, and so on. Each pass deduplicates
against everything found by earlier passes (``ladder_dedup_psfsigma``).
``[psf].bkg_refit_per_iteration`` re-estimates ``bkg_std`` after each
successful pass — important because subtracting bright sources changes the
robust noise level the lower thresholds key off.

.. _nnls-refit:

The final refit
---------------

After the ladder finishes, two things may have gone wrong:

1. *Compensating pairs may have crept in anyway*, because earlier passes
   committed to a subtraction that later passes can't change.
2. *The ladder may have missed sources* that DAO's matched-filter smoothing
   washed below threshold (typically very compact sources whose width
   doesn't match ``dao_fwhm_factor·psf_sigma``).

The final refit tackles both:

* ``final_refit_method = 'iterative'`` (default in the library config) runs
  ``_final_joint_refit``: a simultaneous LM fit on every accumulated source,
  followed by ``find_peaks`` on the residual to pick up leftovers, repeated
  until residual MAD stops improving.
* ``final_refit_method = 'nnls'`` (default in the test config) runs
  ``_final_nnls_refit``: a Cholesky-Gram non-negative least-squares solve on
  the design matrix whose columns are unit-flux PSF renderings at each
  source's pinned position. NNLS hard-bans negative fluxes, eliminating the
  compensating-pair pathology by construction. The same ``find_peaks``
  leftover loop appends new columns and re-solves.
* ``final_refit_method = 'none'`` skips this stage entirely.

Either way, the new ``(phot_table, residual)`` is what ``iterative_psf_fitting``
returns. ``PSFFitter.fit`` then derives ``psf_modelimg`` by subtraction and
writes ``psf_table``, ``psf_modelimg``, ``psf_sub_data``, ``residual``,
``residual_masked`` onto the cutoutdata. The next Sersic iteration reads
``psf_sub_data``; the kernel recalibrator reads ``psf_table``.

Quality cuts
------------

Both refit branches filter rows through :func:`sphot.psf.filter_psfphot_results`
before exposing them as the final ``psf_table``. The cuts (all in ``[psf]``)
are documented in :doc:`/rst/config`:

* ``cuts_pos_err_max`` — drop fits whose centroid uncertainty exceeds N px.
* ``cuts_res_cen_sigma_clip`` — drop sources whose central residual is more
  than σ above the typical residual.
* ``cuts_cfit_sigma_clip`` — drop fits whose χ²/dof is an outlier.
* ``cuts_chi2dof_median_factor`` / ``cuts_pos_diff_median_factor`` —
  population-level outlier cuts.
* ``cuts_flux_SNR_min`` — drop everything below a minimum flux/error.

References
----------

* :func:`sphot.psf.iterative_psf_fitting` — wraps the ladder + final refit.
* :func:`sphot.psf.do_psf_photometry` — one ladder pass.
* :func:`sphot.psf._final_joint_refit`, :func:`sphot.psf._final_nnls_refit`
  — the two final-refit branches.
* :class:`sphot.psf.PSFFitter` — the cutoutdata-level wrapper.

------------------------------------------------------------------

.. _recalibrate-psf:

Kernel recalibration (``_maybe_recalibrate_psf``)
=================================================

The library PSF supplied by the user is rarely a pixel-perfect match for the
data: telescope focus drifts, dither sub-pixel sampling, and detector
charge-diffusion all leave a residual *effective-PSF blur* that survives
PSFPhotometry's per-source LM fit. Sphot's solution is to **fit a kernel**
``K`` such that ``cd.psf = library_PSF ⊛ K`` reproduces the bright-source
residuals best, then **convolve** that kernel into the working PSF for the
next iteration.

When this runs (only if ``[psf-calib].in_mainloop = true``):

1. After each PSF photometry pass, the calibrator picks the ``K`` brightest
   anchors from ``cd.psf_table`` (``[psf-calib].kernel_anchor_K``).
2. It builds a multi-source NNLS design (``kernel_fit_objective='multi_source'``)
   or single-source LM (``='single_source'``) over small stamps around each
   anchor and fits the kernel parameters of the chosen ``kernel_family``
   (``'gaussian'`` / ``'moffat'`` / ``'drizzle'``).
3. The optional FWHM scan (``fwhm_method``, ``scan_method``) updates
   ``cd.dao_fwhm_factor`` so the next ladder's matched filter matches the
   newly-blurred PSF.
4. ``cd.psf`` is replaced; ``cd.sersic_modelimg`` is **re-rendered** (not
   re-fit) so it stays consistent with the new effective PSF; the next
   Sersic / PSF iteration picks up the updated state automatically.

The default library config has ``in_mainloop = false`` (kernel calibration
is opt-in). The test config has it on — recommended for any data where the
library PSF is known to be approximate.

References
----------

* :func:`sphot.calibrate_psf.calibrate_psf_step` — the recalibrator.
* :doc:`/api/calibrate_psf`
* :doc:`/rst/config` — every ``[psf-calib]`` knob.

------------------------------------------------------------------

.. _run-aperphot:

Aperture photometry (``run_aperphot``)
======================================

The optional fourth stage runs Petrosian aperture photometry on
``cd.psf_sub_data`` (the PSF-cleaned image) to produce final integrated
fluxes per filter.

* Aperture geometry is determined from a base-filter isophote fit
  (``[aperture].isophot_base_filter`` / ``fit_isophot_to``) and then applied
  unchanged across all filters.
* Sky is measured from ``cd.residual_masked`` (the post-PSF, post-Sersic
  image) — see ``[aperture].measure_sky_on``.
* Inner aperture mask radius is set by ``[aperture].center_mask``.

The output is a single CSV alongside the sphot HDF5 with one row per
filter and per source.

References
----------

* :func:`sphot.core.run_aperphot`
* :func:`sphot.aperture.aperture_routine`
* :doc:`/api/aperture`

------------------------------------------------------------------

.. sidebar:: Why the alternation works

   Each sub-step would be unbiased if the *other* two were perfect. Sersic
   needs PSF light gone; PSF photometry needs the galaxy core gone; sky
   needs both gone. Iterating works because each pass operates on the
   cleanest version of the *other* components currently available, and the
   non-modelled component each step injects (Sersic-fit residual into PSF
   data, PSF-fit residual into Sersic data) shrinks geometrically. Empirically,
   five to ten iterations with a strict convergence atol is enough on the
   targets sphot was built for.

Putting it all together
=======================

A typical end-to-end run on a multi-band cutout looks like:

.. code-block:: bash

   run_sphot mygalaxy.h5 --out_folder=results/

That single command performs:

1. Load + per-filter cutouts + auto-crop.
2. ``run_basefit`` on the base filter — the bulk of the runtime.
3. ``run_scalefit`` in parallel over every other filter — typically much
   faster because the Sersic shape is fixed.
4. ``run_aperphot`` (only if ``--photometry`` is passed) — Petrosian aperture
   photometry on the cleaned images.

To customize behaviour, drop a ``sphot_config.toml`` next to the data file
(see :doc:`/rst/config` for the full reference) — every threshold, ladder
step, fitter method, and convergence tolerance is exposed there.
