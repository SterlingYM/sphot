"""Parallel execution of run_scalefit across filters.

Architecture
------------
- One worker process per filter (multiprocessing.spawn).
- Each worker reads the galaxy from a parent-saved temp h5, calls
  `sphot.core.run_scalefit` on its assigned filter, and returns the
  modified CutoutData. The parent merges results back into the galaxy.
- Per-filter progress (Rich) is rendered in the parent via:
    * one PRE-ALLOCATED `Progress` row per filter (the "pre-spaced"
      layout the user asked for — rows do not jump as filters start),
    * dynamically-added child rows for nested tasks (NNLS refit,
      iPSF ladder, calibrate_psf_step, ...).
- Workers communicate progress events via a `multiprocessing.Manager`
  Queue. A listener thread in the parent translates events into real
  `Progress.update / add_task / remove_task` calls.
- Workers redirect stdout/stderr to per-filter log files so the parent's
  Live display stays clean.

Public API
----------
`parallel_scalefit(galaxy, base_params, filters, blur_psf, ...)`
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
import tempfile
import threading
import time
import warnings
from typing import Sequence

from rich.progress import (BarColumn, Progress, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.live import Live
from rich.console import Group
from rich.text import Text


# ----- pickleable Rich proxies (workers use these) --------------------

class QueueProgressProxy:
    """Pickleable proxy for rich.Progress.

    Mirrors the subset of Progress's API that sphot's core / psf code
    uses: add_task, update, remove_task.
    """

    def __init__(self, queue, filter_label):
        self.queue = queue
        self.filter_label = filter_label
        self._counter = 0

    def add_task(self, description, total=None, **kw):
        self._counter += 1
        child_id = self._counter
        try:
            self.queue.put(('add_task', self.filter_label, child_id,
                            description, total))
        except Exception:
            pass
        return child_id

    def update(self, task_id, advance=None, completed=None, total=None,
               refresh=None, description=None, **kw):
        try:
            self.queue.put(('update', self.filter_label, task_id,
                            advance, completed, total, description))
        except Exception:
            pass

    def remove_task(self, task_id):
        try:
            self.queue.put(('remove_task', self.filter_label, task_id))
        except Exception:
            pass


# ----- worker entry point --------------------------------------------

# Module-global queue handle. Set by _init_worker via Pool's
# initializer (which IS allowed to pass an mp.Queue at spawn time).
# We then read it from worker tasks instead of including it in
# pool.imap args (which goes through pickle and would raise
# "Queue objects should only be shared between processes through
# inheritance").
_WORKER_QUEUE = None


def _init_worker(queue):
    global _WORKER_QUEUE
    _WORKER_QUEUE = queue


def _scalefit_worker(args):
    """Spawn-mode worker. Reads galaxy from disk, runs scalefit on one
    filter, returns (filter, modified CutoutData) or (filter, None) on
    failure.
    """
    (galaxy_path, base_params, filtername, blur_psf_filt,
     allow_refit, fit_complex_model, n_mainloop_iter, log_file,
     working_dir) = args
    queue = _WORKER_QUEUE

    if log_file:
        try:
            f = open(log_file, 'w', buffering=1)
            sys.stdout = f
            sys.stderr = f
            # The parent set SPHOT_QUIET_IMPORT=1 around the Pool, so
            # bootstrap imports in this worker (sphot.config.load_config
            # etc.) installed a NullHandler and were silenced — that
            # prevents log lines from leaking to the parent's terminal and
            # clobbering its Live region before this redirect happens.
            # Now that stdout/stderr are pointed at our per-filter log
            # file, restore real logging into that file.
            import logging
            sphot_logger = logging.getLogger('sphot')
            for h in list(sphot_logger.handlers):
                sphot_logger.removeHandler(h)
            sphot_logger.setLevel(logging.INFO)
            sphot_logger.propagate = True
            sh = logging.StreamHandler(f)
            sh.setFormatter(logging.Formatter(
                "[sphot %(levelname)s] (%(asctime)s): %(message)s "
                "(%(module)s.%(funcName)s)",
                "%m/%d/%y %H:%M:%S"))
            sphot_logger.addHandler(sh)
        except Exception:
            pass
    if working_dir:
        try:
            os.chdir(working_dir)
        except Exception:
            pass
    warnings.filterwarnings('ignore')
    import matplotlib
    matplotlib.use('Agg')

    from sphot.data import read_sphot_h5
    from sphot.core import run_scalefit

    try:
        galaxy = read_sphot_h5(galaxy_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            queue.put(('filter_failed', filtername,
                       f'read_sphot_h5: {e}'))
        except Exception:
            pass
        return filtername, None

    proxy = QueueProgressProxy(queue, filtername)
    try:
        run_scalefit(galaxy, filtername, base_params,
                     allow_refit=allow_refit,
                     fit_complex_model=fit_complex_model,
                     N_mainloop_iter=n_mainloop_iter,
                     blur_psf=blur_psf_filt,
                     progress=proxy)
        try:
            queue.put(('filter_done', filtername))
        except Exception:
            pass
        return filtername, galaxy.images[filtername]
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            queue.put(('filter_failed', filtername, str(e)))
        except Exception:
            pass
        return filtername, None


# ----- public API ----------------------------------------------------

def parallel_scalefit(galaxy, base_params, filters: Sequence[str],
                      blur_psf,
                      *,
                      allow_refit: bool = False,
                      fit_complex_model: bool = False,
                      N_mainloop_iter: int = 5,
                      n_workers: int | None = None,
                      working_dir: str | None = None,
                      log_dir: str | None = None,
                      console=None):
    """Run run_scalefit on each filter in `filters` in parallel.

    Returns the list of (filter, CutoutData) tuples returned by workers.
    Modifies `galaxy.images[filter]` in-place with the worker results.

    `working_dir` should be the directory where sphot_config.toml lives
    (workers chdir to it before importing sphot.config). `log_dir` is
    where each filter's stdout/stderr is captured.
    """
    if n_workers is None:
        n_workers = min(len(filters), os.cpu_count() or 1, 6)

    if log_dir is None:
        log_dir = working_dir or os.getcwd()
    os.makedirs(log_dir, exist_ok=True)

    if working_dir is None:
        working_dir = os.getcwd()

    # Save the galaxy once for workers to read.
    tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False,
                                       dir=working_dir)
    tmp.close()
    galaxy.save(tmp.name)

    # Suppress sphot logger output during spawn-bootstrap imports in any
    # child process (Manager + Pool workers) so log lines emitted at
    # import time don't leak to the parent's terminal and clobber the
    # Live region. Spawn children inherit env vars; sphot/logging.py
    # checks this flag at module load time.
    _prev_quiet = os.environ.get('SPHOT_QUIET_IMPORT')
    os.environ['SPHOT_QUIET_IMPORT'] = '1'

    ctx = mp.get_context('spawn')
    # `ctx.Queue()` instead of `ctx.Manager().Queue()`: avoids spawning
    # a separate SyncManager subprocess (which has been flaky on macOS
    # spawn-mode — BrokenPipeError during the manager's address-send
    # handshake). A regular mp.Queue is a pipe + lock pair, fully
    # picklable across the Pool boundary.
    queue = ctx.Queue()

    # Share a single Console between the sphot RichHandler and the Live
    # display so log records emitted during the run render above the live
    # region instead of clobbering it. Without this, RichHandler writes to
    # its own Console while Live owns a different one — both target the
    # same stdout but neither knows the other is there, so a log line
    # printed mid-render breaks the bars (this was the "queued" rows being
    # truncated).
    if console is None:
        for h in logging.getLogger('sphot').handlers:
            h_console = getattr(h, 'console', None)
            if h_console is not None:
                console = h_console
                break

    # One renderable block per filter: headline + (child OR blank line).
    # The blank-line placeholder keeps each block at exactly 2 lines so that
    # children appearing/disappearing don't reflow the layout (which was
    # making a re-appearing child clobber the next filter's headline below).
    def _make_progress():
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
            console=console,
        )

    class _FilterBlock:
        def __init__(self, filter_name):
            self.filter_name = filter_name
            self.headline_progress = _make_progress()
            self.headline_id = self.headline_progress.add_task(
                f'[bold cyan]{filter_name}[/]: scalefit (queued)',
                total=None, start=False)
            self.child_progress = _make_progress()
            self.child_real_id = None  # real task id when a child is active

        def __rich_console__(self, console_, options):
            yield self.headline_progress
            if self.child_real_id is not None:
                yield self.child_progress
            else:
                yield Text("")  # placeholder so block height stays constant

    blocks = {f: _FilterBlock(f) for f in filters}
    started = set()

    live = Live(Group(*[blocks[f] for f in filters]),
                console=console, refresh_per_second=10, transient=False)

    # (filter, worker child_id) -> ('headline'|'child', real task id)
    task_map = {}

    stop_listener = threading.Event()

    def listener():
        while not stop_listener.is_set():
            try:
                msg = queue.get(timeout=0.2)
            except Exception:
                continue
            try:
                cmd = msg[0]
                if cmd == 'add_task':
                    _, filt, child_id, desc, total = msg
                    blk = blocks[filt]
                    if child_id == 1:
                        # First add_task per worker = the main loop. Promote
                        # it onto the pre-allocated headline so the headline
                        # gets a real bar/total and the layout stays stable.
                        blk.headline_progress.update(
                            blk.headline_id,
                            description=f'[bold cyan]{filt}[/]: {desc}',
                            total=total)
                        blk.headline_progress.start_task(blk.headline_id)
                        task_map[(filt, child_id)] = ('headline', blk.headline_id)
                        started.add(filt)
                    else:
                        # If a previous child is still in the slot (worker
                        # didn't remove_task before adding a new one), clear
                        # it first so the slot only ever holds one task.
                        if blk.child_real_id is not None:
                            try:
                                blk.child_progress.remove_task(blk.child_real_id)
                            except Exception:
                                pass
                            blk.child_real_id = None
                        real = blk.child_progress.add_task(f'    {desc}',
                                                            total=total)
                        blk.child_real_id = real
                        task_map[(filt, child_id)] = ('child', real)
                        # Force an immediate render so very short-lived child
                        # tasks (e.g. Sersic iNM, which can finish in <250ms)
                        # are visible at least once instead of being skipped
                        # over by the Live's 4Hz auto-refresh.
                        try:
                            live.refresh()
                        except Exception:
                            pass
                elif cmd == 'update':
                    _, filt, child_id, advance, completed, total, descr = msg
                    blk = blocks[filt]
                    target = task_map.get((filt, child_id))
                    if target is None:
                        continue
                    kind, real = target
                    p = (blk.headline_progress if kind == 'headline'
                         else blk.child_progress)
                    kw = {}
                    if advance is not None:
                        kw['advance'] = advance
                    if completed is not None:
                        kw['completed'] = completed
                    if total is not None:
                        kw['total'] = total
                    if descr is not None:
                        if kind == 'headline':
                            kw['description'] = f'[bold cyan]{filt}[/]: {descr}'
                        else:
                            kw['description'] = f'    {descr}'
                    if kw:
                        try:
                            p.update(real, **kw)
                        except Exception:
                            pass
                elif cmd == 'remove_task':
                    _, filt, child_id = msg
                    blk = blocks[filt]
                    target = task_map.pop((filt, child_id), None)
                    if target is None:
                        continue
                    kind, real = target
                    if kind == 'headline':
                        # Never remove the headline — it stays for the whole
                        # run and gets a final 'done'/'FAILED' state below.
                        continue
                    # Don't remove the bar from the slot immediately. iNM
                    # (and other short-lived child tasks) finish in <1s,
                    # which is below the user's perception threshold; if we
                    # remove instantly, the bar would never visibly settle
                    # at "complete". Instead mark the task 100% done and
                    # leave it in the slot. The next add_task overwrites
                    # this slot (existing logic) when the next nested step
                    # begins.
                    try:
                        task = next((t for t in blk.child_progress.tasks
                                     if t.id == real), None)
                        if task is not None and task.total is not None:
                            blk.child_progress.update(real,
                                                       completed=task.total)
                    except Exception:
                        pass
                elif cmd == 'filter_done':
                    _, filt = msg
                    started.add(filt)
                    blk = blocks[filt]
                    h = blk.headline_id
                    total = next((t.total for t in blk.headline_progress.tasks
                                  if t.id == h), None)
                    kw = {'description': f'[bold green]{filt}[/]: done'}
                    if total is not None:
                        kw['completed'] = total
                    blk.headline_progress.update(h, **kw)
                elif cmd == 'filter_failed':
                    _, filt, err = msg
                    started.add(filt)
                    short_err = str(err)[:60]
                    blk = blocks[filt]
                    blk.headline_progress.update(
                        blk.headline_id,
                        description=f'[bold red]{filt}[/]: FAILED: {short_err}')
            except Exception:
                pass

    listener_thread = threading.Thread(target=listener, daemon=True)
    listener_thread.start()

    args_list = [
        (tmp.name, base_params, f, blur_psf[f],
         allow_refit, fit_complex_model, N_mainloop_iter,
         os.path.join(log_dir, f'parallel_{f}.log'), working_dir)
        for f in filters
    ]

    results = []
    try:
        with live:
            # `initializer` + `initargs` is the standard way to share an
            # mp.Queue with workers in spawn mode (the queue is hooked
            # up at process bootstrap, not via pickle of imap args).
            with ctx.Pool(processes=n_workers, maxtasksperchild=1,
                          initializer=_init_worker,
                          initargs=(queue,)) as pool:
                for r in pool.imap_unordered(_scalefit_worker, args_list):
                    results.append(r)
            # let listener drain trailing messages
            time.sleep(0.5)
    finally:
        if _prev_quiet is None:
            os.environ.pop('SPHOT_QUIET_IMPORT', None)
        else:
            os.environ['SPHOT_QUIET_IMPORT'] = _prev_quiet
        stop_listener.set()
        listener_thread.join(timeout=2.0)
        try:
            os.remove(tmp.name)
        except Exception:
            pass

    # merge results back into the parent galaxy.
    # NOTE: `galaxy.images` is a @property that builds a fresh dict each
    # access, so `galaxy.images[f] = cd` mutates a throwaway view and
    # the real CutoutData (stored as galaxy.<filtername>) is unchanged.
    # Use setattr so the modification persists.
    for f, cd in results:
        if cd is not None:
            setattr(galaxy, f, cd)
    return results
