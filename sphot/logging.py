# sphot/logger_config.py
import logging
import os
from rich.logging import RichHandler

# Create a logger
logger = logging.getLogger('sphot')
if logger.hasHandlers():
    logger.handlers.clear()

if os.environ.get('SPHOT_QUIET_IMPORT'):
    # Set by parallel.parallel_scalefit around its multiprocessing Pool so
    # spawn workers inherit it. Bootstrap imports in those workers (e.g.
    # sphot.config.load_config logging "User config file loaded ...") would
    # otherwise leak to the parent's terminal and clobber the Live region
    # before the worker can redirect its stdout. The worker's
    # _scalefit_worker reinstalls a real handler against its log file
    # before doing any user-visible work.
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL + 1)
    logger.propagate = False
elif not logger.handlers:
    custom_format = "[sphot %(levelname)s] (%(asctime)s): %(message)s (%(module)s.%(funcName)s)"
    # custom_format = "[sphot] %(message)s (%(module)s.%(funcName)s)"

    date_format = "%m/%d/%y %H:%M:%S"
    formatter = logging.Formatter(custom_format, date_format)

    # handler = logging.StreamHandler()
    handler = RichHandler(rich_tracebacks=False,show_time=False,show_level=False,show_path=False)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)