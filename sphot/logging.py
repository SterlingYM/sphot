# sphot/logger_config.py
import logging
from rich.logging import RichHandler

# Create a logger
logger = logging.getLogger('sphot')
if logger.hasHandlers():
    logger.handlers.clear()
    
if not logger.handlers:
    custom_format = "[sphot %(levelname)s] (%(asctime)s): %(message)s (%(module)s.%(funcName)s)"
    # custom_format = "[sphot] %(message)s (%(module)s.%(funcName)s)"

    date_format = "%m/%d/%y %H:%M:%S"
    formatter = logging.Formatter(custom_format, date_format)

    # handler = logging.StreamHandler()
    handler = RichHandler(rich_tracebacks=False,show_time=False,show_level=False,show_path=False)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)