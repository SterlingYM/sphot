import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
import logging

# Configure logging for sphot
logger = logging.getLogger('sphot')
logger.setLevel(logging.WARNING)
sphot_formatter = logging.Formatter('[sphot] %(levelname)s: %(message)s (%(funcName)s)')
sphot_handler = logging.StreamHandler()
sphot_handler.setFormatter(sphot_formatter)
logger.addHandler(sphot_handler)

from .data import read
from .core import *
from .plotting import plot_sphot_results as plot_results
from .plotting import astroplot

