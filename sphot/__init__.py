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

