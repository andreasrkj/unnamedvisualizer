# Grab the packages we need
import sys, os

from . import data_generation as data
from . import plotting as plot
from .main import *
from .sink_config import radmc_datadir, sink_dirs

print(f"RADMC data folder specified as '{radmc_datadir}', and RAMSES data paths have been supplied to the following sinks: {list(sink_dirs.keys())}")