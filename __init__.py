from .main import continuum_image, line_image
from .helper_functions import goto_folder
from .radmc_visualization import create_sed, plot_sed
from .sink_config import radmc_datadir, sink_dirs

print(f"RADMC data folder specified as '{radmc_datadir}', and RAMSES data paths have been supplied to the following sinks: {list(sink_dirs.keys())}")