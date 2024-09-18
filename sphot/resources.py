import importlib.resources as pkg_resources
from . import samples

# Assuming your data file is within a package named 'mypackage' and is called 'data.txt'
sample_files =  []
with pkg_resources.as_file(
    pkg_resources.files(samples) / 'g281.h5',
    ) as data_file_path:
    sample_files.append(data_file_path)
    
psf_data = None
with pkg_resources.as_file(
    pkg_resources.files(samples) / 'psf_data.h5',
    ) as data_file_path:
    psf_data = data_file_path