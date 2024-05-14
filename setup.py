# setup.py

from setuptools import setup, find_packages

setup(
    name='sphot',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26',
        'scipy>=1.11',
        'matplotlib>=3.8',
        'pandas>=2.1',
        'astropy>=6.0',
        'photutils>=1.12',
        'petrofit>=0.5',  
        'h5py>=3.10',
        'tqdm>=4.66',
    ],
)
