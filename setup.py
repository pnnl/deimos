
from setuptools import setup, find_packages
from deimos import __version__

pkgs = find_packages(exclude=('examples', 'docs', 'resources'))

setup(
    name='deimos',
    version=__version__,
    description='Data Extraction for Integrated Multidimensional Spectrometry',
    author='Sean M. Colby',
    author_email='sean.colby@pnnl.gov',
    url='https://github.com/pnnl/deimos',
    packages=pkgs,
    entry_points={
        'console_scripts': ['spx = deimos.cli:main']
    },
    package_data={'': ['Snakefile']},
    include_package_data=True
)
