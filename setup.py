
from setuptools import setup, find_packages
from spextractor import __version__

pkgs = find_packages(exclude=('examples', 'docs', 'resources'))

setup(
    name='spextractor',
    version=__version__,
    description='spectrum extractor',
    author='Sean M. Colby',
    author_email='sean.colby@pnnl.gov',
    url='https://github.com/pnnl/spextractor',
    packages=pkgs,
    entry_points={
        'console_scripts': ['spx = spextractor.cli:main']
    },
    package_data={'': ['Snakefile']},
    include_package_data=True
)
