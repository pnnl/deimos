from setuptools import setup, find_packages
from deimos import __version__

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

pkgs = find_packages(exclude=('examples', 'docs', 'resources'))

setup(
    name='deimos',
    version=__version__,
    description='Data Extraction for Integrated Multidimensional Spectrometry',
    long_description=readme,
    author='Sean M. Colby',
    author_email='sean.colby@pnnl.gov',
    url='https://github.com/pnnl/deimos',
    license=license,
    packages=pkgs,
    entry_points={
        'console_scripts': ['deimos = deimos.cli:main']
    },
    package_data={'': ['Snakefile']},
    include_package_data=True
)
