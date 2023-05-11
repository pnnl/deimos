import os
from setuptools import find_packages, setup
import sys

sys.path.append(os.path.dirname(__file__))

import deimos


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    requirements = f.read()

pkgs = find_packages(exclude=('examples', 'docs', 'tests'))

setup(
    name='deimos',
    version=deimos.__version__,
    description='Data Extraction for Integrated Multidimensional Spectrometry',
    long_description=readme,
    author='Sean M. Colby',
    author_email='sean.colby@pnnl.gov',
    url='https://github.com/pnnl/deimos',
    install_requires=requirements,
    python_requires='==3.8.*',
    license=license,
    packages=pkgs,
    entry_points={
        'console_scripts': ['deimos = deimos.cli:main']
    },
    package_data={'': ['*.smk', '*.yml']},
)
