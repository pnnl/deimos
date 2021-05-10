from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    requirements = f.read()

pkgs = find_packages(exclude=('examples', 'docs', 'resources'))

setup(
    name='deimos',
    version='0.1.0',
    description='Data Extraction for Integrated Multidimensional Spectrometry',
    long_description=readme,
    author='Sean M. Colby',
    author_email='sean.colby@pnnl.gov',
    url='https://github.com/pnnl/deimos',
    install_requires=requirements,
    python_requires='>=3.7',
    license=license,
    packages=pkgs,
    entry_points={
        'console_scripts': ['deimos = deimos.cli:main']
    },
    package_data={'': ['Snakefile']},
    include_package_data=True
)
