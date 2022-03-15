DEIMoS
=======
DEIMoS, or Data Extraction for Integrated Multidimensional Spectrometry, is a Python application 
programming interface and command-line tool for high-dimensional mass spectrometry (MS) data 
analysis workflows that offers ease of development and access to efficient algorithmic implementations. 
Functionality includes feature detection, feature alignment, collision cross section (CCS) calibration, 
isotope detection, and MS/MS spectral deconvolution, with the output comprising detected features aligned 
across study samples and characterized by mass, CCS, tandem mass spectra, and isotopic signature. 
Notably, DEIMoS operates on N-dimensional data, largely agnostic to acquisition instrumentation; 
algorithm implementations simultaneously utilize all dimensions to (i) offer greater separation between features, 
thus improving detection sensitivity, (ii) increase alignment/feature matching confidence among datasets, 
and (iii) mitigate convolution artifacts in tandem mass spectra. 

Installation
------------
Use [``conda``](https://www.anaconda.com/download/) to create a virtual environment with required dependencies. First, ensure ``conda`` and ``anaconda`` are updated:
```
$ conda update conda
$ conda update anaconda
```

Create the virtual environment:
```
$ conda create -n deimos -c conda-forge -c bioconda python=3.7 numpy scipy pandas matplotlib snakemake pymzml h5py statsmodels scikit-learn dask pytables
```

Activate the virtual environment:
```
$ conda activate deimos
```

Install DEIMoS using [``pip``](https://pypi.org/project/pip/):
```
# clone/install
$ git clone https://github.com/pnnl/deimos.git
$ pip install deimos/

# direct
$ pip install git+https://github.com/pnnl/deimos
```

Command-line interface
----------------------
The CLI is able to process data from mzML through MS1 and MS2 peakpicking. For usage overview, simply invoke ``deimos --help`` or ``-h``. A Snakemake configuration file in [YAML](http://yaml.org/) format is required. DEIMoS will try to find ``config.yaml`` in the current directory, else a configuration file must be specified through the ``--config`` flag. A default [workflow configuration](resources/example_config.yaml) is provided, but this is intended to be modified and supplied by the user to accomodate workflow-specific needs.

```
$ deimos --help
usage: deimos [-h] [-v] [--config PATH] [--dryrun] [--unlock] [--touch]
              [--latency N] [--cores N] [--count N] [--start IDX]
              [--cluster PATH] [--jobs N]

DEIMoS: Data Extraction for Integrated Multidimensional Spectrometry

optional arguments:
  -h, --help      show this help message and exit
  -v, --version   print version and exit
  --config PATH   path to yaml configuration file
  --dryrun        perform a dry run
  --unlock        unlock directory
  --touch         touch output files only
  --latency N     specify filesystem latency (seconds)
  --cores N       number of cores used for execution (local execution only)
  --count N       number of files to process (limits DAG size)
  --start IDX     starting file index (for use with --count)

cluster arguments:
  --cluster PATH  path to cluster execution yaml configuration file
  --jobs N        number of simultaneous jobs to submit to a slurm queue
```

Inputs ([.mzML](http://www.psidev.info/mzML) or .mzML.gz) will automatically be detected in the ``input/`` folder, and results will be populated in a corresponding ``output/`` folder, both relative to the current working directory. For example:

```
.
├── config.yaml
├── input
│   ├── example1.mzML.gz
│   ├── example2.mzML.gz
│   └── ...
└── output
```

For running in cluster environments, please consult the [Snakemake](https://snakemake.readthedocs.io) workflow management system documentation concerning [profiles](https://snakemake.readthedocs.io/en/stable/executing/cli.html#profiles).

Application programming interface
---------------------------------
For more flexibility, the same functionality can be achieved through DEIMoS's API. Please reference the documentation on [Read the Docs](deimos.readthedocs.io) or the tutorial, which gives an overview of most functionality, provided as a [jupyter notebook](examples/tutorial.ipynb).

Citing DEIMoS
-------------
If you would like to reference deimos in an academic paper,we ask you include the following.
The arXiv link will be updated pending completion of the journal review process.
* DEIMoS, version 0.1.0 http://github.com/pnnl/deimos (accessed MMM YYYY)
* Colby, S.M., Chang, C.H., Bade, J.L., Nunez, J.R., Blumer, M.R., Orton, D.J., Bloodsworth, K.J., Nakayasu, E.S., Smith, R.D, Ibrahim, Y.M. and Renslow, R.S., 2021. DEIMoS: an open-source tool for processing high-dimensional mass spectrometry data. arXiv preprint arXiv:2112.03466.

Disclaimer
----------
This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830
