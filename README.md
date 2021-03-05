DEIMoS
=======
DEIMoS, or Data Extraction for Integrated Multidimensional Spectrometry, is a Python application programming interface and command-line tool for mass spectrometry data analysis workflows, offering ease of development and access to efficient algorithmic implementations. Critically, DEIMoS operates on data of arbitrary dimension, regardless of acquisition instrumentation (e.g. feature finding in 3D using retention time, drift time / CCS, and m/z).

Installation
------------
Use [``conda``](https://www.anaconda.com/download/) to create a virtual environment with required dependencies. First, ensure ``conda`` and ``anaconda`` are updated:
```bash
conda update conda
conda update anaconda
```

Create the virtual environment:
```bash
conda create -n deimos -c conda-forge -c bioconda python=3.7 numpy scipy pandas matplotlib snakemake pymzml h5py statsmodels scikit-learn
```

Activate the virtual environment:
```
conda activate deimos
```

Install DEIMoS using [``pip``](https://pypi.org/project/pip/):
```bash
# clone/install
git clone https://github.com/pnnl/deimos.git
pip install deimos/

# direct
pip install git+https://github.com/pnnl/deimos
```

Command-line interface
----------------------
The CLI is able to process data from mzML through MS1 and MS2 peakpicking. For usage overview, simply invoke ``deimos --help`` or ``-h``. A Snakemake configuration file in [YAML](http://yaml.org/) format is required. DEIMoS will try to find ``config.yaml`` in the current directory, else a configuration file must be specified through the ``--config`` flag. A default [workflow configuration](resources/example_config.yaml) is provided, but this is intended to be modified and supplied by the user to accomodate workflow-specific needs.

```bash
$ deimos --h
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

```bash
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
For more flexibility, the same functionality can be achieved through DEIMoS's API. Please see the tutorial, which gives an overview of most functionality, provided as a [jupyter notebook](examples/tutorial.ipynb).

Citing DEIMoS
-------------
If you would like to reference DEIMoS in an academic paper, we ask you include the following:
* DEIMoS, version 0.1.0 http://github.com/pnnl/deimos (accessed MMM YYYY)

Disclaimer
----------
This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830
