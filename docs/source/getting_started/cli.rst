======================
Command Line Interface
======================

The CLI is able to process data from mzML through MS1 and MS2 peakpicking. 
For usage overview, simply invoke ``deimos --help`` or ``-h``. 
A Snakemake configuration file in `YAML <http://yaml.org/>`_ format is required. 
DEIMoS will try to find ``config.yaml`` in the current directory, else a configuration file must be specified through the ``--config`` flag. 
A default workflow configuration (``workflows/default_config.yaml``) is provided, but this is intended to be modified and supplied by the user to accomodate workflow-specific needs.


.. code-block:: console
  
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

Inputs (`.mzML <http://www.psidev.info/mzML>`_ or .mzML.gz) will automatically be detected in the ``input/`` folder, and results will be populated in a corresponding ``output/`` folder, both relative to the current working directory.
For example:


.. code-block:: console
  
  .
  ├── config.yaml
  ├── input
  │  ├── example1.mzML.gz
  │  ├── example2.mzML.gz
  │  └── ...
  └── output

For running in cluster environments, please consult the `Snakemake <https://snakemake.readthedocs.io>`_ workflow management system documentation concerning profiles.