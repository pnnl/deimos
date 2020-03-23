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

Command-line DEIMoS
-------------------
For usage overview, simply invoke ``deimos --help`` or ``-h``. A Snakemake configuration file in [YAML](http://yaml.org/) format is required. DEIMoS will try to find ``config.yaml`` in the current directory, else a configuration file must be specified through the ``--config`` flag. A default [workflow configuration](resources/example_config.yaml) is provided, but this is intended to be modified and supplied by the user to accomodate workflow-specific needs.

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

DEIMoS API
----------
For more flexibility, the same functionality can be achieved through DEIMoS' API. Recreating the command-line workflow through internal standards detection would involve the following:

```python
import deimos
import numpy as np

# load data
data = deimos.utils.read_mzml('path/to/dataset.mzML.gz')

# peakpicking in ms1
peaks = deimos.peakpick.auto(data)
deimos.utils.save_hdf('path/to/peaks.h5', {'ms1': peaks})

# find standards
masses = np.loadtxt('path/to/masses.txt')
stds = deimos.alignment.internal_standards(peaks, masses=masses, tol=0.02)
deimos.utils.save_hdf('path/to/standards.h5', {'ms1': stds})
```

Citing DEIMoS
-------------
If you would like to reference DEIMoS in an academic paper, we ask you include the following:
* DEIMoS, version 0.1.0 http://github.com/pnnl/deimos (accessed MMM YYYY)

Disclaimer
----------
This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830
