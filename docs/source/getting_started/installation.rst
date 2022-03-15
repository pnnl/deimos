============
Installation
============

Use `conda <https://www.anaconda.com/download/>`_ to create a virtual environment with required dependencies.
First, ensure ``conda`` and ``anaconda`` are updated:

.. code-block:: console
  
  $ conda update conda
  $ conda update anaconda


Create the virtual environment:

.. code-block:: console
  
  $ conda create -n deimos -c conda-forge -c bioconda python=3.7 numpy scipy pandas matplotlib snakemake pymzml h5py statsmodels scikit-learn dask pytables

Activate the virtual environment:

.. code-block:: console
  
  $ conda activate deimos

Install DEIMoS using `pip <https://pypi.org/project/pip/>`_:

.. code-block:: console
  
  # clone/install
  $ git clone https://github.com/pnnl/deimos.git
  $ pip install deimos/

  # direct
  $ pip install git+https://github.com/pnnl/deimos
