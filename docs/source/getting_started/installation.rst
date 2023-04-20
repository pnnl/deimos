============
Installation
============

Clone the repository:

.. code-block:: console

  $ git clone https://github.com/pnnl/deimos.git

Use `conda <https://www.anaconda.com/download/>`_ to create a virtual environment with required dependencies.
First, ensure ``conda`` and ``anaconda`` are updated:

.. code-block:: console
  
  $ conda update conda
  $ conda update anaconda

Create the virtual environment:

.. code-block:: console
  
  $ cd deimos/
  $ conda env create -f environment.yml

Activate the virtual environment:

.. code-block:: console
  
  $ conda activate deimos

Install DEIMoS using `pip <https://pypi.org/project/pip/>`_:

.. code-block:: console
  
  $ pip install -e .
