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

For installation instructions, usage overview, project information, and API reference, please see our [documentation](https://deimos.readthedocs.io).

Citing DEIMoS
-------------
If you would like to reference deimos in an academic paper, we ask you include the following.

* DEIMoS, version 1.3.2 http://github.com/pnnl/deimos (accessed MMM YYYY)
* Colby, S.M., Chang, C.H., Bade, J.L., Nunez, J.R., Blumer, M.R., Orton, D.J., Bloodsworth, K.J., Nakayasu, E.S., Smith, R.D, Ibrahim, Y.M. and Renslow, R.S., 2022. DEIMoS: an open-source tool for processing high-dimensional mass spectrometry data. *Analytical Chemistry*, 94(16), pp.6130-6138.

Disclaimer
----------
This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830
