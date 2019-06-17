from pyteomics import mzml
import gzip
from os.path import *


def spectra(path):
    # check for zip
    if splitext(path)[-1].lower() == 'mzml':
        f = path
        close = False
    else:
        f = gzip.open(path, 'rb')
        close = True

    # process mzml
    spectra = []
    for obj in mzml.read(f):
        spectra.append(obj)

    # close gzip file
    if close:
        f.close()

    return spectra
