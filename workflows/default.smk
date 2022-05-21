import deimos
import glob
import h5py
import numpy as np
from os.path import *
import pandas as pd


# infer wildcards from inputs
fns = [basename(x) for x in glob.glob(join('input', '*.*'))]
IDS = [splitext(splitext(x)[0])[0] for x in fns]
lookup = {k: v for k, v in zip(IDS, fns)}


rule all:
    input:
        expand(join('output', 'peakpick', '{id}.h5'), id=IDS),


rule mzml2hdf:
    input:
        lambda wildcards: join('input', lookup[wildcards.id])
    output:
        join('output', 'parsed', '{id}.h5')
    run:
        # read/parse mzml
        data = deimos.load(input[0], accession=config['accession'])

        # save as hdf5
        for k, v in data.items():
            deimos.save(output[0], v, key=k, mode='a')


rule peakpick:
    input:
        rules.mzml2hdf.output
    output:
        join('output', 'peakpick', '{id}.h5')
    run:
        # get keys
        keys = list(h5py.File(input[0], 'r').keys())

        for k in keys:
            # load data
            data = deimos.load(input[0], key=k, config['dims'] + ['intensity'])

            # partition in m/z
            partitions = deimos.partition(data,
                                          split_on=config['partition']['split_on'],
                                          size=config['partition']['size'],
                                          overlap=config['partition']['overlap'])

            # perform peakpicking per partition
            peaks = partitions.map(deimos.peakpick.local_maxima,
                                   dims=config['dims'],
                                   processes=config['partition']['processes'],
                                   **config['peakpick'][k])

            # save
            deimos.save(output[0], peaks, key=k, mode='a')
