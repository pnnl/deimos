import deimos
import glob
import h5py
import numpy as np
from os.path import *
import pandas as pd


# Infer wildcards from inputs
fns = [basename(x) for x in glob.glob(join('input', '*.*'))]
IDS = [splitext(splitext(x)[0])[0] for x in fns]
lookup = {k: v for k, v in zip(IDS, fns)}


# Collect all outputs
rule all:
    input:
        expand(join('output', 'peakpicked', '{id}.h5'), id=IDS),


# Convert mzML to HDF5
rule mzml2hdf:
    input:
        lambda wildcards: join('input', lookup[wildcards.id])
    output:
        join('output', 'parsed', '{id}.h5')
    run:
        # Read/parse mzml
        data = deimos.load(input[0], accession=config['accession'])

        # Enumerate MS levels
        for k, v in data.items():
            # Save as hdf5
            deimos.save(output[0], v, key=k, mode='a')


# Build factors
rule factorize:
    input:
        rules.mzml2hdf.output
    output:
        join('output', 'factors', '{id}.npy')
    run:
        # Get keys
        keys = list(h5py.File(input[0], 'r').keys())

        factors = {}
        # Enumerate MS levels
        for k in keys:
            # Load data
            data = deimos.load(input[0], key=k, columns=config['dims'] + ['intensity'])

            # Build factors
            factors[k] = deimos.build_factors(data, dims=config['dims'])

        # Save factors
        np.save(output[0], factors)


# Threshold data by intensity
rule threshold:
    input:
        rules.mzml2hdf.output
    output:
        join('output', 'thresholded', '{id}.h5')
    run:
        # Get keys
        keys = list(h5py.File(input[0], 'r').keys())

        # Enumerate MS levels
        for k in keys:
            # Load data
            data = deimos.load(input[0], key=k, columns=config['dims'] + ['intensity'])

            # Threshold
            data = deimos.threshold(data, threshold=config['threshold'])

            # Save
            deimos.save(output[0], data, key=k, mode='a')   


# Smooth data
rule smooth:
    input:
        rules.factorize.output,
        rules.threshold.output
    output:
        join('output', 'smoothed', '{id}.h5')
    run:
        # Load factors
        factors = np.load(input[0], allow_pickle=True).item()

        # Get keys
        keys = list(h5py.File(input[1], 'r').keys())

        # Enumerate MS levels
        for k in keys:
            # Load data
            data = deimos.load(input[1], key=k, columns=config['dims'] + ['intensity'])

            # Perform smoothing
            data = deimos.filters.smooth(data,
                                         factors=factors[k],
                                         dims=config['dims'],
                                         iterations=config['smooth']['iters'],
                                         radius=config['smooth']['radius'])

            # Save
            deimos.save(output[0], data, key=k, mode='a')


# Perform peak detection
rule peakpick:
    input:
        rules.factorize.output,
        rules.smooth.output
    output:
        join('output', 'peakpicked', '{id}.h5')
    run:
        # Load factors
        factors = np.load(input[0], allow_pickle=True).item()

        # Get keys
        keys = list(h5py.File(input[1], 'r').keys())

        # Enumerate MS levels
        for k in keys:
            # Load data
            data = deimos.load(input[1], key=k, columns=config['dims'] + ['intensity'])

            # Perform peakpicking
            peaks = deimos.peakpick.persistent_homology(data,
                                                        factors=factors[k],
                                                        dims=config['dims'],
                                                        radius=config['weighted_mean']['radius'])

            # Save
            deimos.save(output[0], peaks, key=k, mode='a')
