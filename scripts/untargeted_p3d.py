import deimos
from os.path import *
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


# plt.switch_backend('agg')


def subset(data, ms_level):
    res = data.loc[data['ms_level'] == ms_level, :].drop('ms_level', axis=1)

    if len(res.index) > 1:
        return res
    return None


def initial_peakpick(ms, threshold=1E3):
    if len(ms.index) > 1:
        # collapse
        ms = deimos.utils.collapse(ms, keep=['mz', 'drift_time'], how=np.sum)

        # find features
        ms = deimos.peakpick.auto(ms, features=['mz', 'drift_time'],
                                  res=[0.01, 0.12], sigma=[0.03, 0.3], truncate=4, threshold=threshold)

        # sort by intensity
        ms = ms.sort_values(by='intensity', ascending=False)

        return ms

    return None


def pseudo3Dpeakpick(peaks, data, threshold=1E3):
    # second pick pick
    newpeaks = pd.DataFrame()
    for idx, peak in peaks.iterrows():
        # 3d ms1
        subset = deimos.targeted.find_feature(data,
                                              by=['mz', 'drift_time'],
                                              loc=[peak['mz'], peak['drift_time']],
                                              tol=[4 * 0.03, 4 * 0.3])
        # subset = deimos.utils.collapse(subset, keep='retention_time', how=np.sum)
        newpeak = deimos.peakpick.auto(subset, features=['mz', 'drift_time', 'retention_time'],
                                       res=[0.01, 0.12, 0.05], sigma=[0.03, 0.3, 0.04], truncate=4, threshold=threshold)

        if len(newpeak.index) > 1:
            newpeaks = newpeaks.append(newpeak)
            print(newpeak)
            print()

    # combine
    deimos.utils.collapse(newpeaks, keep=['mz', 'drift_time', 'retention_time'], how=np.sum)

    return newpeaks


def compute_ccs(ms, beta, tfix):
    # calibrate
    c = deimos.calibrate.ArrivalTimeCalibration()
    c.calibrate(tfix=tfix, beta=beta)

    # apply
    ms['ccs'] = ms.apply(lambda x: c.arrival2ccs(x['drift_time'], x['mz']), axis=1)

    return ms


def main(exp_path, output_path, beta, tfix, ms1_threshold, ms2_threshold):
    # output directory
    if not exists(output_path):
        os.makedirs(output_path)

    # load
    data = deimos.utils.load_hdf(exp_path)

    # initial ms1 peakpick
    ms1 = subset(data, ms_level=1)
    ms1_peaks = initial_peakpick(ms1, threshold=ms1_threshold)
    print('intial peakpick:', len(ms1_peaks.index))

    # second round
    ms1_peaks = pseudo3Dpeakpick(ms1_peaks, ms1, threshold=ms1_threshold)
    print('p3d peakpick:', len(ms1_peaks.index))

    # add ccs column
    if (beta is not None) and (tfix is not None):
        ms1_peaks = compute_ccs(ms1_peaks, beta, tfix)

    # # initial ms2 peakpick
    # ms2 = subset(data, ms_level=2)
    # ms2_peaks = initial_peakpick(data, threshold=ms2_threshold)

    # save
    ms1_peaks.to_csv(join(output_path, '%s.tsv' % splitext(basename(exp_path))[0]), sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEIMoS: untargeted pseudo 3D MS2 extraction script.')
    parser.add_argument('-v', '--version', action='version', version=deimos.__version__, help='print version and exit')
    parser.add_argument('data', type=str, help='path to input .h5 file containing spectral data (str)')
    parser.add_argument('output', type=str, help='path to output folder (str)')
    parser.add_argument('--beta', type=float,
                        help='ccs calibration parameter beta (float)')
    parser.add_argument('--tfix', type=float,
                        help='ccs calibration parameter tfix (float)')
    parser.add_argument('--ms1-thresh', type=float, default=1E3,
                        help='intensity threshold (float, default=1E3)')
    parser.add_argument('--ms2-thresh', type=float, default=1E3,
                        help='intensity threshold (float, default=1E3)')

    # parse arguments
    args = parser.parse_args()

    # run
    main(args.data, args.output, args.beta, args.tfix, args.ms1_thresh, args.ms2_thresh)
