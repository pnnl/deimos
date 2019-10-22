import spextractor as spx
from os.path import *
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


plt.switch_backend('agg')


def main(exp_path, output_path, beta, tfix, ms1_threshold, ms2_threshold):
    # output directory
    if not exists(output_path):
        os.makedirs(output_path)

    # load
    data = spx.utils.load_hdf(exp_path)

    # select ms1
    ms1 = data.loc[data['ms_level'] == 1, :].drop('ms_level', axis=1)

    # collapse
    ms1 = spx.utils.collapse(ms1, keep=['mz', 'drift_time'], how=np.sum)

    # find features
    ms1 = spx.peakpick.auto(ms1, features=['mz', 'drift_time'],
                            res=[0.01, 0.12], sigma=[0.03, 0.3], truncate=4, threshold=ms1_threshold)

    # sort by intensity
    ms1 = ms1.sort_values(by='intensity', ascending=False)

    # ccs
    if (beta is not None) and (tfix is not None):
        # calibrate
        c = spx.calibrate.ArrivalTimeCalibration()
        c.calibrate(tfix=tfix, beta=beta)

        # apply
        ms1['ccs'] = ms1.apply(lambda x: c.arrival2ccs(x['drift_time'], x['mz']), axis=1)

    # select ms2
    ms2 = data.loc[data['ms_level'] == 2, :].drop('ms_level', axis=1)

    # check if ms2 present
    if len(ms2.index) > 1:
        # collapse
        ms2 = spx.utils.collapse(ms2, keep=['mz', 'drift_time'], how=np.sum)

        # peakpick
        ms2_peaks = spx.peakpick.auto(ms2, features=['mz', 'drift_time'],
                                      res=[0.01, 0.12], sigma=[0.03, 0.3], truncate=4, threshold=ms2_threshold)

    # no ms2 present
    else:
        ms2 = None
        ms2_peaks = None

    # if ms2 present
    if ms2 is not None:
        # ms2 containers
        ms2_list = []
        ms2_list_centroid = []

        # iterate ms1 peaks
        for idx, peak in ms1.iterrows():
            # extract ms2
            ms2_subset = spx.targeted.find_feature(ms2,
                                                   by='drift_time',
                                                   loc=peak['drift_time'],
                                                   tol=4 * 0.3)
            # ms2 features
            if ms2_subset is not None:
                # collapse to mz
                ms2_mz = spx.utils.collapse(ms2_subset, keep='mz', how=np.sum)

                # filter
                ms2_out = ms2_mz.loc[(ms2_mz['intensity'] > ms2_threshold) & (ms2_mz['mz'] <= peak['mz'] + 10), :].sort_values(by='mz')

                # string
                ms2_out = ';'.join(['%.4f %i' % (mz, i) for mz, i in zip(ms2_out['mz'].values, ms2_out['intensity'].values)])

                # append
                ms2_list.append(ms2_out)

            # nothing found
            else:
                ms2_list.append(np.nan)

            # extract centroid ms2
            ms2_peaks_subset = spx.targeted.find_feature(ms2_peaks,
                                                         by='drift_time',
                                                         loc=peak['drift_time'],
                                                         tol=4 * 0.3)

            # ms2 centroid features
            if ms2_peaks_subset is not None:
                # collapse to mz
                ms2_peaks_mz = spx.utils.collapse(ms2_peaks_subset, keep='mz', how=np.sum)

                # filter
                ms2_peaks_out = ms2_peaks_mz.loc[ms2_peaks_mz['mz'] <= peak['mz'] + 10, :].sort_values(by='mz')

                # string
                ms2_peaks_out = ';'.join(['%.4f %i' % (mz, i) for mz, i in zip(ms2_peaks_out['mz'].values, ms2_peaks_out['intensity'].values)])

                # append
                ms2_list_centroid.append(ms2_peaks_out)

            # nothing found
            else:
                ms2_list_centroid.append(np.nan)

    else:
        ms2_list = np.nan
        ms2_list_centroid = np.nan

    # save
    ms1['ms2'] = ms2_list
    ms1.to_csv(join(output_path, '%s.tsv' % splitext(basename(exp_path))[0]), sep='\t', index=False)

    ms1['ms2'] = ms2_list_centroid
    ms1.to_csv(join(output_path, '%s_centroid.tsv' % splitext(basename(exp_path))[0]), sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpExtractor: target MS2 extraction script.')
    parser.add_argument('-v', '--version', action='version', version=spx.__version__, help='print version and exit')
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
