import spextractor as spx
from os.path import *
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


plt.switch_backend('agg')


def main(exp_path, output_path, beta, tfix, threshold):
    # output directory
    if not exists(output_path):
        os.makedirs(output_path)

    # calibrate
    c = spx.calibrate.ArrivalTimeCalibration()
    c.calibrate(tfix=tfix, beta=beta)

    # split by ms level
    data = {}
    data['ms1'] = df.loc[df['ms_level'] == 1, :].drop('ms_level', axis=1).reset_index(drop=True)
    data['ms2'] = df.loc[df['ms_level'] == 2, :].drop('ms_level', axis=1).reset_index(drop=True)

    # find features
    ms1_peaks = spx.peakpick.auto(data['ms1'], features=['mz', 'drift_time', 'retention_time'],
                                  res=[0.01, 0.12, 0.05], sigma=[0.03, 0.3, 0.04], truncate=4, threshold=1000)

    # container
    d = {'mz': [], 'rt': [], 'dt': [], 'ccs': [], 'intensity': [], 'ms2': []}

    # iterate through peaks
    if ms1_peaks is not None:
        for idx, peak in ms1_peaks.iterrows():
            mz_exp = peak['mz']
            rt_exp = peak['retention_time']
            dt_exp = peak['drift_time']
            int_exp = peak['intensity']
            ccs_exp = c.arrival2ccs(mz_exp, dt_exp)

            # extract ms2
            ms2 = spx.targeted.find_feature(data['ms2'], by=['drift_time', 'retention_time'],
                                            loc=[dt_exp, rt_exp],
                                            tol=[4 * x for x in [0.03, 0.3, 0.04]])

            if ms2 is not None:
                ms2_mz = spx.utils.collapse(ms2, by='mz', how=np.sum)

                # sort
                ms2_out = ms2_mz.loc[(ms2_mz['intensity'] > threshold) & (ms2_mz['mz'] <= mz_i + 10), :].sort_values(by='mz')
                ms2_out = ''.join(['%.4f %i;' % (mz, i) for mz, i in zip(ms2_out['mz'].values, ms2_out['intensity'].values)])

                # append
                d['mz'].append(mz_exp)
                d['rt'].append(rt_exp)
                d['dt'].append(dt_exp)
                d['ccs'].append(ccs_exp)
                d['intensity'].append(int_exp)
                d['ms2'].append(ms2_out)

    df = pd.DataFrame(d)
    df = df.sort_values(by='intensity', ascending=False)
    df.to_csv(join(output_path, '%s.tsv' % splitext(basename(exp_path))[0]), sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpExtractor: target MS2 extraction script.')
    parser.add_argument('-v', '--version', action='version', version=spx.__version__, help='print version and exit')
    parser.add_argument('data', type=str, help='path to input .h5 file containing spectral data (str)')
    parser.add_argument('output', type=str, help='path to output folder (str)')
    parser.add_argument('--beta', type=float, default=0.12087601109118155,
                        help='ccs calibration parameter beta (float, default=0.1208)')
    parser.add_argument('--tfix', type=float, default=1.9817908554141468,
                        help='ccs calibration parameter tfix (float, default=1.9818)')
    parser.add_argument('--threshold', type=int, default=1000,
                        help='intensity threshold (int, default=1000)')

    # parse arguments
    args = parser.parse_args()

    # run
    main(args.data, args.output, args.beta, args.tfix, args.threshold)
