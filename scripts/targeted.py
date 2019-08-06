import spextractor as spx
from os.path import *
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# global
exp_path = '/Users/colb804/workspace/data/ms_decon/funtimes/70960_1012101_181022_Fixed-10_19Mar19_Aspen_Infuse.h5'
output_path = '/Users/colb804/Desktop/output/'
targets_path = '/Users/colb804/workspace/data/ms_decon/funtimes/Library_InSilico.xlsx'

# run one from agilent
beta = 0.12087601109118155
tfix = 1.9817908554141468

mz_res = 0.005
mz_frames = 2

dt_res = 0.12
dt_frames = 6

mz_frames_plot = 250
dt_frames_plot = 40

adducts = {'+H': 1.00784,
           '-H': -1.00784,
           '+Na': 22.989769}

if __name__ == '__main__':
    # output directory
    if not exists(output_path):
        os.makedirs(output_path)

    # load data
    df = spx.utils.load_hdf(exp_path)

    # load features
    targets = pd.read_excel(targets_path, sheet_name='measuredLibrary')

    # calibrate
    c = spx.calibrate.ArrivalTimeCalibration()
    c.calibrate(tfix=tfix, beta=beta)

    # split by ms level
    data = {}
    data['ms1'] = df.loc[df['ms_level'] == 1, :].drop('ms_level', axis=1).reset_index(drop=True)
    data['ms2'] = df.loc[df['ms_level'] == 2, :].drop('ms_level', axis=1).reset_index(drop=True)

    # find features
    ms1_res = {'ikey': [], 'adduct': [], 'mz': [], 'drift_time': [], 'intensity': []}
    for idx, row in targets.iterrows():
        for adduct, adduct_mass in adducts.items():
            # feature
            mz_i = row['Exact Mass'] + adduct_mass
            dt_i = c.ccs2arrival(mz_i, row['[M%s] CCS' % adduct])

            # check if CCS present
            if pd.isna(dt_i):
                break

            # targeted search
            ms1 = spx.targeted.find_feature(data['ms1'],
                                            mz=mz_i,
                                            dt=dt_i,
                                            dt_tol=dt_res * dt_frames,
                                            mz_tol=mz_res * mz_frames)

            ms2 = spx.targeted.find_feature(data['ms2'],
                                            dt=dt_i,
                                            dt_tol=dt_res * dt_frames)

            if ms1 is not None and ms2 is not None:
                # sum
                ms1_mz = ms1.groupby(by='mz', as_index=False).agg({'intensity': np.sum})
                ms1_dt = ms1.groupby(by='drift_time', as_index=False).agg({'intensity': np.sum})
                ms2_mz = ms2.groupby(by='mz', as_index=False).agg({'intensity': np.sum})

                # experimental mz
                mz_exp = (ms1_mz['mz'] * ms1_mz['intensity'] / ms1_mz['intensity'].sum()).sum()

                # experimental dt
                dt_exp = (ms1_dt['drift_time'] * ms1_dt['intensity'] / ms1_dt['intensity'].sum()).sum()

                # ms1 intensity
                mz_int = ms1_mz['intensity'].sum()

                # if mz_int <= 1000:
                #     break

                print('%s [M%s] found.' % (row['Name'], adduct))
                print('\tms1 intensity: %.2E' % mz_int)

                feature_path = join(output_path, '%s_%s' % (row['Name'], adduct))
                if not exists(feature_path):
                    os.makedirs(feature_path)

                # save ms2
                spx.utils.save_hdf(ms2_mz, join(feature_path, 'ms2.h5'))

                # append
                ms1_res['ikey'].append(row['InChI Key'])
                ms1_res['adduct'].append(adduct)
                ms1_res['mz'].append(mz_exp)
                ms1_res['drift_time'].append(dt_exp)
                ms1_res['intensity'].append(mz_int)

                # for plots
                ms1_plot = spx.targeted.find_feature(data['ms1'],
                                                     mz=mz_i,
                                                     dt=dt_i,
                                                     dt_tol=dt_res * dt_frames_plot,
                                                     mz_tol=mz_res * mz_frames_plot)

                ms1_plot_mz = spx.targeted.find_feature(data['ms1'],
                                                        mz=mz_i,
                                                        dt=dt_i,
                                                        dt_tol=dt_res * dt_frames,
                                                        mz_tol=mz_res * mz_frames_plot)
                ms1_plot_mz = ms1_plot_mz.groupby(by='mz', as_index=False).agg({'intensity': np.sum})

                ms1_plot_dt = spx.targeted.find_feature(data['ms1'],
                                                        mz=mz_i,
                                                        dt=dt_i,
                                                        dt_tol=dt_res * dt_frames_plot,
                                                        mz_tol=mz_res * mz_frames)
                ms1_plot_dt = ms1_plot_dt.groupby(by='drift_time', as_index=False).agg({'intensity': np.sum})

                # overall plot
                fig, ax = plt.subplots(2, 2, dpi=900)

                # features
                spx.plot.features(ms1_plot['mz'], ms1_plot['drift_time'],
                                  ms1_plot['intensity'], log=True,
                                  mz_bins=mz_frames_plot, dt_bins=dt_frames_plot, ax=ax[0, 0])
                ax[0, 0].plot((mz_i, mz_exp), (dt_i, dt_exp), color='red')

                # mass profile
                spx.plot.frag_pattern(ms1_plot_mz['mz'],
                                      ms1_plot_mz['intensity'],
                                      ticks=5, ax=ax[0, 1])
                idx = abs(ms1_mz['mz'] - mz_i).idxmin()
                ax[0, 1].scatter(ms1_mz.loc[idx, 'mz'],
                                 ms1_mz.loc[idx, 'intensity'],
                                 color='red')

                # drift profile
                spx.plot.frag_pattern(ms1_plot_dt['drift_time'],
                                      ms1_plot_dt['intensity'],
                                      ticks=5, ax=ax[1, 1])
                idx = abs(ms1_plot_dt['drift_time'] - dt_i).idxmin()
                ax[1, 1].scatter(ms1_plot_dt.loc[idx, 'drift_time'],
                                 ms1_plot_dt.loc[idx, 'intensity'],
                                 color='red')
                ax[1, 1].set_xlabel('drift time (ms)')

                # frag pattern
                spx.plot.frag_pattern(ms2_mz.loc[ms2_mz['intensity'] > 1000, 'mz'],
                                      ms2_mz.loc[ms2_mz['intensity'] > 1000, 'intensity'],
                                      ticks=5, ax=ax[1, 0])

                plt.tight_layout()
                plt.savefig(join(feature_path, 'figures.png'))
                plt.close()

    ms1_res = pd.DataFrame(ms1_res)
    ms1_res.to_csv(join(output_path, '%s.tsv' % basename(exp_path)), sep='\t', index=False)
