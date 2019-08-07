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

# how far to search?
# how much to sum across?
# fwhm for now
mz_res = 0.005
mz_frames = 20

dt_res = 0.12
dt_frames = 5
dt_frames_plot = 30

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

            # output path
            feature_path = join(output_path, '%s_%s' % (row['Name'], adduct))
            if not exists(feature_path):
                os.makedirs(feature_path)

            # targeted search
            ms1 = spx.targeted.find_feature(data['ms1'],
                                            mz=mz_i,
                                            dt=dt_i,
                                            dt_tol=dt_res * dt_frames,
                                            mz_tol=mz_res * mz_frames)

            ms2 = spx.targeted.find_feature(data['ms2'],
                                            dt=dt_i,
                                            dt_tol=dt_res * dt_frames)

            # wide in mz and dt
            ms1_plot = spx.targeted.find_feature(data['ms1'],
                                                 mz=mz_i,
                                                 dt=dt_i,
                                                 dt_tol=dt_res * dt_frames_plot,
                                                 mz_tol=3.5)
            # wide in mz
            ms1_plot_mz = spx.targeted.find_feature(data['ms1'],
                                                    mz=mz_i,
                                                    dt=dt_i,
                                                    dt_tol=dt_res * dt_frames,
                                                    mz_tol=3.5)

            # wide in dt
            ms1_plot_dt = spx.targeted.find_feature(data['ms1'],
                                                    mz=mz_i,
                                                    dt=dt_i,
                                                    dt_tol=dt_res * dt_frames_plot,
                                                    mz_tol=mz_res * mz_frames)

            if ms1 is not None:
                # sum
                ms1_mz = ms1.groupby(by='mz', as_index=False).agg({'intensity': np.sum})
                ms1_dt = ms1.groupby(by='drift_time', as_index=False).agg({'intensity': np.sum})

                # experimental mz
                mz_exp = (ms1_mz['mz'] * ms1_mz['intensity'] / ms1_mz['intensity'].sum()).sum()

                # experimental dt
                dt_exp = (ms1_dt['drift_time'] * ms1_dt['intensity'] / ms1_dt['intensity'].sum()).sum()

                # ms1 intensity
                mz_int = ms1_mz['intensity'].sum()
            else:
                mz_exp = mz_i
                dt_exp = dt_i
                mz_int = 0
                ms1_mz = None
                ms1_dt = None

            print(row['Name'])
            print('\tintensity:', mz_int)

            if ms2 is not None:
                ms2_mz = ms2.groupby(by='mz', as_index=False).agg({'intensity': np.sum})

                # save ms2
                spx.utils.save_hdf(ms2_mz, join(feature_path, 'ms2.h5'))
            else:
                ms2_mz = None

            # append
            ms1_res['ikey'].append(row['InChI Key'])
            ms1_res['adduct'].append(adduct)
            ms1_res['mz'].append(mz_exp)
            ms1_res['drift_time'].append(dt_exp)
            ms1_res['intensity'].append(mz_int)

            # overall plot
            fig = plt.figure(figsize=(7.5, 7.5), dpi=900)
            fig.suptitle('%s [M%s]\nm/z: %.2f, dt: %.2f, intensity: %.2E' % (row['Name'], adduct, mz_exp, dt_exp, mz_int))

            gs = fig.add_gridspec(3, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
            ax5 = fig.add_subplot(gs[2, :])

            # features
            if ms1_plot is not None:
                spx.plot.grid(ms1_plot['mz'], ms1_plot['drift_time'], ms1_plot['intensity'],
                              log=False, x_res=mz_res, y_res=dt_res,
                              xlabel='m/z', ylabel='drift time (ms)', ax=ax1)
                ax1.scatter(mz_i, dt_i, s=2, color='red')
                ax1.set_xlim(mz_i - 1.5, mz_i + 3.5)

                # features log
                spx.plot.grid(ms1_plot['mz'], ms1_plot['drift_time'], ms1_plot['intensity'],
                              log=True, x_res=mz_res, y_res=dt_res,
                              xlabel='m/z', ylabel='drift time (ms)', ax=ax2)
                ax2.plot((mz_i + mz_res * mz_frames, mz_i - mz_res * mz_frames),
                         (dt_i, dt_i),
                         linewidth=1.5, color='red')
                ax2.plot((mz_i, mz_i),
                         (dt_i + dt_res * dt_frames, dt_i - dt_res * dt_frames),
                         linewidth=1.5, color='red')
                ax2.set_xlim(mz_i - 1.5, mz_i + 3.5)

            # mass profile
            if ms1_plot_mz is not None:
                ms1_plot_mz = ms1_plot_mz.groupby(by='mz', as_index=False).agg({'intensity': np.sum})

                spx.plot.stem(ms1_plot_mz['mz'], ms1_plot_mz['intensity'],
                              width=0.5, xlabel='m/z', ylabel='intensity', ticks=4, ax=ax3)
                ax3.plot((mz_i + mz_res * mz_frames, mz_i - mz_res * mz_frames),
                         (0, 0),
                         linewidth=5, color='red')
                ax3.set_xlim(mz_i - 1.5, mz_i + 3.5)

            # drift profile
            if ms1_plot_dt is not None:
                ms1_plot_dt = ms1_plot_dt.groupby(by='drift_time', as_index=False).agg({'intensity': np.sum})

                spx.plot.stem(ms1_plot_dt['drift_time'], ms1_plot_dt['intensity'],
                              width=0.5, points=True, xlabel='drift time (ms)', ylabel='intensity',
                              ticks=4, ax=ax4)
                ax4.plot((dt_i + dt_res * dt_frames, dt_i - dt_res * dt_frames),
                         (0, 0),
                         linewidth=5, color='red')
                ax4.set_xlim(dt_i - dt_res * dt_frames_plot, dt_i + dt_res * dt_frames_plot)

            # frag pattern
            if ms2 is not None:
                if ms2_mz['intensity'].max() > 10000:
                    spx.plot.stem(ms2_mz.loc[ms2_mz['intensity'] > 1000, 'mz'],
                                  ms2_mz.loc[ms2_mz['intensity'] > 1000, 'intensity'],
                                  width=1, xlabel='m/z', ylabel='intensity', ticks=4, ax=ax5)
                elif ms2_mz['intensity'].max() > 100:
                    spx.plot.stem(ms2_mz['mz'], ms2_mz['intensity'], width=1,
                                  xlabel='m/z', ylabel='intensity', ticks=4, ax=ax5)
                else:
                    spx.plot.stem(ms2_mz['mz'], ms2_mz['intensity'], width=1,
                                  xlabel='m/z', ylabel='intensity', ticks=4, ax=ax5)
                    ax5.set_ylim(0, 100)
                ax5.set_xlim(0, mz_i + 10)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(join(feature_path, 'figures.png'))
            plt.close()

    ms1_res = pd.DataFrame(ms1_res)
    ms1_res.to_csv(join(output_path, '%s.tsv' % basename(exp_path)), sep='\t', index=False)
