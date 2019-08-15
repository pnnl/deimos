import spextractor as spx
from os.path import *
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


plt.switch_backend('agg')


def main(exp_path, output_path, targets_path, mode,
         beta, tfix, mz_res, mz_frames, dt_res, dt_frames, dt_frames_plot=30):
    # output directory
    if not exists(output_path):
        os.makedirs(output_path)

    # adducts
    if mode == 'pos':
        adducts = {'+H': 1.00784,
                   '+Na': 22.989769}
    elif mode == 'neg':
        adducts = {'-H': -1.00784}

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
    dfs = []
    for idx, row in targets.iterrows():
        for adduct, adduct_mass in adducts.items():
            # feature
            mz_i = row['Exact Mass'] + adduct_mass
            dt_i = c.ccs2arrival(mz_i, row['[M%s] CCS' % adduct])

            # check if CCS present
            if pd.isna(dt_i):
                break

            # # output path
            # feature_path = join(output_path, '%s_%s' % (row['InChI Key'], adduct))
            # if not exists(feature_path):
            #     os.makedirs(feature_path)

            # targeted search
            ms1 = spx.targeted.find_feature(data['ms1'],
                                            mz=mz_i,
                                            dt=dt_i,
                                            dt_tol=dt_res * dt_frames,
                                            mz_tol=[mz_res * mz_frames, mz_res * mz_frames * 2])

            ms2 = spx.targeted.find_feature(data['ms2'],
                                            dt=dt_i,
                                            dt_tol=dt_res * dt_frames)

            # wide in mz and dt
            ms1_plot = spx.targeted.find_feature(data['ms1'],
                                                 mz=mz_i,
                                                 dt=dt_i,
                                                 dt_tol=dt_res * dt_frames_plot,
                                                 mz_tol=[1.5, 3.5])
            # wide in mz
            ms1_plot_mz = spx.targeted.find_feature(data['ms1'],
                                                    mz=mz_i,
                                                    dt=dt_i,
                                                    dt_tol=dt_res * dt_frames,
                                                    mz_tol=[1.5, 3.5])

            # wide in dt
            ms1_plot_dt = spx.targeted.find_feature(data['ms1'],
                                                    mz=mz_i,
                                                    dt=dt_i,
                                                    dt_tol=dt_res * dt_frames_plot,
                                                    mz_tol=[mz_res * mz_frames, mz_res * mz_frames * 2])

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
                mz_exp = np.nan
                dt_exp = np.nan
                mz_int = np.nan
                ms1_mz = None
                ms1_dt = None

            if ms2 is not None:
                ms2_mz = ms2.groupby(by='mz', as_index=False).agg({'intensity': np.sum})

                # # save ms2
                # spx.utils.save_hdf(ms2_mz, join(feature_path, 'ms2.h5'))

                # sort
                ms2_out = ms2_mz.loc[(ms2_mz['intensity'] > 100) & (ms2_mz['mz'] <= mz_i + 10), :].sort_values(by='mz')
                ms2_out = ''.join(['%.4f %i;' % (mz, i) for mz, i in zip(ms2_out['mz'].values, ms2_out['intensity'].values)])
            else:
                ms2_out = np.nan

            # append
            res = {}
            res['ikey'] = row['InChI Key']
            res['adduct'] = adduct
            res['ms1_mz_lib'] = mz_i
            res['dt_lib'] = dt_i
            res['ms1_mz_exp'] = mz_exp
            res['dt_exp'] = dt_exp
            res['ms1_intensity'] = mz_int
            res['ms2'] = ms2_out

            dfs.append(pd.DataFrame(res, index=[0]))

            # overall plot
            fig = plt.figure(figsize=(7.5, 7.5), dpi=900)
            fig.suptitle('%s [M%s]\nlibrary m/z: %.2f, dt: %.2f,\nexp m/z: %.2f, dt: %.2f, intensity: %.2E' % (row['InChI Key'], adduct, mz_i, dt_i, mz_exp, dt_exp, mz_int))

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
                ax2.plot((mz_i + mz_res * mz_frames * 2, mz_i - mz_res * mz_frames),
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
                ax3.plot((mz_i + mz_res * mz_frames * 2, mz_i - mz_res * mz_frames),
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
                    spx.plot.stem(ms2_mz.loc[ms2_mz['intensity'] > 0, 'mz'],
                                  ms2_mz.loc[ms2_mz['intensity'] > 0, 'intensity'],
                                  width=1, xlabel='m/z', ylabel='intensity', ticks=4, ax=ax5)
                elif ms2_mz['intensity'].max() > 100:
                    spx.plot.stem(ms2_mz['mz'], ms2_mz['intensity'], width=1,
                                  xlabel='m/z', ylabel='intensity', ticks=4, ax=ax5)
                else:
                    spx.plot.stem(ms2_mz['mz'], ms2_mz['intensity'], width=1,
                                  xlabel='m/z', ylabel='intensity', ticks=4, ax=ax5)
                    ax5.set_ylim(0, 100)
                ax5.set_xlim(0, mz_i + 10)

            plt.tight_layout(rect=[0, 0.03, 1, 0.90])
            # plt.savefig(join(feature_path, 'figures.png'))
            plt.savefig(join(output_path, '%s_%s.png' % (row['InChI Key'], adduct)))
            plt.close()

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df.sort_values(by='ms1_intensity', ascending=False)
    df.to_csv(join(output_path, '%s.tsv' % splitext(basename(exp_path))[0]), sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpExtractor: target MS2 extraction script.')
    parser.add_argument('-v', '--version', action='version', version=spx.__version__, help='print version and exit')
    parser.add_argument('data', type=str, help='path to input .h5 file containing spectral data (str)')
    parser.add_argument('output', type=str, help='path to output folder (str)')
    parser.add_argument('targets', type=str, help='path to input .xlsx spreadsheet containing targets (str)')
    parser.add_argument('--mode', choices=['pos', 'neg'], help='experiment acquisition mode')
    # parser.add_argument('--beta', type=float, default=0.12087601109118155,
    #                     help='ccs calibration parameter beta (float, default=0.1208)')
    # parser.add_argument('--tfix', type=float, default=1.9817908554141468,
    #                     help='ccs calibration parameter tfix (float, default=1.9818)')
    parser.add_argument('--mzres', type=float, default=0.005,
                        help='m/z resolution (float, default=0.005)')
    parser.add_argument('--dtres', type=float, default=0.12,
                        help='drift time resolution (float, default=0.12)')
    parser.add_argument('--mzframes', type=int, default=20,
                        help='m/z frames search window (int, default=20)')
    parser.add_argument('--dtframes', type=int, default=5,
                        help='drift time frames search window (int, default=5)')

    # parse arguments
    args = parser.parse_args()

    # for now, hard code
    if args.mode == 'pos':
        beta = 0.12087601109118155
        tfix = 1.9817908554141468
    elif args.mode == 'neg':
        beta = 0.13228105974417567
        tfix = -0.40169687139721333

    # run
    main(args.data, args.output, args.targets, args.mode, beta, tfix,
         args.mzres, args.mzframes, args.dtres, args.dtframes, dt_frames_plot=30)
