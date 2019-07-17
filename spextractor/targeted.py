

def find_feature(ms1, ms2, mz=None, dt=None, mz_tol=1E-6, dt_tol=0.12):
    ms1_f = ms1.loc[(ms1['mz'] <= mz + mz_tol) &
                    (ms1['mz'] >= mz - mz_tol) &
                    (ms1['drift_time'] <= dt + dt_tol) &
                    (ms1['drift_time'] >= dt - dt_tol), :]

    ms2_f = ms2.loc[(ms2['drift_time'] <= dt + dt_tol) &
                    (ms2['drift_time'] >= dt - dt_tol) &
                    (ms2['mz'] <= mz - mz_tol), :]  # not sure if we want this last one

    return ms1_f, ms2_f
