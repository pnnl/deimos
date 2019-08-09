

def find_feature(ms, mz=None, dt=None, mz_tol=6E-6, dt_tol=0.12):
    if not isinstance(mz_tol, list):
        mz_tol = [mz_tol, mz_tol]

    if not isinstance(dt_tol, list):
        dt_tol = [dt_tol, dt_tol]

    if (mz is not None) and (dt is not None):
        feature = ms.loc[(ms['mz'] <= mz + mz_tol[1]) &
                         (ms['mz'] >= mz - mz_tol[0]) &
                         (ms['drift_time'] <= dt + dt_tol[1]) &
                         (ms['drift_time'] >= dt - dt_tol[0]), :].reset_index(drop=True)

    elif mz is not None:
        feature = ms.loc[(ms['mz'] <= mz + mz_tol[1]) &
                         (ms['mz'] >= mz - mz_tol[0]), :].reset_index(drop=True)

    elif dt is not None:
        feature = ms.loc[(ms['drift_time'] <= dt + dt_tol[1]) &
                         (ms['drift_time'] >= dt - dt_tol[0]), :].reset_index(drop=True)

    else:
        raise ValueError('Either mz or dt must not be None.')

    if len(feature.index) > 0:
        return feature
    else:
        return None
