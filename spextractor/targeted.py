from spextractor.utils import safelist, check_length
import pandas as pd


def find_feature(data, by=['mz', 'drift_time', 'retention_time'],
                 loc=[0, 0, 0], tol=[0, 0, 0]):
    # safely cast to list
    by = safelist(by)
    loc = safelist(loc)
    tol = safelist(tol)

    # check dims
    check_length([by, loc, tol])

    # extend columns
    cols = data.columns
    cidx = [cols.get_loc(x) for x in by]

    # subset by each dim
    data = data.values
    for i, x, dx in zip(cidx, loc, tol):
        data = data[(data[:, i] <= x + dx) & (data[:, i] >= x - dx)]

    # data found
    if data.shape[0] > 0:
        return pd.DataFrame(data, columns=cols)

    # no data
    return None
