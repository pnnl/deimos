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


if __name__ == '__main__':
    import numpy as np
    import time

    data = pd.DataFrame(np.random.rand(10000, 4), columns=['mz', 'drift_time', 'retention_time', 'intensity'])
    loc = [0.5, 0.5, 0.5]
    tol = [0.1, 0.1, 0.1]

    start = time.time()

    find_feature(data, loc=loc, tol=tol)

    stop = time.time()
    print(stop - start)
