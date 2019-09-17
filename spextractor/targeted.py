from spextractor.utils import safelist, check_length


def find_feature(data, by=['mz', 'drift_time', 'retention_time'],
                 loc=[0, 0, 0], tol=[0, 0, 0]):
    # safely cast to list
    by = safelist(by)
    loc = safelist(loc)
    tol = safelist(tol)

    # check dims
    check_length([by, loc, tol])

    # subset by each dim
    for feature, x, dx in zip(by, loc, tol):
        data = data.loc[(data[feature] <= x + dx) &
                        (data[feature] >= x - dx), :].reset_index(drop=True)

    # if no data found
    if len(data.index) > 0:
        return data

    return None
