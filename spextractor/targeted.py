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
    q = ' and '.join(['{} <= {} <= {}'.format(x - dx, feature, x + dx) for feature, x, dx in zip(by, loc, tol)])
    data = data.query(q)

    # data found
    if len(data.index) > 0:
        return data

    # no data
    return None
