

def find_feature(data, by=['mz', 'drift_time', 'retention_time'],
                 loc=[0, 0, 0], tol=[0, 0, 0]):

    for feature, x, dx in zip(by, loc, tol):
        data = data.loc[(data[feature] <= x + dx) &
                        (data[feature] >= x - dx), :].reset_index(drop=True)

    if len(data.index) > 0:
        return data
    else:
        return None
