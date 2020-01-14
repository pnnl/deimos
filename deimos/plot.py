import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import deimos
import pandas as pd
from scipy.interpolate import griddata


def _ceil(value):
    ndigits = int(np.log10(value))
    upper = round(value.max(), -ndigits)
    if upper < value.max():
        upper = upper + 10 ** ndigits
    return upper


def fill_between(x, y, xlabel='drift time (ms)', ylabel='intensity',
                 ax=None, ticks=5, dpi=600):
    # sort
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    df = pd.DataFrame({'x': x, 'y': y})
    df = df.groupby(by='x', as_index=False).sum()

    x = df['x'].values
    y = df['y'].values

    # initialize figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.85, 3), dpi=dpi)

    # plot
    ax.plot(x, y, color='black')
    ax.fill_between(x, y, alpha=0.1, color='black')

    # axis setup
    ax.set_ylim(0, _ceil(y.max()))
    ax.yaxis.set_major_locator(tick.MaxNLocator(nbins=ticks, integer=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

    # axis labels
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')

    return ax


def stem(x, y, points=False, xlabel='m/z', ylabel='intensity',
         width=0.1, ax=None, ticks=4, dpi=600):

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.85, 3), dpi=dpi)

    # plot
    _, stemlines, _ = ax.stem(x, y,
                              markerfmt=" ",
                              basefmt=" ",
                              linefmt="k-",
                              use_line_collection=True)
    plt.setp(stemlines, 'linewidth', width)
    if points is True:
        ax.scatter(x, y, s=1, c='k')

    # axis setup
    ax.set_ylim(0, _ceil(y.max()))
    ax.yaxis.set_major_locator(tick.MaxNLocator(nbins=ticks, integer=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

    # axis labels
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')

    return ax


def grid(data, features=['mz', 'drift_time'], method='linear', gridsize=1000j, log=False, cmap='gray_r',
         ticks=4, ax=None, dpi=600):
    # safely cast to list
    features = deimos.utils.safelist(features)

    # check dims
    if len(features) != 2:
        raise ValueError('grid plots only support in 2 dimensions')

    # initialize figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.85, 3), dpi=dpi)

    # split features and values
    points = data.loc[:, features].values
    values = data.loc[:, ['intensity']].values.flatten()

    # interpolation grid
    grid_x, grid_y = np.mgrid[data[features[0]].min():data[features[0]].max():gridsize,
                              data[features[1]].min():data[features[1]].max():gridsize]

    # grid data
    gridded = np.nan_to_num(griddata(points, values, (grid_x, grid_y), method=method))

    # plot
    ax.pcolormesh(grid_x, grid_y, gridded, zorder=1, cmap=cmap)

    # axis format
    ax.xaxis.set_major_locator(tick.MaxNLocator(nbins=ticks, integer=True))

    # axis labels
    names = _rename(features)
    ax.set_xlabel(names[0], fontweight='bold')
    ax.set_ylabel(names[1], fontweight='bold')

    return ax


def _rename(features):
    names = ['m/z' if x == 'mz' else x for x in features]
    names = ['retention time (min)' if x == 'retention_time' else x for x in names]
    names = ['drift time (ms)' if x == 'drift_time' else x for x in names]
    return names
