import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import spextractor as spx
import pandas as pd


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


def grid(x, y, z, x_res='auto', y_res='auto', log=False,
         xlabel='m/z', ylabel='drift time (ms)', ax=None, dpi=600):

    xx, yy, H = spx.grid.data2grid(x, y, z, x_res=x_res, y_res=y_res)

    if log is True:
        H = np.log(H + 1)

    # initialize figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.85, 3), dpi=dpi)

    # plot
    ax.pcolormesh(xx, yy, H, zorder=1, cmap='gray_r')

    # axis format
    ax.xaxis.set_major_locator(tick.MaxNLocator(integer=True))

    # axis labels
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')

    return ax
