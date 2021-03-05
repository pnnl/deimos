import matplotlib.pyplot as plt
import numpy as np
import deimos
import pandas as pd
from scipy.interpolate import griddata
import types


def _ceil(value):
    ndigits = int(np.log10(value))
    upper = round(value.max(), -ndigits)
    if upper < value.max():
        upper = upper + 10 ** ndigits
    return upper


def fill_between(x, y, xlabel='Drift Time (ms)', ylabel='Intensity',
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
        fig, ax = plt.subplots(figsize=(4.85, 3), dpi=dpi, facecolor='white')

    # plot
    # ax.plot(x, y, color='black')
    ax.fill_between(x, y, color='lightgrey')

    # axis setup
    ax.set_ylim(0, None)
    # ax.yaxis.set_major_locator(tick.MaxNLocator(nbins=ticks, integer=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

    # axis labels
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')

    return ax


def stem(x, y, points=False, xlabel='m/z', ylabel='Intensity',
         width=0.1, ax=None, ticks=4, dpi=600):

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.85, 3), dpi=dpi, facecolor='white')

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
    ax.set_ylim(0, None)
    # ax.yaxis.set_major_locator(tick.MaxNLocator(nbins=ticks, integer=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

    # axis labels
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')

    return ax


def grid(data, features=['mz', 'drift_time'], method='linear', gridsize=1000j, cmap='gray_r',
         ticks=4, ax=None, dpi=600):
    # safely cast to list
    features = deimos.utils.safelist(features)

    # check dims
    if len(features) != 2:
        raise ValueError('grid plots only supported in 2 dimensions')

    # initialize figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.85, 3), dpi=dpi, facecolor='white')

    # split features and values
    points = data.loc[:, features].values
    values = data.loc[:, ['intensity']].values.flatten()

    # interpolation grid
    grid_x, grid_y = np.mgrid[data[features[0]].min():data[features[0]].max():gridsize,
                              data[features[1]].min():data[features[1]].max():gridsize]

    # grid data
    gridded = np.nan_to_num(griddata(points, values, (grid_x, grid_y), method=method))

    # plot
    ax.pcolormesh(grid_x, grid_y, gridded, zorder=1, cmap=cmap, shading='auto')

    # # axis format
    # ax.xaxis.set_major_locator(tick.MaxNLocator(nbins=ticks, integer=True))

    # axis labels
    names = _rename(features)
    ax.set_xlabel(names[0], fontweight='bold')
    ax.set_ylabel(names[1], fontweight='bold')

    return ax


def multipanel(data, method='linear', dpi=600, grid_kwargs={}):
    def _sync_y_with_x(self, event):
        self.set_xlim(event.get_ylim(), emit=False)

    def _sync_x_with_y(self, event):
        self.set_ylim(event.get_xlim(), emit=False)

    def sync_y_with_x(ax1, ax2):
        ax1.update_ylim = types.MethodType(_sync_x_with_y, ax1)
        ax2.update_xlim = types.MethodType(_sync_y_with_x, ax2)

        ax1.callbacks.connect("xlim_changed", ax2.update_xlim)
        ax2.callbacks.connect("ylim_changed", ax1.update_ylim)

        return ax1, ax2

    def sync_x_with_y(ax1, ax2):
        ax1.update_xlim = types.MethodType(_sync_y_with_x, ax1)
        ax2.update_ylim = types.MethodType(_sync_x_with_y, ax2)

        ax1.callbacks.connect("xlim_changed", ax2.update_ylim)
        ax2.callbacks.connect("ylim_changed", ax1.update_xlim)

        return ax1, ax2

    # init figure, axes
    fig = plt.figure(figsize=(6.4, 3.8), dpi=dpi, facecolor='white')
    gs = fig.add_gridspec(2, 3)
    axes = {}
    axes['mz'] = fig.add_subplot(gs[1, 0])
    axes['dt'] = fig.add_subplot(gs[1, 1])
    axes['rt'] = fig.add_subplot(gs[1, 2])
    axes['mz-dt'] = fig.add_subplot(gs[0, 0], sharex=axes['mz'])
    axes['dt-rt'] = fig.add_subplot(gs[0, 1], sharex=axes['dt'])
    axes['rt-mz'] = fig.add_subplot(gs[0, 2], sharex=axes['rt'])

    # sync axes y with x
    sync_y_with_x(axes['mz-dt'], axes['dt'])
    sync_y_with_x(axes['dt-rt'], axes['rt'])
    sync_y_with_x(axes['rt-mz'], axes['mz'])
    sync_y_with_x(axes['mz-dt'], axes['dt-rt'])
    sync_y_with_x(axes['dt-rt'], axes['rt-mz'])
    sync_y_with_x(axes['rt-mz'], axes['mz-dt'])

    # sync axes x with y
    sync_x_with_y(axes['mz'], axes['rt-mz'])
    sync_x_with_y(axes['dt'], axes['mz-dt'])
    sync_x_with_y(axes['rt'], axes['dt-rt'])
    sync_x_with_y(axes['mz-dt'], axes['rt-mz'])
    sync_x_with_y(axes['dt-rt'], axes['mz-dt'])
    sync_x_with_y(axes['rt-mz'], axes['dt-rt'])

    # mz
    tmp = deimos.collapse(data, keep='mz')
    stem(tmp['mz'], tmp['intensity'], ax=axes['mz'])
    plt.setp(axes['mz'].get_xticklabels(), ha="right", rotation=30)

    # dt
    tmp = deimos.collapse(data, keep='drift_time')
    fill_between(tmp['drift_time'], tmp['intensity'], xlabel='Drift Time (ms)', ax=axes['dt'])
    plt.setp(axes['dt'].get_xticklabels(), ha="right", rotation=30)

    # rt
    tmp = deimos.collapse(data, keep='retention_time')
    fill_between(tmp['retention_time'], tmp['intensity'], xlabel='Retention Time (min)', ax=axes['rt'])
    plt.setp(axes['rt'].get_xticklabels(), ha="right", rotation=30)

    # mz-dt
    tmp = deimos.collapse(data, keep=['mz', 'drift_time'])
    grid(tmp, features=['mz', 'drift_time'], ax=axes['mz-dt'], **grid_kwargs)
    axes['mz-dt'].xaxis.label.set_visible(False)
    axes['mz-dt'].tick_params(labelbottom=False)

    # dt-rt
    tmp = deimos.collapse(data, keep=['drift_time', 'retention_time'])
    grid(tmp, features=['drift_time', 'retention_time'], ax=axes['dt-rt'], **grid_kwargs)
    axes['dt-rt'].xaxis.label.set_visible(False)
    axes['dt-rt'].tick_params(labelbottom=False)

    # rt-mz
    tmp = deimos.collapse(data, keep=['retention_time', 'mz'])
    grid(tmp, features=['retention_time', 'mz'], ax=axes['rt-mz'], **grid_kwargs)
    axes['rt-mz'].xaxis.label.set_visible(False)
    axes['rt-mz'].tick_params(labelbottom=False)

    return axes


def _rename(features):
    names = ['m/z' if x == 'mz' else x for x in features]
    names = ['Retention Time (min)' if x == 'retention_time' else x for x in names]
    names = ['Drift Time (ms)' if x == 'drift_time' else x for x in names]
    return names
