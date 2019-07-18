import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
from scipy import stats


def _ceil(value):
    ndigits = int(np.log10(value))
    upper = round(value.max(), -ndigits)
    if upper < value.max():
        upper = upper + 10 ** ndigits
    return upper


def drift_time(dt, intensity,
               ax=None, ticks=5, dpi=600):
    # sort
    idx = np.argsort(dt)
    dt = dt[idx]
    intensity = intensity[idx]

    # initialize figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.85, 3), dpi=dpi)

    # plot
    ax.plot(dt, intensity, color='black')
    ax.fill_between(dt, intensity, alpha=0.1, color='black')

    # axis setup
    ax.set_ylim(0, _ceil(intensity.max()))
    ax.yaxis.set_major_locator(tick.MaxNLocator(nbins=ticks, integer=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

    # axis labels
    ax.set_xlabel('drift time (ms)', fontweight='bold')
    ax.set_ylabel('intensity', fontweight='bold')

    return ax


def frag_pattern(mz, intensity,
                 ax=None, ticks=4, dpi=600):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.85, 3), dpi=dpi)

    # plot
    _, stemlines, _ = ax.stem(mz, intensity,
                              markerfmt=" ",
                              basefmt=" ",
                              linefmt="k-",
                              use_line_collection=True)
    plt.setp(stemlines, 'linewidth', 0.1)

    # axis setup
    ax.set_ylim(0, _ceil(intensity.max()))
    ax.yaxis.set_major_locator(tick.MaxNLocator(nbins=ticks, integer=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

    # axis labels
    ax.set_xlabel('m/z', fontweight='bold')
    ax.set_ylabel('intensity', fontweight='bold')

    return ax


def features(mz, dt, intensity, mz_bins='auto', dt_bins='auto',
             ax=None, dpi=600):
    # determine bin counts
    if mz_bins == 'auto':
        mz_bins = (mz.max() - mz.min()) / np.min(np.diff(np.sort(np.unique(mz))))

    if dt_bins == 'auto':
        dt_bins = (dt.max() - dt.min()) / np.min(np.diff(np.sort(np.unique(dt))))

    # binned statistic
    H, xe, ye, bn = stats.binned_statistic_2d(mz, dt, intensity,
                                              statistic='sum',
                                              bins=(mz_bins, dt_bins))
    H = np.nan_to_num(H)
    XX, YY = np.meshgrid(xe, ye, indexing='ij')

    # initialize figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.85, 3), dpi=dpi)

    # plot
    ax.pcolormesh(XX, YY, np.log(H + 1), zorder=1, cmap='viridis')

    # axis labels
    ax.set_xlabel('m/z', fontweight='bold')
    ax.set_ylabel('drift time (ms)', fontweight='bold')

    return ax
