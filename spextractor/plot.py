import matplotlib.pyplot as plt
import numpy as np


def drift_time(drift_time, intensity, ax=None, dpi=600):
    # sort
    idx = np.argsort(drift_time)
    drift_time = drift_time[idx]
    intensity = intensity[idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.85, 3), dpi=dpi)

    ax.plot(drift_time, intensity)
    ax.fill_between(drift_time, intensity, alpha=0.5)
    ax.set_xlabel('drift time (ms)', fontweight='bold')
    ax.set_ylabel('intensity', fontweight='bold')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    return ax


def frag_pattern(mz, intensity, ax=None, dpi=600):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.85, 3), dpi=dpi)

    _, stemlines, _ = ax.stem(mz, intensity, use_line_collection=True, markerfmt=" ", basefmt=" ")
    plt.setp(stemlines, 'linewidth', 0.1)

    ax.set_xlabel('m/z', fontweight='bold')
    ax.set_ylabel('intensity', fontweight='bold')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    return ax
