import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

import deimos


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%d"  # Give format here


def _ceil(value):
    ndigits = int(np.log10(value))
    upper = round(value.max(), -ndigits)
    if upper < value.max():
        upper = upper + 10**ndigits
    return upper


def fill_between(x, y, xlabel="Drift Time (ms)", ylabel="Intensity", ax=None, dpi=600):
    # Sort
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    df = pd.DataFrame({"x": x, "y": y})
    df = df.groupby(by="x", as_index=False).sum()

    x = df["x"].values
    y = df["y"].values

    # Initialize figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.85, 3), dpi=dpi, facecolor="white")

    # Plot
    # ax.plot(x, y, color='black')
    ax.fill_between(x, y, color="lightgrey")

    # Axis setup
    ax.set_ylim(0, None)
    yfmt = ScalarFormatterForceFormat()
    yfmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(yfmt)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
    oom = 10 ** np.floor(np.log10(y.max()))
    ax.yaxis.set_ticks([0, y.max() // oom * oom])

    # Axis labels
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")

    return ax


def stem(
    x, y, points=False, xlabel="m/z", ylabel="Intensity", width=0.1, ax=None, dpi=600
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.85, 3), dpi=dpi, facecolor="white")

    # Plot
    _, stemlines, _ = ax.stem(
        x, y, markerfmt=" ", basefmt=" ", linefmt="k-", use_line_collection=True
    )
    plt.setp(stemlines, "linewidth", width)
    if points is True:
        ax.scatter(x, y, s=1, c="k")

    # Axis setup
    ax.set_ylim(0, None)
    yfmt = ScalarFormatterForceFormat()
    yfmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(yfmt)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
    oom = 10 ** np.floor(np.log10(y.max()))
    ax.yaxis.set_ticks([0, y.max() // oom * oom])

    # Axis labels
    ax.set_xlabel(xlabel, fontweight="bold", fontstyle="italic")
    ax.set_ylabel(ylabel, fontweight="bold")

    return ax


def grid(
    features,
    dims=["mz", "drift_time"],
    method="nearest",
    gridsize=1000j,
    cmap="cividis",
    ax=None,
    dpi=600,
    **kwargs
):
    # Safely cast to list
    dims = deimos.utils.safelist(dims)

    # Check dims
    if len(dims) != 2:
        raise ValueError("grid plots only supported in 2 dimensions")

    # Initialize figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.85, 3), dpi=dpi, facecolor="white")

    # Split features and values
    points = features.loc[:, dims].values
    values = features.loc[:, ["intensity"]].values.flatten()

    # Interpolation grid
    grid_x, grid_y = np.mgrid[
        features[dims[0]].min() : features[dims[0]].max() : gridsize,
        features[dims[1]].min() : features[dims[1]].max() : gridsize,
    ]

    # Grid data
    gridded = (
        np.nan_to_num(griddata(points, values, (grid_x, grid_y), method=method)) + 1
    )

    # Plot
    im = ax.pcolormesh(
        grid_x, grid_y, gridded, zorder=1, cmap=cmap, shading="auto", **kwargs
    )

    # Colorbar axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.05)

    # Colorbar
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")

    # Ticks on top
    cbar.ax.xaxis.set_ticks_position("top")

    # Custom ticks for linear case
    if kwargs.get("norm") is None:
        yfmt = ScalarFormatterForceFormat()
        yfmt.set_powerlimits((0, 0))
        cbar.ax.xaxis.set_major_formatter(yfmt)

        oom = np.floor(np.log10(0.95 * gridded.max()))
        cmax = 0.95 * gridded.max() // 10**oom

        cbar.ax.xaxis.set_ticks([1, cmax * 10**oom])
        cbar.ax.xaxis.set_ticklabels(["0", r"%i$\times$10$^{%i}$" % (cmax, oom)])

    # Axis labels
    names = _rename(dims)
    ax.set_xlabel(names[0], fontweight="bold")
    ax.set_ylabel(names[1], fontweight="bold")

    return ax


def multipanel(features, method="linear", grid_kwargs={}, normalize_grid={}, dpi=600):
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

    # Init figure, axes
    fig = plt.figure(figsize=(6.4, 3.8), dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])

    axes = {}
    axes["mz"] = fig.add_subplot(gs[1, 0])
    axes["dt"] = fig.add_subplot(gs[1, 1])
    axes["rt"] = fig.add_subplot(gs[1, 2])
    axes["mz-dt"] = fig.add_subplot(gs[0, 0], sharex=axes["mz"])
    axes["dt-rt"] = fig.add_subplot(gs[0, 1], sharex=axes["dt"])
    axes["rt-mz"] = fig.add_subplot(gs[0, 2], sharex=axes["rt"])

    # Sync axes y with x
    sync_y_with_x(axes["mz-dt"], axes["dt"])
    sync_y_with_x(axes["dt-rt"], axes["rt"])
    sync_y_with_x(axes["rt-mz"], axes["mz"])
    sync_y_with_x(axes["mz-dt"], axes["dt-rt"])
    sync_y_with_x(axes["dt-rt"], axes["rt-mz"])
    sync_y_with_x(axes["rt-mz"], axes["mz-dt"])

    # Sync axes x with y
    sync_x_with_y(axes["mz"], axes["rt-mz"])
    sync_x_with_y(axes["dt"], axes["mz-dt"])
    sync_x_with_y(axes["rt"], axes["dt-rt"])
    sync_x_with_y(axes["mz-dt"], axes["rt-mz"])
    sync_x_with_y(axes["dt-rt"], axes["mz-dt"])
    sync_x_with_y(axes["rt-mz"], axes["dt-rt"])

    # Mz
    tmp = deimos.collapse(features, keep="mz")
    stem(tmp["mz"].values, tmp["intensity"].values, xlabel="m/z", ax=axes["mz"])
    plt.setp(axes["mz"].get_xticklabels(), ha="center", rotation=0)

    # Dt
    tmp = deimos.collapse(features, keep="drift_time")
    fill_between(
        tmp["drift_time"].values,
        tmp["intensity"].values,
        xlabel="Drift Time",
        ax=axes["dt"],
    )
    plt.setp(axes["dt"].get_xticklabels(), ha="center", rotation=0)

    # Rt
    tmp = deimos.collapse(features, keep="retention_time")
    fill_between(
        tmp["retention_time"].values,
        tmp["intensity"].values,
        xlabel="Retention Time",
        ax=axes["rt"],
    )
    plt.setp(axes["rt"].get_xticklabels(), ha="center", rotation=0)

    # Mz-dt
    tmp = deimos.collapse(features, keep=["mz", "drift_time"])
    grid(
        tmp,
        dims=["mz", "drift_time"],
        ax=axes["mz-dt"],
        norm=normalize_grid.get("mz-dt", None),
        **grid_kwargs
    )

    axes["mz-dt"].xaxis.label.set_visible(False)
    axes["mz-dt"].tick_params(labelbottom=False)

    # Dt-rt
    tmp = deimos.collapse(features, keep=["drift_time", "retention_time"])
    grid(
        tmp,
        dims=["drift_time", "retention_time"],
        ax=axes["dt-rt"],
        norm=normalize_grid.get("dt-rt", None),
        **grid_kwargs
    )

    axes["dt-rt"].xaxis.label.set_visible(False)
    axes["dt-rt"].tick_params(labelbottom=False)

    # Rt-mz
    tmp = deimos.collapse(features, keep=["retention_time", "mz"])
    grid(
        tmp,
        dims=["retention_time", "mz"],
        ax=axes["rt-mz"],
        norm=normalize_grid.get("rt-mz", None),
        **grid_kwargs
    )

    axes["rt-mz"].xaxis.label.set_visible(False)
    axes["rt-mz"].tick_params(labelbottom=False)

    return axes


def _rename(dims):
    names = ["m/z" if x == "mz" else x for x in dims]
    names = ["Retention Time" if x == "retention_time" else x for x in names]
    names = ["Drift Time" if x == "drift_time" else x for x in names]
    return names
