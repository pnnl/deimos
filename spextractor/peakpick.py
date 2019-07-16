import numpy as np
import scipy.ndimage as ndi
from scipy import stats
from skimage.morphology import disk
import pandas as pd
import spextractor as spx


def nonmaximumsuppression(df, x='mz', y='drift_time', z='intensity',
                          xbins='auto', ybins='auto', fwhm=3, sum_peaks=True,
                          denoise=None, percentile=0.5):
    sigma = fwhm / 2.35482004503
    struct = disk(int(4 * sigma))

    if xbins.lower() == 'auto':
        # percentile?
        xbins = (df[x].max() - df[x].min()) / np.min(np.diff(np.sort(df[x].unique())))
    if ybins.lower() == 'auto':
        ybins = (df[y].max() - df[y].min()) / np.min(np.diff(np.sort(df[y].unique())))

    H, xe, ye, bn = stats.binned_statistic_2d(df[x], df[y], df[z],
                                              statistic='sum',
                                              bins=(xbins, ybins))
    H = np.nan_to_num(H)
    XX, YY = np.meshgrid(xe, ye, indexing='ij')

    # denoise block
    if denoise is None:
        pass
    elif denoise.lower() == 'gaussian':
        H = spx.denoise.gaussian(H, (sigma, sigma))
    elif denoise.lower() == 'median':
        H = spx.denoise.median(H, struct)
    elif denoise.lower() == 'percentile':
        H = spx.denoise.percentile(H, struct, p=percentile)

    # sum peaks
    if sum_peaks is True:
        H = ndi.convolve(H, struct, mode='constant')

    # peak pick
    H_max = ndi.maximum_filter(H, footprint=struct, mode='constant')
    peaks = np.where(H == H_max, H, 0)

    return {'x': XX, 'y': YY, 'z': peaks}


def grid2df(x, y, z, top=None):
    # bin centers
    x = (x[1:, 1:] + x[:-1, :-1]) / 2
    y = (y[1:, 1:] + y[:-1, :-1]) / 2

    data = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))
    data = pd.DataFrame(data, columns=['mz', 'drift_time', 'intensity'])
    data = data.loc[data['intensity'] > 0, :].sort_values(by='intensity', ascending=False).reset_index(drop=True)

    if top is not None:
        return data.loc[:top, :]
    else:
        return data
