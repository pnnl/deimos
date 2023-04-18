import deimos
import numpy as np
import pandas as pd
import pytest

from tests import localfile


@pytest.fixture()
def ms1():
    return deimos.load(localfile('resources/example_data.h5'),
                       key='ms1')


@pytest.fixture()
def ms2():
    return deimos.load(localfile('resources/example_data.h5'),
                       key='ms2')


@pytest.mark.parametrize('threshold,length',
                         [(0, 1229849),
                          (1E3, 20564),
                          (1E4, 2522),
                          (1E6, 0)])
def test_threshold(ms1, threshold, length):
    thresholded = deimos.threshold(ms1, by='intensity', threshold=threshold)
    assert len(thresholded.index) == length


@pytest.mark.parametrize('keep,how,length',
                         [(['mz', 'drift_time'], np.sum, 651076),
                          (['retention_time'], np.max, 41),
                          (['mz', 'drift_time', 'retention_time'], np.sum, 1958605)])
def test_collapse(ms1, keep, how, length):
    collapsed = deimos.collapse(ms1, keep=keep, how=how)

    assert all(collapsed.columns == keep + ['intensity'])
    assert len(collapsed.index) == length


@pytest.mark.parametrize('by,loc,tol,return_index',
                         [(['mz', 'drift_time', 'retention_time'],
                           [212.0, 17.2, 4.79],
                           [1.06E-3, 1.03, 0.8],
                           False),
                          (['mz', 'drift_time', 'retention_time'],
                           [212.0, 17.2, 4.79],
                           [1.06E-3, 1.03, 0.8],
                           True),
                          ('mz', 212.0, 1.06E-3, False),
                          ('mz', 212.0, 1.06E-3, True)])
def test_locate(ms1, by, loc, tol, return_index):
    if return_index is True:
        subset, idx = deimos.locate(ms1, by=by, loc=loc, tol=tol,
                                    return_index=return_index)
    else:
        subset = deimos.locate(ms1, by=by, loc=loc, tol=tol,
                               return_index=return_index)

    assert all(subset.columns == ms1.columns)

    for x, dx, dim in zip(deimos.utils.safelist(loc),
                          deimos.utils.safelist(tol),
                          deimos.utils.safelist(by)):
        assert subset[dim].min() >= x - dx
        assert subset[dim].max() <= x + dx

    if return_index is True:
        assert len(idx) == len(ms1.index)


@pytest.mark.parametrize('by,loc,tol,return_index',
                         [(['mz', 'drift_time', 'retention_time'],
                           [212.0, 17.2, 4.79],
                           [1.06E-3, 1.03, 0.8],
                           False),
                          (['mz', 'drift_time', 'retention_time'],
                           [212.0, 17.2, 4.79],
                           [1.06E-3, 1.03, 0.8],
                           True)])
def test_locate_pass_none(by, loc, tol, return_index):
    if return_index is True:
        subset, idx = deimos.locate(None, by=by, loc=loc, tol=tol,
                                    return_index=return_index)
        assert subset is None
        assert idx is None
    else:
        subset = deimos.locate(None, by=by, loc=loc, tol=tol,
                               return_index=return_index)

        assert subset is None


@pytest.mark.parametrize('by,loc,tol,return_index',
                         [('mz', -212.0, 1.06E-3, False),
                          ('mz', 212.0E3, 1.06E-3, True)])
def test_locate_return_none(ms1, by, loc, tol, return_index):
    if return_index is True:
        subset, idx = deimos.locate(ms1, by=by, loc=loc, tol=tol,
                                    return_index=return_index)
        assert subset is None
        assert len(idx) == len(ms1.index)
        assert all(idx) is False
    else:
        subset = deimos.locate(ms1, by=by, loc=loc, tol=tol,
                               return_index=return_index)

        assert subset is None


@pytest.mark.parametrize('by,low,high,return_index',
                         [(['mz', 'drift_time', 'retention_time'],
                           [211.0, 16.2, 4.00],
                           [213.0, 18.2, 5.16],
                           False),
                          (['mz', 'drift_time', 'retention_time'],
                           [211.0, 16.2, 4.00],
                           [213.0, 18.2, 5.16],
                           True),
                          ('mz', 211.0, 213.0, False),
                          ('mz', 211.0, 213.0, True)])
def test_slice(ms1, by, low, high, return_index):
    if return_index is True:
        subset, idx = deimos.slice(ms1, by=by, low=low, high=high,
                                   return_index=return_index)
    else:
        subset = deimos.slice(ms1, by=by, low=low, high=high,
                              return_index=return_index)

    assert all(subset.columns == ms1.columns)

    for ll, hh, dim in zip(deimos.utils.safelist(low),
                           deimos.utils.safelist(high),
                           deimos.utils.safelist(by)):
        assert subset[dim].min() >= ll
        assert subset[dim].max() <= hh

    if return_index is True:
        assert len(idx) == len(ms1.index)


@pytest.mark.parametrize('by,low,high,return_index',
                         [(['mz', 'drift_time', 'retention_time'],
                           [211.0, 16.2, 4.00],
                           [213.0, 18.2, 5.16],
                           False),
                          (['mz', 'drift_time', 'retention_time'],
                           [211.0, 16.2, 4.00],
                           [213.0, 18.2, 5.16],
                           True)])
def test_slice_pass_none(by, low, high, return_index):
    if return_index is True:
        subset, idx = deimos.slice(None, by=by, low=low, high=high,
                                   return_index=return_index)
        assert subset is None
        assert idx is None
    else:
        subset = deimos.slice(None, by=by, low=low, high=high,
                              return_index=return_index)

        assert subset is None


@pytest.mark.parametrize('by,low,high,return_index',
                         [('mz', -211.0, -213.0, False),
                          ('mz', 211.0E3, 213.0E3, True)])
def test_slice_return_none(ms1, by, low, high, return_index):
    if return_index is True:
        subset, idx = deimos.slice(ms1, by=by, low=low, high=high,
                                   return_index=return_index)
        assert subset is None
        assert len(idx) == len(ms1.index)
        assert all(idx) is False
    else:
        subset = deimos.slice(ms1, by=by, low=low, high=high,
                              return_index=return_index)

        assert subset is None


@pytest.mark.parametrize('by,loc,low,high,relative,return_index',
                         [(['mz', 'drift_time', 'retention_time'],
                           [212.0, 17.2, 4.79],
                           [-0.1, -0.03, -0.3],
                           [1, 0.03, 0.3],
                           [False, True, False],
                           False)])
def test_locate_asym(ms1, by, loc, low, high, relative, return_index):
    if return_index is True:
        subset, idx = deimos.locate_asym(ms1, by=by, low=low, high=high,
                                         return_index=return_index)
    else:
        subset = deimos.locate_asym(ms1, loc=loc, by=by, low=low, high=high,
                                    return_index=return_index)

    assert all(subset.columns == ms1.columns)

    for x, ll, hh, rel, dim in zip(deimos.utils.safelist(loc),
                                   deimos.utils.safelist(low),
                                   deimos.utils.safelist(high),
                                   deimos.utils.safelist(relative),
                                   deimos.utils.safelist(by)):
        if rel is True:
            assert subset[dim].min() >= x * (1 + ll)
            assert subset[dim].max() <= x * (1 + hh)
        else:
            assert subset[dim].min() >= x + ll
            assert subset[dim].max() <= x + hh

    if return_index is True:
        assert len(idx) == len(ms1.index)


@pytest.mark.parametrize('by,loc,low,high,relative,return_index',
                         [(['mz', 'drift_time', 'retention_time'],
                           [212.0, 17.2, 4.79],
                           [-0.1, -0.03, -0.3],
                           [1, 0.03, 0.3],
                           [False, True, False],
                           False)])
def test_locate_asym_pass_none(ms1, by, loc, low, high, relative, return_index):
    if return_index is True:
        subset, idx = deimos.locate_asym(None, by=by, loc=loc, low=low, high=high,
                                         relative=relative, return_index=return_index)
        assert subset is None
        assert idx is None
    else:
        subset = deimos.locate_asym(None, by=by, loc=loc, low=low, high=high,
                                    relative=relative, return_index=return_index)

        assert subset is None


@pytest.mark.parametrize('by,loc,low,high,relative,return_index',
                         [(['mz', 'drift_time', 'retention_time'],
                           [212.0, 17.2, 4.79],
                           [0.1, 0.03, 0.3],
                           [-1, -0.03, -0.3],
                           [False, True, False],
                           False)])
def test_locate_asym_return_none(ms1, by, loc, low, high, relative, return_index):
    if return_index is True:
        subset, idx = deimos.locate_asym(None, by=by, loc=loc, low=low, high=high,
                                         relative=relative, return_index=return_index)
        assert subset is None
        assert len(idx) == len(ms1.index)
        assert all(idx) is False
    else:
        subset = deimos.locate_asym(None, by=by, loc=loc, low=low, high=high,
                                    relative=relative, return_index=return_index)

        assert subset is None


class TestPartitions:

    @pytest.mark.parametrize('split_on,size,overlap',
                             [('mz', 1000, 0.05),
                              ('mz', 2000, 0.5)])
    def test_init(self, ms1, split_on, size, overlap):
        partitions = deimos.partition(ms1, split_on=split_on, size=size,
                                      overlap=overlap)

        assert partitions.features.equals(ms1)
        assert partitions.split_on == split_on
        assert partitions.size == size

    @pytest.mark.parametrize('split_on,size,overlap',
                             [('mz', 1000, 0.05),
                              ('mz', 2000, 0.5)])
    def test__compute_splits(self, ms1, split_on, size, overlap):
        partitions = deimos.partition(ms1, split_on=split_on, size=size,
                                      overlap=overlap)

        idx_unq = np.unique(ms1[split_on].values)

        assert len(partitions.bounds) == np.ceil(len(idx_unq) / size)
        assert len(partitions.fbounds) == np.ceil(len(idx_unq) / size)

        for i, (b, fb) in enumerate(zip(partitions.bounds,
                                        partitions.fbounds)):
            # first partition
            if i < 1:
                assert b[0] == fb[0]
                assert 2 * (b[1] - fb[1]) - overlap < 1E-3

            # middle partitions
            elif i < len(partitions.bounds) - 1:
                assert abs(2 * (fb[0] - b[0]) - overlap) < 1E-3
                assert abs(2 * (b[1] - fb[1]) - overlap) < 1E-3

            # last partition
            else:
                assert abs(2 * (fb[0] - b[0]) - overlap) < 1E-3
                assert b[1] == fb[1]

    def test_iter(self, ms1):
        partitions = deimos.partition(ms1, split_on='mz', size=2000,
                                      overlap=1)
        for i, part in enumerate(partitions):
            assert type(part) is pd.DataFrame
            assert all(part.columns == ms1.columns)

            lb, ub = partitions.bounds[i]

            assert part['mz'].min() >= lb
            assert part['mz'].max() <= ub

        assert i == 82

    @pytest.mark.parametrize('processes',
                             [(1),
                              (2)])
    def test_map(self, ms1, processes):
        partitions = deimos.partition(ms1, split_on='mz', size=2000,
                                      overlap=1)
        pres = partitions.map(deimos.threshold, processes=processes,
                              by='intensity', threshold=1E3)
        pres = pres.sort_values(by=['mz',
                                    'drift_time',
                                    'retention_time']).reset_index(drop=True)

        res = deimos.threshold(ms1, by='intensity', threshold=1E3)
        res = res.sort_values(by=['mz',
                                  'drift_time',
                                  'retention_time']).reset_index(drop=True)

        assert pres.equals(res)

    @pytest.mark.parametrize('processes',
                             [(1),
                              (2)])
    def test_zipmap(self, ms1, ms2, processes):
        ms1 = deimos.threshold(ms1, by='intensity', threshold=1E4)
        ms2 = deimos.threshold(ms2, by='intensity', threshold=1E4)

        partitions = deimos.partition(ms1, split_on='mz', size=2000,
                                      overlap=1)

        pres_a, pres_b = partitions.zipmap(deimos.alignment.tolerance, ms2,
                                           processes=processes,
                                           dims=['mz', 'drift_time',
                                                 'retention_time'],
                                           tol=[5E-6, 0.025, 0.3],
                                           relative=[True, True, False])

        res_a, res_b = deimos.alignment.tolerance(ms1, ms2,
                                                  dims=['mz', 'drift_time',
                                                        'retention_time'],
                                                  tol=[5E-6, 0.025, 0.3],
                                                  relative=[True, True, False])

        assert pres_a.equals(res_a)
        assert pres_b.equals(res_b)


@pytest.mark.parametrize('split_on,size,overlap',
                         [('mz', 1000, 0.05),
                          ('mz', 2000, 0.5)])
def test_partition(ms1, split_on, size, overlap):
    partitions = deimos.partition(ms1, split_on=split_on, size=size,
                                  overlap=overlap)

    assert type(partitions) is deimos.subset.Partitions


class TestMultiSamplePartitions:
    def test_init(self):
        with pytest.raises(NotImplementedError):
            raise NotImplementedError

    def test__compute_splits(self):
        with pytest.raises(NotImplementedError):
            raise NotImplementedError

    def test_iter(self):
        with pytest.raises(NotImplementedError):
            raise NotImplementedError

    def test_next(self):
        with pytest.raises(NotImplementedError):
            raise NotImplementedError

    def test_map(self):
        with pytest.raises(NotImplementedError):
            raise NotImplementedError


def test_multi_sample_partition():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError
