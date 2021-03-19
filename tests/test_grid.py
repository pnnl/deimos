import pytest
import deimos
import numpy as np
from tests import localfile


@pytest.fixture()
def ms1():
    return deimos.load_hdf(localfile('resources/example_data.h5'),
                           level='ms1')


@pytest.mark.parametrize('features,dims',
                         [(['mz', 'drift_time', 'retention_time'], (164659, 18, 41)),
                          (['mz', 'drift_time'], (164659, 18)),
                          (['retention_time'], (41,))])
def test_data2grid(ms1, features, dims):
    edges, grid = deimos.grid.data2grid(ms1, features=features)

    # Index checks
    assert len(edges) == len(dims)

    for edge, expected in zip(edges, dims):
        assert len(edge) == expected

    # Grid checks
    assert grid.shape == dims


@pytest.mark.parametrize('features,additional,zeros',
                         [(['mz', 'drift_time', 'retention_time'], False, True),
                          (['mz', 'drift_time', 'retention_time'], True, False)])
def test_grid2df(ms1, features, additional, zeros):
    edges, grid = deimos.grid.data2grid(ms1)

    if additional is True:
        aux = {'extra': np.ones_like(grid)}
    else:
        aux = None

    df = deimos.grid.grid2df(edges, grid,
                             features=features, additional=aux,
                             preserve_explicit_zeros=zeros)

    # Check feature dimensions are present
    for f in features:
        assert f in df.columns

    # Check additional is present/absent
    if additional is True:
        assert 'extra' in df.columns
    else:
        assert 'extra' not in df.columns

    # Check intensity is present
    assert 'intensity' in df.columns

    # Check length
    if zeros is True:
        assert len(df.index) == len(ms1.index)
    else:
        assert len(df.loc[df['intensity'] > 0, :].index) == len(ms1.loc[ms1['intensity'] > 0, :].index)
