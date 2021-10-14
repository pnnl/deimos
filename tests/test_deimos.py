import deimos
import pytest


def test_has_version():
    assert hasattr(deimos, '__version__')


@pytest.mark.parametrize('module',
                         [('alignment'),
                          ('calibration'),
                          ('deconvolution'),
                          ('filters'),
                          ('grid'),
                          ('io'),
                          ('isotopes'),
                          ('peakpick'),
                          ('plot'),
                          ('subset'),
                          ('utils')])
def test_has_module(module):
    assert hasattr(deimos, module)


@pytest.mark.parametrize('attr',
                         [('read_mzml'),
                          ('save_hdf'),
                          ('load_hdf'),
                          ('save_mgf'),
                          ('threshold'),
                          ('collapse'),
                          ('locate'),
                          ('locate_asym'),
                          ('slice'),
                          ('Partitions'),
                          ('partition'),
                          ('MultiSamplePartitions'),
                          ('multi_sample_partition')])
def test_toplevel_imports(attr):
    assert hasattr(deimos, attr)


@pytest.mark.parametrize('attr',
                         [('match'),
                          ('tolerance'),
                          ('fit_spline'),
                          ('agglomerative_clustering'),
                          ('join')])
def test_alignment_namespace(attr):
    assert hasattr(deimos.alignment, attr)


@pytest.mark.parametrize('attr',
                         [('ArrivalTimeCalibration'),
                          ('calibrate_ccs')])
def test_calibration_namespace(attr):
    assert hasattr(deimos.calibration, attr)


@pytest.mark.parametrize('attr',
                         [('get_1D_profiles'),
                          ('MS2Deconvolution'),
                          ('deconvolve_ms2')])
def test_calibration_namespace(attr):
    assert hasattr(deimos.deconvolution, attr)


@pytest.mark.parametrize('attr',
                         [('stdev'),
                          ('maximum'),
                          ('minimum'),
                          ('sum'),
                          ('mean'),
                          ('matched_gaussian'),
                          ('signal_to_noise_ratio'),
                          ('count'),
                          ('kurtosis')])
def test_filters_namespace(attr):
    assert hasattr(deimos.filters, attr)


@pytest.mark.parametrize('attr',
                         [('data2grid'),
                          ('grid2df')])
def test_grid_namespace(attr):
    assert hasattr(deimos.grid, attr)


@pytest.mark.parametrize('attr',
                         [('read_mzml'),
                          ('save_hdf'),
                          ('load_hdf'),
                          ('save_mgf')])
def test_io_namespace(attr):
    assert hasattr(deimos.io, attr)


@pytest.mark.parametrize('attr',
                         [('detect')])
def test_isotopes_namespace(attr):
    assert hasattr(deimos.isotopes, attr)


@pytest.mark.parametrize('attr',
                         [('local_maxima')])
def test_peakpick_namespace(attr):
    assert hasattr(deimos.peakpick, attr)


@pytest.mark.parametrize('attr',
                         [('fill_between'),
                          ('stem'),
                          ('grid'),
                          ('multipanel')])
def test_plot_namespace(attr):
    assert hasattr(deimos.plot, attr)


@pytest.mark.parametrize('attr',
                         [('threshold'),
                          ('collapse'),
                          ('locate'),
                          ('locate_asym'),
                          ('slice'),
                          ('Partitions'),
                          ('partition'),
                          ('MultiSamplePartitions'),
                          ('multi_sample_partition')])
def test_subset_namespace(attr):
    assert hasattr(deimos.subset, attr)


@pytest.mark.parametrize('attr',
                         [('safelist'),
                          ('check_length'),
                          ('detect_dims')])
def test_utils_namespace(attr):
    assert hasattr(deimos.utils, attr)


@pytest.mark.parametrize('toplevel,attr',
                         [(deimos.read_mzml, deimos.io.read_mzml),
                          (deimos.save_hdf, deimos.io.save_hdf),
                          (deimos.load_hdf, deimos.io.load_hdf),
                          (deimos.save_mgf, deimos.io.save_mgf),
                          (deimos.threshold, deimos.subset.threshold),
                          (deimos.collapse, deimos.subset.collapse),
                          (deimos.locate, deimos.subset.locate),
                          (deimos.locate_asym, deimos.subset.locate_asym),
                          (deimos.slice, deimos.subset.slice),
                          (deimos.Partitions, deimos.subset.Partitions),
                          (deimos.partition, deimos.subset.partition)])
def test_toplevel_namespace(toplevel, attr):
    assert toplevel == attr
