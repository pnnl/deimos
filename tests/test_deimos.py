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
                         [('save'),
                          ('load'),
                          ('build_index'),
                          ('build_factors'),
                          ('get_accessions'),
                          ('threshold'),
                          ('collapse'),
                          ('locate'),
                          ('locate_asym'),
                          ('slice'),
                          ('partition'),
                          ('multi_sample_partition')])
def test_toplevel_imports(attr):
    assert hasattr(deimos, attr)


@pytest.mark.parametrize('attr',
                         [('match'),
                          ('tolerance'),
                          ('fit_spline'),
                          ('agglomerative_clustering')])
def test_alignment_namespace(attr):
    assert hasattr(deimos.alignment, attr)


@pytest.mark.parametrize('attr',
                         [('CCSCalibration'),
                          ('calibrate_ccs'),
                          ('tunemix')])
def test_calibration_namespace(attr):
    assert hasattr(deimos.calibration, attr)


@pytest.mark.parametrize('attr',
                         [('get_1D_profiles'),
                          ('offset_correction_model'),
                          ('MS2Deconvolution')])
def test_calibration_namespace(attr):
    assert hasattr(deimos.deconvolution, attr)


@pytest.mark.parametrize('attr',
                         [('std'),
                          ('std_pdf'),
                          ('maximum'),
                          ('minimum'),
                          ('sum'),
                          ('mean'),
                          ('mean_pdf'),
                          ('matched_gaussian'),
                          ('count'),
                          ('skew_pdf'),
                          ('kurtosis_pdf'),
                          ('sparse_upper_star'),
                          ('sparse_mean_filter'),
                          ('sparse_weighted_mean_filter'),
                          ('sparse_median_filter'),
                          ('smooth')])
def test_filters_namespace(attr):
    assert hasattr(deimos.filters, attr)


@pytest.mark.parametrize('attr',
                         [('data2grid'),
                          ('grid2df')])
def test_grid_namespace(attr):
    assert hasattr(deimos.grid, attr)


@pytest.mark.parametrize('attr',
                         [('save'),
                          ('load'),
                          ('build_factors'),
                          ('build_index'),
                          ('get_accessions'),
                          ('load_mzml'),
                          ('save_hdf'),
                          ('load_hdf'),
                          ('load_hdf_single'),
                          ('load_hdf_multi'),
                          ('save_mgf')])
def test_io_namespace(attr):
    assert hasattr(deimos.io, attr)


@pytest.mark.parametrize('attr',
                         [('detect')])
def test_isotopes_namespace(attr):
    assert hasattr(deimos.isotopes, attr)


@pytest.mark.parametrize('attr',
                         [('local_maxima'),
                          ('persistent_homology')])
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
                         [(deimos.save, deimos.io.save),
                          (deimos.load, deimos.io.load),
                          (deimos.build_factors, deimos.io.build_factors),
                          (deimos.build_index, deimos.io.build_index),
                          (deimos.get_accessions, deimos.io.get_accessions),
                          (deimos.threshold, deimos.subset.threshold),
                          (deimos.collapse, deimos.subset.collapse),
                          (deimos.locate, deimos.subset.locate),
                          (deimos.locate_asym, deimos.subset.locate_asym),
                          (deimos.slice, deimos.subset.slice),
                          (deimos.partition, deimos.subset.partition),
                          (deimos.multi_sample_partition, deimos.subset.multi_sample_partition)])
def test_toplevel_namespace(toplevel, attr):
    assert toplevel == attr
