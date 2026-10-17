import numpy as np
import pytest

import deimos


@pytest.mark.parametrize(
    "b,expected",
    [
        (np.array([10.0, 11.0, 12.0, 11.0, 10.0]), 0.0),
        (np.array([0.0, 3.0, 6.0, 3.0, 0.0]), 0.0),
        (np.array([0.0, -1.0, -2.0, -1.0, 0.0]), 2.0),
        (np.array([1.0, 0.0, 1.0, 0.0, 1.0]), 1 + 0.4 / np.sqrt(3.36)),
    ],
    ids=["baseline-offset", "scaled", "inverted", "unrelated"],
)
def test_pearson(b, expected):
    a = np.array([0.0, 1.0, 2.0, 1.0, 0.0])

    assert deimos.deconvolution.pearson(a, b) == pytest.approx(expected)


def test_get_1D_profiles():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError


def test_offset_correction_model():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError


class TestMS2Deconvolution:
    def test_init(self):
        with pytest.raises(NotImplementedError):
            raise NotImplementedError

    def test_construct_putative_pairs(self):
        with pytest.raises(NotImplementedError):
            raise NotImplementedError

    def test_configure_profile_extraction(self):
        with pytest.raises(NotImplementedError):
            raise NotImplementedError

    def test_apply(self):
        with pytest.raises(NotImplementedError):
            raise NotImplementedError

    def test_apply_unknown_similarity(self):
        decon = deimos.deconvolution.MS2Deconvolution(None, None, None, None)

        with pytest.raises(ValueError, match="similarity"):
            decon.apply(dims="drift_time", resolution=0.01, similarity="spearman")
