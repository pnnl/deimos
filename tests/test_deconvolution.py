import numpy as np
import pandas as pd
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
        # Peaks with a float64 column beside float32 coordinates, as peak picking
        # returns them
        features = pd.DataFrame(
            {
                "mz": np.array([100.0, 200.0, 300.0], dtype=np.float32),
                "drift_time": np.array([10.0, 20.0, 30.0], dtype=np.float32),
                "intensity": np.array([5.0, 6.0, 7.0], dtype=np.float32),
                "persistence": [1.0, 2.0, 3.0],
            }
        )
        # Float32 99.99 is below the first peak's lower bound in float64, the
        # common dtype of the peaks, but on it in float32
        data = pd.DataFrame(
            {
                "mz": np.array(
                    [100.0, 100.01, 100.05, 200.0, 200.0, 99.99], dtype=np.float32
                ),
                "drift_time": np.array(
                    [10.0, 10.2, 10.0, 21.0, 20.5, 10.0], dtype=np.float32
                ),
                "intensity": np.ones(6, dtype=np.float32),
            }
        )
        decon = deimos.deconvolution.MS2Deconvolution(features, data, features, data)

        decon.configure_profile_extraction(
            dims=["mz", "drift_time"],
            low=[-100e-6, -0.3],
            high=[400e-6, 0.6],
            relative=[True, False],
        )
        profiles = decon.profiler(features, data)

        assert [None if x is None else x.index.tolist() for x in profiles] == [
            [0, 1],
            [4],
            None,
        ]

    def test_apply_unknown_similarity(self):
        decon = deimos.deconvolution.MS2Deconvolution(None, None, None, None)

        with pytest.raises(ValueError, match="similarity"):
            decon.apply(dims="drift_time", resolution=0.01, similarity="spearman")

    def test_apply(self):
        with pytest.raises(NotImplementedError):
            raise NotImplementedError
