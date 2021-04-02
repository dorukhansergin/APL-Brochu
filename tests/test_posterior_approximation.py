import pytest
import numpy as np
import numpy.testing as npt
from apl.posterior_approximation import LogLikelihood


@pytest.mark.parametrize(
    "D,f_x,expected",
    [
        (
            np.asarray([[0, 1], [2, 0], [2, 3]]),
            np.asarray([1.5, 0.7, 2.1, 1.2], dtype=np.float32),
            np.asarray([0.8, 0.6, 0.9], dtype=np.float32),
        )
    ],
)
def test_ll_get_diffs(D, f_x, expected):
    ll = LogLikelihood()
    ll.register_data(D)
    npt.assert_almost_equal(ll._get_diffs(f_x).flatten(), expected.flatten())
