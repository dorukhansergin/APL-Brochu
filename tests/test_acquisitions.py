import pytest
import numpy as np
import numpy.testing as npt
from apl.acquisitions import UpperConfidenceBound, ExpectedImprovement


@pytest.mark.parametrize(
    "kappa,mu,s2,expected",
    [
        (
            1.0,
            np.asarray([1.5, 0.7, 2.1], dtype=np.float32),
            np.asarray([1.0, 0.0, 4.0], dtype=np.float32),
            np.asarray([2.5, 0.7, 4.1], dtype=np.float32),
        )
    ],
)
def test_ucb_output(kappa, mu, s2, expected):
    ucb = UpperConfidenceBound(kappa=kappa)
    npt.assert_almost_equal(ucb(mu, s2), expected)
