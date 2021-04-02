import pytest
import numpy as np
from numpy.testing import assert_equal
from apl.utils import transfer_id_from_query_to_explored


@pytest.mark.parametrize(
    "i,queryable,explored,expected_queryable,expected_explored",
    [
        (
            5,
            np.asarray([1, 2, 5], dtype=np.int16),
            np.asarray([3, 4], dtype=np.int16),
            np.asarray([1, 2], dtype=np.int16),
            np.asarray([3, 4, 5], dtype=np.int16),
        ),
        (
            5,
            np.asarray([5], dtype=np.int16),
            np.asarray([], dtype=np.int16),
            np.asarray([], dtype=np.int16),
            np.asarray([5], dtype=np.int16),
        ),
    ],
)
def test_transfer_id_from_query_to_explored(
    i, queryable, explored, expected_queryable, expected_explored
):
    print(i, queryable, explored)
    queryable, explored = transfer_id_from_query_to_explored(
        i, queryable, explored
    )
    assert_equal(queryable, expected_queryable)
    assert_equal(explored, expected_explored)