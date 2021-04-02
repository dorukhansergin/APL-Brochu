from typing import Tuple
import numpy as np


def transfer_id_from_query_to_explored(
    i: int, query_item_idx: np.ndarray, explored_item_idx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Transfers an idx from the array of queryables to the set of

    Parameters
    ----------
    i : int
        The index to query
    query_item_idx : np.ndarray
        The array of queryable item indices
    explored_item_idx : np.ndarray
        The array of already explored item indices

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The updated versions of queryable and explored indices, respectively
    """
    return (
        np.delete(query_item_idx, np.argwhere(query_item_idx == i)),
        np.append(explored_item_idx, i),
    )