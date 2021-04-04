from typing import Tuple

import numpy as np
from numpy.random import default_rng
from sklearn.gaussian_process.kernels import Kernel

from .posterior_approximation import LogLikelihood, laplace_approximation
from .acquisitions import Acquisition
from .gaussian_process import gaussian_process_conditional
from .utils import transfer_id_from_query_to_explored


class ActivePreferenceLearning:
    def __init__(
        self,
        kernel: Kernel,
        loglikelihood: LogLikelihood,
        acquisition: Acquisition,
        random_state: int = 0,
    ):
        self.kernel = kernel
        self.loglikelihood = loglikelihood
        self.acquisition = acquisition
        self.rng = default_rng(random_state)

    def query(
        self,
        X: np.ndarray,
        explored_item_idx: np.ndarray,
        query_item_idx: np.ndarray,
        mu=None,
        pair_selections=None,
    ) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
        if (
            len(explored_item_idx) == 0
        ):  # first query, just pick two randomly from the query set
            return self.first_query(
                query_item_idx=query_item_idx,
                explored_item_idx=explored_item_idx,
            )
        else:
            return self.subsequent_query(
                X, explored_item_idx, query_item_idx, mu, pair_selections
            )

    def first_query(
        self, query_item_idx: np.ndarray, explored_item_idx: np.ndarray
    ) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
        opt1_idx, opt2_idx = self.rng.choice(query_item_idx, size=2)
        for idx in (opt1_idx, opt2_idx):
            (
                query_item_idx,
                explored_item_idx,
            ) = transfer_id_from_query_to_explored(
                idx, query_item_idx, explored_item_idx
            )
        return (
            opt1_idx,
            opt2_idx,
            explored_item_idx,
            query_item_idx,
            np.zeros(len(explored_item_idx)),
        )

    def subsequent_query(
        self,
        X: np.ndarray,
        explored_item_idx: np.ndarray,
        query_item_idx: np.ndarray,
        mu=None,
        pair_selections=None,
    ) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
        X_train = X[explored_item_idx]
        cov = self.kernel(X_train)
        self.loglikelihood.register_data(pair_selections)
        mu_map, _ = laplace_approximation(mu, cov, self.loglikelihood)

        mu_query, s2_query = gaussian_process_conditional(
            X_train, mu_map, X[query_item_idx], self.kernel
        )
        acquisitions_on_query_set = self.acquisition(
            mu_query, s2_query, **{"mu_max": mu_map.max()}
        )

        opt1_idx = explored_item_idx[pair_selections[-1, 0].item()]
        opt2_idx = query_item_idx[np.argmax(acquisitions_on_query_set).item()]
        (
            query_item_idx,
            explored_item_idx,
        ) = transfer_id_from_query_to_explored(
            opt2_idx, query_item_idx, explored_item_idx
        )
        x = X[opt2_idx].reshape(1, -1)
        k_star = self.kernel(X_train, x)
        mu_x = k_star.T @ np.linalg.solve(cov, mu_map)

        return (
            opt1_idx,
            opt2_idx,
            explored_item_idx,
            query_item_idx,
            np.append(mu_map, mu_x),
        )
