from typing import Tuple, Any
import numpy as np
from scipy.stats import multivariate_normal
import scipy as sp

ROOT_TWO = np.sqrt(2)


class LogLikelihood:
    def __init__(self) -> None:
        self.D = None

    def __call__(self, f_x: np.ndarray) -> Any:
        raise NotImplementedError

    def register_data(self, D: np.ndarray) -> None:
        self.D = D

    def _get_diffs(self, f_x: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(
            lambda pair: -np.diff(f_x.flatten()[pair]),
            arr=self.D,
            axis=1,
        )


class ProbitDataGivenF(LogLikelihood):
    def __init__(self, sigma_err: float) -> None:
        super().__init__()
        self.σ_err_ = sigma_err

    def __call__(self, f_x: np.ndarray) -> Any:
        return np.sum(
            sp.special.log_ndtr(
                self._get_diffs(f_x) / (ROOT_TWO * self.σ_err_)
            )
        )


def laplace_approximation(
    μ: np.ndarray, Σ: np.ndarray, loglikelihood: LogLikelihood
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate the posterior distribution using Laplace's approximation.
    The posterior is the simple summation of logpdf of the prior which
    is a multivariate normal, and the log likelihood of data given the
    prior. Since scipy's optimization module handles minimization problems
    the posterior is negated before fed into that function.

    Parameters
    ----------
    μ: np.ndarray
        The mean of the prior multivariate distribution

    Σ: np.ndarray
        The covariance matrix of the prior multivariate distribution

    loglikelihood : LogLikelihood
        A callable instance of the log-likelihood class chosen by the user.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The Maximum A Posteriori values of the mean and covariance.
    """
    log_pf = multivariate_normal(mean=μ, cov=Σ).logpdf

    neg_posterior = lambda f_x: -(log_pf(f_x) + loglikelihood(f_x))
    neg_posterior_opt_res = sp.optimize.minimize(
        neg_posterior, μ, method="BFGS"
    )
    μ_map, Σ_map = neg_posterior_opt_res.x, neg_posterior_opt_res.hess_inv
    return μ_map, Σ_map
