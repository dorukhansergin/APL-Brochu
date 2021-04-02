from typing import Tuple
import numpy as np
import scipy as sp
from sklearn.gaussian_process.kernels import Kernel


def gaussian_process_conditional(
    X1: np.ndarray, y1: np.ndarray, X2: np.ndarray, kernel: Kernel
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Copied from https://peterroelants.github.io/posts/gaussian-process-tutorial/
    which is a great tutorial if you are not familiar with GPs.

    Slightly modified to only return the diagonal of Σ2 because that's
    needed only.

    Parameters
    ----------
    X1 : np.ndarray
        Training input.
    y1 : np.ndarray
        Training targets.
    X2 : np.ndarray
        Test inputs for which the f(x) distribution to be predicted.
    kernel : Kernel
        Input covariance kernel.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Mean and covariance of the conditional
    """
    # Kernel of the observations
    Σ11 = kernel(X1, X1)
    # Kernel of observations vs to-predict
    Σ12 = kernel(X1, X2)
    # Solve
    solved = sp.linalg.solve(Σ11, Σ12, assume_a="pos").T
    # Compute posterior mean
    μ2 = solved @ y1
    # Compute the posterior covariance
    Σ22 = kernel(X2, X2)
    Σ2 = Σ22 - (solved @ Σ12)
    return μ2, np.diag(Σ2)