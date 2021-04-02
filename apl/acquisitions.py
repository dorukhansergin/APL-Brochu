from typing import Any
import scipy as sp
import numpy as np
from numpy.typing import ArrayLike


class Acquisition:
    def __call__(self, *args: Any, **kwds: Any) -> ArrayLike:
        raise NotImplementedError


class ExpectedImprovement(Acquisition):
    def __init__(self, xi: float) -> None:
        """
        xi: larger values will make acquisition
        favor exploration more over exploitation.
        """
        self.ξ = xi

    def __call__(
        self, μ: ArrayLike, σ2: ArrayLike, **kwargs: Any
    ) -> ArrayLike:
        """Computes expected improvement for each item,
        given the mean and variance estimates.
        """
        μ_max = kwargs.pop("mu_max")
        σ = np.sqrt(σ2, out=np.zeros_like(σ2), where=np.not_equal(σ2, 0))
        δ = np.subtract(μ, μ_max + self.ξ)
        z = np.divide(
            δ,
            σ,
            out=np.zeros_like(δ),
            where=np.not_equal(σ, 0),
        )
        return δ * sp.special.ndtr(z) + σ * sp.stats.norm().pdf(z)


class UpperConfidenceBound(Acquisition):
    def __init__(self, kappa: float) -> None:
        """kappa: larger values will make acquisition
        favor exploration more over exploitation.
        """
        self.κ = kappa

    def __call__(
        self, μ: ArrayLike, σ2: ArrayLike, **kwargs: Any
    ) -> ArrayLike:
        """Computes upper confidence bound for each item,
        given the mean and variance estimates.
        """
        σ = np.sqrt(σ2, out=np.zeros_like(σ2), where=np.not_equal(σ2, 0))
        return μ + σ * self.κ