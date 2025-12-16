import numpy as np
from scipy.stats import norm


def expected_improvement(mu, sigma, best_y, xi: float = 0.01, maximize: bool = True):
    """
    EI for maximization by default.
    xi encourages exploration.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # avoid division by 0
    sigma = np.maximum(sigma, 1e-12)

    if maximize:
        improvement = mu - best_y - xi
    else:
        improvement = best_y - mu - xi

    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

    # if sigma ~ 0, EI should be ~0
    ei = np.where(sigma <= 1e-12, 0.0, ei)
    return np.maximum(ei, 0.0)
