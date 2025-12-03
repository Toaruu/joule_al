import numpy as np
from scipy.stats import norm

def expected_improvement(mu, sigma, best_y, xi=0.01):
    """EI for maximization."""
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - best_y - xi
    z = improvement / sigma

    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma < 1e-9] = 0.0
    return ei
