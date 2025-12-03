import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

def build_gp():
    # Matern(ν=2.5) with ARD (length_scale per dim), plus noise term
    kernel = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(9),
        length_scale_bounds=(1e-2, 1e3),
        nu=2.5
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=42,
    )
    return gp

def fit_gp(X, y):
    gp = build_gp()
    gp.fit(X, y)
    return gp
