import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.inspection import permutation_importance


def build_gp(n_dims: int) -> GaussianProcessRegressor:
    """
    ARD Matern GP with noise. n_dims is inferred from X.shape[1].
    """
    kernel = (
        C(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(n_dims),
            length_scale_bounds=(1e-2, 1e3),
            nu=2.5,
        )
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1))
    )

    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=42,
    )


def fit_gp(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    gp = build_gp(X.shape[1])
    gp.fit(X, y)
    return gp


def gp_feature_importance(
    gp: GaussianProcessRegressor,
    feature_names,
    n_repeats: int = 100,
    random_state: int = 0,
):
    """
    Permutation importance on the GP surrogate:
    "How much does prediction quality drop if we shuffle this feature?"

    Returns normalized importances that sum to 1.
    """
    X = gp.X_train_
    y = gp.y_train_

    r = permutation_importance(
        gp,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    imp = np.maximum(r.importances_mean, 0.0)
    if imp.sum() <= 0:
        # fallback
        n = len(feature_names)
        return {name: 1.0 / n for name in feature_names}

    imp = imp / imp.sum()
    return dict(zip(feature_names, imp))
