# src/model_xgb.py

import numpy as np
from xgboost import XGBRegressor

def fit_xgb_ensemble(X, y, n_models: int = 5, base_seed: int = 123):
    """
    Fit an ensemble of XGBoost regressors.
    Use the ensemble mean as prediction and std across models as an uncertainty proxy.
    """
    models = []
    for i in range(n_models):
        seed = base_seed + i
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            tree_method="hist",
        )
        model.fit(X, y)
        models.append(model)
    return models

def predict_xgb_ensemble(models, X):
    """
    Predict mean and std over the ensemble.
    Returns (mu, sigma).
    """
    preds = np.stack([m.predict(X) for m in models], axis=0)  # (n_models, n_samples)
    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0)
    return mu, sigma
