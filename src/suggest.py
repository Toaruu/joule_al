import numpy as np
import pandas as pd
from pathlib import Path

from .config import load_config
from .data_io import load_experiments
from .model_gp import fit_gp
from .acquisition import expected_improvement

def sample_candidates(cfg):
    params = cfg["parameters"]
    n = cfg["candidate_pool_size"]
    X = np.zeros((n, len(params)))

    for j, p in enumerate(params):
        low, high = p["low"], p["high"]
        vals = np.random.uniform(low, high, size=n)
        if p["type"] == "int":
            vals = np.round(vals).astype(int)
        X[:, j] = vals

    return X

def suggest_new_experiments(config_path="config.yaml",
                            data_path="data/experiments.csv",
                            output_path="outputs/suggestions.csv"):

    cfg = load_config(config_path)
    param_names = [p["name"] for p in cfg["parameters"]]
    target_col = cfg["target_column"]

    # 1. Load data
    df, X, y = load_experiments(data_path, param_names, target_col)
    best_y = np.max(y)

    # 2. Fit GP
    gp = fit_gp(X, y)

    # 3. Sample candidate pool
    X_cand = sample_candidates(cfg)

    # 4. Predict on candidates
    mu, sigma = gp.predict(X_cand, return_std=True)

    # 5. Compute EI
    ei = expected_improvement(mu, sigma, best_y)

    # 6. Pick top-k
    k = cfg["suggestions_per_batch"]
    idx = np.argsort(ei)[::-1][:k]
    best_candidates = X_cand[idx, :]

    # 7. Save suggestions
    suggestions_df = pd.DataFrame(best_candidates, columns=param_names)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    suggestions_df.to_csv(output_path, index=False)

    print(f"Saved {k} suggestions to {output_path}")

if __name__ == "__main__":
    suggest_new_experiments()
