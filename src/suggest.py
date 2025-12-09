# src/suggest.py

import numpy as np
import pandas as pd
from pathlib import Path

from .config import load_config
from .data_io import load_experiments
from .model_gp import fit_gp
from .model_xgb import fit_xgb_ensemble, predict_xgb_ensemble
from .acquisition import expected_improvement


def sample_candidates(cfg):
    params = cfg["parameters"]
    n = cfg["candidate_pool_size"]
    X = np.zeros((n, len(params)))

    for j, p in enumerate(params):
        low, high = p["low"], p["high"]

        # Generate random samples first
        vals = np.random.uniform(low, high, size=n)

        # Apply resolution rules
        if p["name"].startswith("P"):
            # Power parameters → increments of 0.5
            vals = np.round(vals / 0.5) * 0.5

        elif p["name"].startswith("t"):
            # Time parameters → 1 decimal place
            vals = np.round(vals, 1)

        elif p.get("type") == "int":
            vals = np.round(vals).astype(int)

        # Store into design matrix
        X[:, j] = vals

    return X


def is_safe(row: pd.Series) -> bool:
    # Example safety rule: P3 >= P2 >= P1
    try:
        return (row["P3"] >= row["P2"] >= row["P1"])
    except KeyError:
        return True


def suggest_new_experiments(
    config_path: str = "src/config.yaml",
    data_path: str = "data/experiments.csv",
    output_path: str = "outputs/suggestions.csv",
):
    cfg = load_config(config_path)
    params_cfg = cfg["parameters"]
    param_names = [p["name"] for p in params_cfg]
    target_col = cfg["target_column"]
    model_type = cfg.get("model", {}).get("type", "gp")

    # 1. Load historical data
    df, X, y = load_experiments(data_path, param_names, target_col)
    best_y = np.max(y)

    # 2. Fit surrogate model
    if model_type == "gp":
        surrogate = fit_gp(X, y)
    elif model_type == "xgb_ensemble":
        surrogate = fit_xgb_ensemble(X, y)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 3. Sample candidate pool
    X_cand = sample_candidates(cfg)
    cand_df = pd.DataFrame(X_cand, columns=param_names)

    # 4. Apply safety constraints
    safe_mask = cand_df.apply(is_safe, axis=1)
    safe_df = cand_df[safe_mask]
    if safe_df.empty:
        raise RuntimeError("All candidate experiments filtered out by safety rules!")

    X_safe = safe_df[param_names].values

    # 5. Predict mean + uncertainty
    if model_type == "gp":
        mu, sigma = surrogate.predict(X_safe, return_std=True)
    elif model_type == "xgb_ensemble":
        mu, sigma = predict_xgb_ensemble(surrogate, X_safe)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 6. Compute EI
    ei = expected_improvement(mu, sigma, best_y)

    # 7. Select top-k batch
    k = cfg["suggestions_per_batch"]
    if k > len(ei):
        k = len(ei)
    idx = np.argsort(ei)[::-1][:k]
    best_candidates = X_safe[idx, :]

    # Build output DataFrame
    suggestions_df = pd.DataFrame(best_candidates, columns=param_names)

    # Add sample/expt number
    suggestions_df.insert(0, "sample_id", [f"S{i+1:03d}" for i in range(k)])

    # Add predictions + EI
    suggestions_df["predicted_capacity"] = mu[idx]
    suggestions_df["predicted_uncertainty"] = sigma[idx]
    suggestions_df["ei"] = ei[idx]

    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    suggestions_df.to_csv(output_path, index=False)

    # ---- Console summary ----
    print(f"[{model_type}] Current best *observed* capacity: {best_y:.2f} mAh/g\n")

    print("New suggested experiments (sorted by EI):")
    for row in suggestions_df.itertuples(index=False):
        print(
            f"  {row.sample_id}: "
            f"(P1={row.P1:.3f}, N1={row.N1:.0f}, t1={row.t1:.3f}) | "
            f"(P2={row.P2:.3f}, N2={row.N2:.0f}, t2={row.t2:.3f}) | "
            f"(P3={row.P3:.3f}, N3={row.N3:.0f}, t3={row.t3:.3f}) | "
            f"Pred cap={row.predicted_capacity:.2f} mAh/g, "
            f"Unc={row.predicted_uncertainty:.2f}, EI={row.ei:.3f}"
        )

    print(f"\n[{model_type}] Saved {k} suggested (safe) experiments to {output_path}")

if __name__ == "__main__":
    suggest_new_experiments()
