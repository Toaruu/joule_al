# src/doe.py

import numpy as np
import pandas as pd
from pathlib import Path

from .config import load_config


def latin_hypercube(n_samples: int, n_dim: int, random_state: int | None = None):
    """
    Basic Latin Hypercube Sampling in [0, 1]^d.
    Returns an (n_samples, n_dim) array.
    """
    rng = np.random.default_rng(random_state)
    cut = np.linspace(0, 1, n_samples + 1)

    H = np.zeros((n_samples, n_dim), dtype=float)
    for j in range(n_dim):
        u = rng.uniform(low=cut[:-1], high=cut[1:])
        rng.shuffle(u)
        H[:, j] = u

    return H


def scale_lhs_to_params(lhs: np.ndarray, params_cfg: list[dict]) -> np.ndarray:
    """
    Scale LHS samples from [0,1] to the parameter ranges given in config.
    """
    X = np.zeros_like(lhs)
    for j, p in enumerate(params_cfg):
        low, high = p["low"], p["high"]
        vals = low + lhs[:, j] * (high - low)

        if p.get("type", "float") == "int":
            vals = np.round(vals).astype(int)

        X[:, j] = vals

    return X


def generate_initial_doe(
    config_path: str = "config.yaml",
    output_path: str = "data/initial_doe.csv",
    n_samples: int | None = None,
    random_state: int = 42,
):
    """
    Generate an initial space-filling DOE using Latin Hypercube Sampling.

    This ONLY outputs parameter combinations (no target column yet).
    After the lab runs these experiments and measures capacity, you append
    the results into data/experiments.csv.
    """
    cfg = load_config(config_path)
    params_cfg = cfg["parameters"]
    param_names = [p["name"] for p in params_cfg]

    # If not specified, use config value or fallback to 3x number of dims
    if n_samples is None:
        n_samples = cfg.get("initial_doe_samples")
        if n_samples is None:
            n_dim = len(params_cfg)
            n_samples = 3 * n_dim

    lhs = latin_hypercube(n_samples, len(params_cfg), random_state=random_state)
    X = scale_lhs_to_params(lhs, params_cfg)

    df = pd.DataFrame(X, columns=param_names)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated {n_samples} DOE points and saved to {output_path}")


if __name__ == "__main__":
    generate_initial_doe()
