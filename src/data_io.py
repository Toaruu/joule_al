from pathlib import Path
import pandas as pd

def load_experiments(csv_path: str, parameter_names, target_column):
    df = pd.read_csv(Path(csv_path))
    missing = [c for c in parameter_names + [target_column] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    X = df[parameter_names].values
    y = df[target_column].values
    return df, X, y
