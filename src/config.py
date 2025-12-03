import yaml
from pathlib import Path

def load_config(path: str = "config.yaml"):
    with open(Path(path), "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
