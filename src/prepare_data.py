# src/prepare_data.py

import pandas as pd
from pathlib import Path

def convert_master_to_experiments(
    input_path: str = "data/JP_P2_MasterData.xlsx",
    output_path: str = "data/experiments.csv",
):
    df = pd.read_excel(input_path)

    col_map = {
        "P1": "S1 power",
        "N1": "S1 pulse amount",
        "t1": "S1 pulse duration (second)",
        "P2": "S2 power",
        "N2": "S2 pulse amount",
        "t2": "S2 pulse duration (second)",
        "P3": "S3 power",
        "N3": "S3 pulse amount",
        "t3": "S3 pulse duration (second)",
        "charge_capacity_mAh_g": "Charge capacity (mAh/g)",
        "discharge_capacity_mAh_g": "Discharge capacity (mAh/g)",
    }

    # Select and rename
    out = df[list(col_map.values())].copy()
    out.columns = list(col_map.keys())

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Saved {len(out)} experiments to {output_path}")

if __name__ == "__main__":
    convert_master_to_experiments()
