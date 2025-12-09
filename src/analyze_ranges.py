import pandas as pd

TARGET = "charge_capacity_mAh_g"   # first-cycle charge capacity

# Load experiments
df = pd.read_csv("data/experiments.csv")

print("Columns:", list(df.columns))

# Choose the top X% highest capacity samples
TOP_PERCENT = 0.20   # top 20% – you can change to 0.1, 0.3, etc.

df_sorted = df.sort_values(TARGET, ascending=False)
cutoff = max(1, int(len(df) * TOP_PERCENT))
top_df = df_sorted.head(cutoff)

print(f"\n=== Using top {TOP_PERCENT*100:.0f}% highest {TARGET} (n = {len(top_df)}) ===")
print(top_df[[TARGET]].describe())

print("\nSuggested new bounds (with 5% margin around top region):")
for col in ["P1", "N1", "t1", "P2", "N2", "t2", "P3", "N3", "t3"]:
    low = top_df[col].min()
    high = top_df[col].max()

    # add a small margin
    margin_low = low - 0.05 * (high - low)
    margin_high = high + 0.05 * (high - low)

    # ints stay ints
    if col.startswith("N"):
        margin_low = int(round(margin_low))
        margin_high = int(round(margin_high))
    print(f"{col}: low={margin_low}, high={margin_high}")
