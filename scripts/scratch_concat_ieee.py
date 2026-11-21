import pandas as pd
from pathlib import Path

folder = Path("data/ieee_cis")

csv_files = folder.glob("*.csv")

df = pd.concat(
    [pd.read_csv(file) for file in csv_files],
    ignore_index=True
)

print(df.shape)
print(df.head())