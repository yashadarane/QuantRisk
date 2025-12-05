import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

PROC = Path("data/processed")
PRICES_RET = PROC / "india_asset_returns_log.parquet"
MACRO = PROC / "macro_fred_india.parquet"
OUT = PROC / "features_hmm_india.parquet"

rets = pd.read_parquet(PRICES_RET)
macro = pd.read_parquet(MACRO)

# Align on month-end intersection
df = rets.join(macro, how="inner")

# Select macro features (example)
macro_feats = [c for c in macro.columns if ("CPI" in c.upper() or "Gsec" in c or "Repo" in c or "M3" in c or "GDP" in c.upper())]
# If not found, take all macro columns
if not macro_feats:
    macro_feats = list(macro.columns)

features = df[macro_feats].copy()
features = features.dropna(how="any")
# Replace any remaining inf values with NaN, then drop
features = features.replace([np.inf, -np.inf], np.nan)
features = features.dropna(how="any")

if features.empty:
    raise ValueError("No valid features after removing NaN/inf values")

scaler = StandardScaler()
X = scaler.fit_transform(features.values)
Xdf = pd.DataFrame(X, index=features.index, columns=features.columns)

Xdf.to_parquet(OUT)
print("Saved HMM features to", OUT)
