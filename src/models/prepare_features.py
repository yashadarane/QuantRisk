import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Paths
PRICES_RET = Path("data/processed/india_asset_returns_monthly.parquet")
MACRO = Path("data/processed/macro_fred_india.parquet")
OUT = Path("data/processed/features_hmm_india.parquet")

def build_features():
    print("--- Building Features for HMM ---")
    
    # 1. Load Data
    if not PRICES_RET.exists() or not MACRO.exists():
        print("Error: Input files not found. Run previous data fetch scripts.")
        return

    m_ret = pd.read_parquet(PRICES_RET)
    macro = pd.read_parquet(MACRO)
    
    # Identify NIFTY column (First column matching 'NSEI' or 'NIFTY')
    nifty_candidates = [c for c in m_ret.columns if "NSEI" in c or "NIFTY" in c]
    if not nifty_candidates:
        print("Error: No NIFTY/NSEI column found in monthly returns.")
        return
    nifty_col = nifty_candidates[0]
    print(f"Using {nifty_col} as market proxy.")

    # 2. Engineer Market Features
    df_feat = pd.DataFrame(index=m_ret.index)
    
    # Volatility (3-month rolling std dev)
    df_feat["Market_Vol"] = m_ret[nifty_col].rolling(window=3).std()
    
    # Momentum (6-month rolling mean)
    df_feat["Market_Mom"] = m_ret[nifty_col].rolling(window=6).mean()
    
    # 3. Engineer Macro Features
    # FIX 1: Handle pct_change deprecation warning
    # We forward fill NaNs first so pct_change has clean data
    macro_filled = macro.ffill()
    
    # Calculate YoY Growth (12-month percent change)
    # fill_method=None explicitly avoids the deprecated behavior
    macro_growth = macro_filled.pct_change(12, fill_method=None)
    
    # FIX 2: Handle Division by Zero (Infinity)
    # Replace inf with NaN immediately
    macro_growth = macro_growth.replace([np.inf, -np.inf], np.nan)

    # LAG MACRO DATA (Shift forward 1 month to prevent look-ahead bias)
    macro_shifted = macro_growth.shift(1)
    
    # Select specific macro columns
    # We look for keywords like CPI (Inflation) and Gsec/Yields (Interest Rates)
    for c in macro_shifted.columns:
        if "CPI" in c: 
            df_feat["Inflation_YoY"] = macro_shifted[c]
        if "Gsec" in c or "Yield" in c: 
            df_feat["Rates_YoY"] = macro_shifted[c]
        if "GDP" in c:
            df_feat["GDP_Growth"] = macro_shifted[c]

    # 4. Clean and Scale
    # Drop rows where we don't have enough history (e.g., first 12 months)
    df_final = df_feat.dropna()
    
    # Sanity Check for Infinity again just in case
    if np.isinf(df_final.values).any():
        print("Warning: Infinity values found. Replacing with 0.")
        df_final = df_final.replace([np.inf, -np.inf], 0)

    print(f"Final Data Shape before scaling: {df_final.shape}")
    
    if df_final.empty:
        print("Error: DataFrame is empty after dropping NaNs. Check your rolling windows or data alignment.")
        return

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df_final.values)
    
    Xdf = pd.DataFrame(X, index=df_final.index, columns=df_final.columns)
    
    # Ensure directory exists
    OUT.parent.mkdir(parents=True, exist_ok=True)
    Xdf.to_parquet(OUT)
    print(f"Features saved to {OUT}")
    print(f"Columns: {Xdf.columns.tolist()}")

if __name__ == "__main__":
    build_features()