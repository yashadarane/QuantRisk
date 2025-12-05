# src/risk/regime_var_check.py
import pandas as pd
import numpy as np
from pathlib import Path

RETS_PATH = Path("data/processed/india_asset_returns_log.parquet")
REGIMES_PATH = Path("data/processed/hmm_regimes_india.parquet")

def main():
    rets = pd.read_parquet(RETS_PATH)
    regimes = pd.read_parquet(REGIMES_PATH)

    # select tickers for a small portfolio (first 5 with valid data)
    tickers = [c for c in rets.columns][:5]
    if not tickers:
        print("No tickers found")
        return

    w = np.repeat(1/len(tickers), len(tickers))
    df = rets[tickers].join(regimes['regime'], how='inner').dropna()

    alpha = 0.01  # 1% VaR
    rows = []
    for r in sorted(df['regime'].unique()):
        sub = df[df['regime'] == r][tickers]
        port = (sub * w).sum(axis=1)
        if port.empty:
            continue
        var = np.quantile(port, alpha)
        cvar = port[port <= var].mean()
        rows.append({"regime": r, "VaR_1%": var, "CVaR_1%": cvar, "n": len(port), "mean": port.mean(), "std": port.std()})
    out = pd.DataFrame(rows).set_index("regime")
    print(out)

if __name__ == "__main__":
    main()
