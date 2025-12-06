import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import t  # <--- NEW LIBRARY FOR FAT TAILS

# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
DAILY_RETS = Path("data/processed/india_asset_returns_daily.parquet")
REGIMES = Path("data/processed/hmm_regimes_india.parquet")
MODEL_DIR = Path("models")
OUTPUT_PATH = Path("reports/figures")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Simulation Settings
INITIAL_INV = 100_000
DAYS = 30
SIMS = 5000

def run_fat_tailed_monte_carlo(mu, cov, days, sims, dof=5):
    """
    Advanced Monte Carlo: Uses Student-t Distribution for 'Fat Tails'
    dof (Degrees of Freedom):
      - 3 to 5 = Heavy Tails (Extreme Crashes possible)
      - 30+ = Normal Distribution (Standard)
    """
    n_assets = len(mu)
    
    # Cholesky Decomposition
    try:
        L = np.linalg.cholesky(cov)
    except:
        u, s, vh = np.linalg.svd(cov)
        L = u @ np.diag(np.sqrt(s))
        
    # 1. Generate UNCORRELATED Student-t Shocks
    # t.rvs(df, size) generates fat-tailed random numbers
    uncorrelated = t.rvs(dof, size=(days, sims, n_assets))
    
    # Normalize variance: Student-t has variance > 1, so we scale it back
    # Variance of t-dist is dof / (dof - 2) for dof > 2
    if dof > 2:
        scale = np.sqrt((dof - 2) / dof)
        uncorrelated = uncorrelated * scale

    # 2. Correlate them using Cholesky
    correlated = np.einsum('dsa, ab -> dsb', uncorrelated, L)
    
    # 3. Add Drift
    returns = mu + correlated
    
    # 4. Portfolio Path
    weights = np.ones(n_assets) / n_assets
    port_rets = np.dot(returns, weights)
    
    # Accumulate
    cum_rets = np.exp(np.cumsum(port_rets, axis=0))
    paths = INITIAL_INV * cum_rets
    return paths

def main():
    print("--- Running ADVANCED (Fat-Tailed) Stress Tests ---")
    
    # Load Data
    daily = pd.read_parquet(DAILY_RETS)
    monthly = pd.read_parquet(REGIMES)
    
    nifty_col = [c for c in daily.columns if "NIFTY" in c.upper() or "NSEI" in c.upper()][0]
    assets = [nifty_col] 
    
    with open(MODEL_DIR / "state_mapping.json", "r") as f:
        mapping = json.load(f)
    
    bear_state = mapping["bear_state"]
    bull_state = mapping["bull_state"]
    
    # Align Data
    regimes_daily = monthly.reindex(daily.index, method='ffill')
    df = daily[assets].join(regimes_daily['regime']).dropna()
    
    # ---------------------------------------------------------
    # SCENARIO 1: NORMAL BEAR (The "Gaussian" Assumption)
    # ---------------------------------------------------------
    print(f"1. Simulating Bear Regime (Standard Gaussian)...")
    bear_data = df[df['regime'] == bear_state][assets]
    mu_bear = bear_data.mean().values
    cov_bear = bear_data.cov().values
    
    # DoF = 100 effectively behaves like a Normal Distribution
    paths_normal = run_fat_tailed_monte_carlo(mu_bear, cov_bear, DAYS, SIMS, dof=100)
    
    # ---------------------------------------------------------
    # SCENARIO 2: FAT-TAILED BEAR (The "Real World" Assumption)
    # ---------------------------------------------------------
    print("2. Simulating Bear Regime (Fat-Tailed Student-t)...")
    
    # DoF = 4 is standard for financial returns (High Kurtosis)
    paths_fat = run_fat_tailed_monte_carlo(mu_bear, cov_bear, DAYS, SIMS, dof=4)

    # ---------------------------------------------------------
    # SCENARIO 3: DOOMSDAY (Fat Tails + 2x Vol)
    # ---------------------------------------------------------
    print("3. Simulating Doomsday (Fat Tails + 2x Vol)...")
    cov_swan = cov_bear * 4.0 # Double Volatility
    mu_swan = np.full(mu_bear.shape, -0.30 / 252) # -30% Annual Drift
    
    paths_doom = run_fat_tailed_monte_carlo(mu_swan, cov_swan, DAYS, SIMS, dof=4)

    # ---------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    days_range = range(DAYS)
    
    # 1. Normal Bear (Orange)
    plt.plot(np.median(paths_normal, axis=1), color='orange', label='Gaussian Bear (P50)', linestyle='--')
    plt.fill_between(days_range, 
                     np.percentile(paths_normal, 5, axis=1), 
                     np.percentile(paths_normal, 95, axis=1), 
                     color='orange', alpha=0.1, label='Gaussian 95% CI')

    # 2. Fat-Tailed Bear (Blue) - This is the UPGRADE
    plt.plot(np.median(paths_fat, axis=1), color='blue', label='Student-t (Fat-Tail) Bear (P50)')
    plt.fill_between(days_range, 
                     np.percentile(paths_fat, 5, axis=1), 
                     np.percentile(paths_fat, 95, axis=1), 
                     color='blue', alpha=0.1, label='Fat-Tail 95% CI')
    
    # 3. Doomsday (Red)
    plt.plot(np.median(paths_doom, axis=1), color='red', label='Doomsday (P50)', linestyle=':')
    plt.fill_between(days_range, 
                     np.percentile(paths_doom, 5, axis=1), 
                     np.percentile(paths_doom, 95, axis=1), 
                     color='red', alpha=0.1, label='Doomsday 95% CI')

    plt.title(f"Gaussian vs. Fat-Tailed Stress Test (Next {DAYS} Days)", fontsize=14)
    plt.ylabel("Portfolio Value (₹)")
    plt.xlabel("Trading Days")
    plt.axhline(INITIAL_INV, color='black', linestyle=':', label='Break Even')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    save_file = OUTPUT_PATH / "fat_tail_stress_test.png"
    plt.savefig(save_file)
    print(f"Comparison Chart saved to {save_file}")

if __name__ == "__main__":
    main()