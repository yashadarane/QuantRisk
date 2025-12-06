import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path

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
SIMS = 5000  # Lower sims for speed in multiple scenarios

def run_monte_carlo(mu, cov, days, sims):
    """Core Monte Carlo Engine"""
    n_assets = len(mu)
    # Cholesky Decomposition
    try:
        L = np.linalg.cholesky(cov)
    except:
        u, s, vh = np.linalg.svd(cov)
        L = u @ np.diag(np.sqrt(s))
        
    # Generate random shocks
    uncorrelated = np.random.normal(0, 1, (days, sims, n_assets))
    correlated = np.einsum('dsa, ab -> dsb', uncorrelated, L)
    returns = mu + correlated
    
    # Portfolio Path (Equal Weights)
    weights = np.ones(n_assets) / n_assets
    port_rets = np.dot(returns, weights)
    cum_rets = np.exp(np.cumsum(port_rets, axis=0))
    paths = INITIAL_INV * cum_rets
    return paths

def main():
    print("--- Running Scenario Stress Tests ---")
    
    # Load Data
    daily = pd.read_parquet(DAILY_RETS)
    monthly = pd.read_parquet(REGIMES)
    
    # Identify Nifty (Asset)
    nifty_col = [c for c in daily.columns if "NIFTY" in c.upper() or "NSEI" in c.upper()][0]
    assets = [nifty_col] # For simplicity, we stick to Nifty. Add more columns to 'daily' for multi-asset.
    
    # Load Regime Mapping
    with open(MODEL_DIR / "state_mapping.json", "r") as f:
        mapping = json.load(f)
    bear_state = mapping["bear_state"]
    
    # Align Data
    # We need to map daily returns to regimes to get stats
    # (Simplified alignment for this script)
    regimes_daily = monthly.reindex(daily.index, method='ffill')
    df = daily[assets].join(regimes_daily['regime']).dropna()
    
    # ---------------------------------------------------------
    # SCENARIO 1: HISTORICAL CRASH (The "Bear Regime")
    # ---------------------------------------------------------
    print(f"1. Simulating Historical Bear Regime (State {bear_state})...")
    bear_data = df[df['regime'] == bear_state][assets]
    
    mu_bear = bear_data.mean().values
    cov_bear = bear_data.cov().values
    
    paths_bear = run_monte_carlo(mu_bear, cov_bear, DAYS, SIMS)
    
    # ---------------------------------------------------------
    # SCENARIO 2: BLACK SWAN (2x Volatility, -5% Mean Drift)
    # ---------------------------------------------------------
    print("2. Simulating Black Swan (2x Volatility)...")
    
    # MANUAL OVERRIDES (Stress Testing)
    # 1. Double the Volatility (Variance * 4 approx, or just Covariance * 4)
    # Actually, Covariance = Correlation * Vol_A * Vol_B. 
    # If we double Vol, we quadruple Covariance elements.
    cov_swan = cov_bear * 4.0 
    
    # 2. Force Negative Drift (Market crashing hard)
    # Assume -5% monthly drop (-0.05 / 21 days daily drift)
    mu_swan = np.full(mu_bear.shape, -0.05 / 21)
    
    paths_swan = run_monte_carlo(mu_swan, cov_swan, DAYS, SIMS)

    # ---------------------------------------------------------
    # SCENARIO 3: BREAK-CORRELATION (Correlation = 1.0)
    # ---------------------------------------------------------
    # Only useful if you have >1 asset. For 1 asset, this is same as Bear.
    # We'll skip for single asset, or just add a "Fed Pivot" (Bull) Scenario.
    print("3. Simulating Fed Pivot / Bull Regime...")
    bull_state = mapping["bull_state"]
    bull_data = df[df['regime'] == bull_state][assets]
    mu_bull = bull_data.mean().values
    cov_bull = bull_data.cov().values
    paths_bull = run_monte_carlo(mu_bull, cov_bull, DAYS, SIMS)

    # ---------------------------------------------------------
    # VISUALIZATION: THE COMPARISON
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # Plot P50 (Median) lines for each scenario
    # We plot the 'Cone of Uncertainty' for the Bear scenario
    
    # 1. Bull Scenario
    plt.plot(np.median(paths_bull, axis=1), color='green', label='Scenario: Bull Market (P50)', linewidth=2)
    
    # 2. Bear Scenario (Historical)
    plt.plot(np.median(paths_bear, axis=1), color='orange', label='Scenario: Historical Crash (P50)', linewidth=2)
    # Add confidence interval shading for Bear
    plt.fill_between(range(DAYS), 
                     np.percentile(paths_bear, 5, axis=1), 
                     np.percentile(paths_bear, 95, axis=1), 
                     color='orange', alpha=0.1, label='Bear 95% CI')

    # 3. Black Swan Scenario
    plt.plot(np.median(paths_swan, axis=1), color='red', label='Scenario: Black Swan (2x Vol) (P50)', linewidth=2, linestyle='--')
    # Add confidence interval shading for Swan
    plt.fill_between(range(DAYS), 
                     np.percentile(paths_swan, 5, axis=1), 
                     np.percentile(paths_swan, 95, axis=1), 
                     color='red', alpha=0.1, label='Black Swan 95% CI')

    plt.title(f"Stress Testing: Projected Portfolio Value (Next {DAYS} Days)", fontsize=14)
    plt.ylabel("Portfolio Value (₹)")
    plt.xlabel("Trading Days")
    plt.axhline(INITIAL_INV, color='black', linestyle=':', label='Break Even')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_file = OUTPUT_PATH / "scenario_stress_test.png"
    plt.savefig(save_file)
    print(f"Comparison Chart saved to {save_file}")

if __name__ == "__main__":
    main()