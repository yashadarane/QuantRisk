import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import t  # Student-t for fat tails

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
RNG_SEED = 12345  # reproducibility

def run_fat_tailed_monte_carlo(mu, cov, days, sims, dof=5, is_log_returns=False, rng=None):
    """
    Monte Carlo with Student-t shocks correlated by covariance matrix.

    Parameters
    ----------
    mu : array_like, shape (n_assets,)
        Expected daily drift (mean) for each asset. If is_log_returns=True, mu should be daily log-returns.
    cov : array_like, shape (n_assets, n_assets)
        Covariance matrix of asset returns (daily).
    days : int
    sims : int
    dof : float
        Degrees of freedom for Student-t. High dof -> closer to Gaussian.
    is_log_returns : bool
        If True: treat simulated returns as log-returns and accumulate via exp(cumsum).
        If False: treat as arithmetic returns and accumulate via cumulative product of (1+rt).
    rng : np.random.Generator or None
        Random number generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = np.asarray(mu).reshape(-1)
    n_assets = len(mu)

    # Cholesky (fallback to SVD if not SPD)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        u, s, vh = np.linalg.svd(cov)
        # Construct a matrix whose product with its transpose equals cov (approx)
        L = (u * np.sqrt(s))  # shape (n_assets, n_assets)

    # 1) Generate uncorrelated Student-t samples (shape: days, sims, n_assets)
    # Use the RNG to allow reproducibility
    uncorrelated = t.rvs(dof, size=(days, sims, n_assets), random_state=rng)

    # 1b) Normalize variance to 1 (student-t variance = df/(df-2) for df>2)
    if dof > 2:
        scale = np.sqrt((dof - 2) / dof)
        uncorrelated = uncorrelated * scale

    # 2) Correlate using Cholesky. If cov = L @ L.T, then x = z @ L.T gives Cov(x)=L@L.T
    #    So we want to right-multiply by L.T
    correlated = np.matmul(uncorrelated, L.T)  # result shape: (days, sims, n_assets)

    # 3) Add drift
    # Broadcast mu across days and sims:
    returns = correlated + mu.reshape((1, 1, n_assets))

    # 4) Portfolio returns (equal weight by default)
    weights = np.ones(n_assets) / n_assets
    # returns @ weights => shape (days, sims)
    port_rets = np.matmul(returns, weights)

    # 5) Accumulate portfolio value
    if is_log_returns:
        # port_rets are log-returns: wealth = INITIAL_INV * exp(cumsum(log-returns))
        cum_rets = np.exp(np.cumsum(port_rets, axis=0))
        paths = INITIAL_INV * cum_rets
    else:
        # port_rets are arithmetic returns: wealth = INITIAL_INV * cumprod(1 + r)
        paths = INITIAL_INV * np.cumprod(1.0 + port_rets, axis=0)

    return paths

def main():
    print("--- Running ADVANCED (Fat-Tailed) Stress Tests ---")
    rng = np.random.default_rng(RNG_SEED)

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

    # IMPORTANT: Are the values in 'daily' log-returns or arithmetic returns?
    # Set this flag accordingly. If unsure, inspect typical magnitudes:
    # - if values like 0.01, -0.02 etc -> arithmetic returns
    # - if values like 0.0001 or -0.0002 or cumulative plotting used earlier as log -> might be log-returns
    RETURNS_ARE_LOG = False

    # ---------------------------------------------------------
    # SCENARIO 1: NORMAL BEAR (The "Gaussian" Assumption)
    # ---------------------------------------------------------
    print("1. Simulating Bear Regime (Standard Gaussian)...")
    bear_data = df[df['regime'] == bear_state][assets]

    mu_bear = bear_data.mean().values  # daily mean
    cov_bear = bear_data.cov().values  # daily covariance

    paths_normal = run_fat_tailed_monte_carlo(mu_bear, cov_bear, DAYS, SIMS, dof=100,
                                             is_log_returns=RETURNS_ARE_LOG, rng=rng)

    # ---------------------------------------------------------
    # SCENARIO 2: FAT-TAILED BEAR (The "Real World" Assumption)
    # ---------------------------------------------------------
    print("2. Simulating Bear Regime (Fat-Tailed Student-t)...")
    paths_fat = run_fat_tailed_monte_carlo(mu_bear, cov_bear, DAYS, SIMS, dof=4,
                                           is_log_returns=RETURNS_ARE_LOG, rng=rng)

    # ---------------------------------------------------------
    # SCENARIO 3: DOOMSDAY (Fat Tails + 2x Vol)
    # ---------------------------------------------------------
    print("3. Simulating Doomsday (Fat Tails + 2x Vol)...")
    cov_swan = cov_bear * 4.0  # variance *4 => volatility x2
    # Example: set a strong negative drift, here -30% annualized converted to daily (simple approx)
    mu_swan = np.full(mu_bear.shape, -0.30 / 252.0)

    paths_doom = run_fat_tailed_monte_carlo(mu_swan, cov_swan, DAYS, SIMS, dof=4,
                                            is_log_returns=RETURNS_ARE_LOG, rng=rng)

    # ---------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    days_range = np.arange(DAYS)

    # 1. Normal Bear (Orange)
    plt.plot(days_range, np.median(paths_normal, axis=1), color='orange',
             label='Gaussian Bear (P50)', linestyle='--')
    plt.fill_between(days_range,
                     np.percentile(paths_normal, 5, axis=1),
                     np.percentile(paths_normal, 95, axis=1),
                     color='orange', alpha=0.12, label='Gaussian 5-95%')

    # 2. Fat-Tailed Bear (Blue)
    plt.plot(days_range, np.median(paths_fat, axis=1), color='blue',
             label='Student-t (Fat-Tail) Bear (P50)')
    plt.fill_between(days_range,
                     np.percentile(paths_fat, 5, axis=1),
                     np.percentile(paths_fat, 95, axis=1),
                     color='blue', alpha=0.12, label='Fat-Tail 5-95%')

    # 3. Doomsday (Red)
    plt.plot(days_range, np.median(paths_doom, axis=1), color='red',
             label='Doomsday (P50)', linestyle=':')
    plt.fill_between(days_range,
                     np.percentile(paths_doom, 5, axis=1),
                     np.percentile(paths_doom, 95, axis=1),
                     color='red', alpha=0.12, label='Doomsday 5-95%')
    for i in range(200):     
        plt.plot(paths_fat[:, i], color="blue", alpha=0.02)

    plt.title(f"Gaussian vs. Fat-Tailed Stress Test (Next {DAYS} Trading Days)", fontsize=14)
    plt.ylabel("Portfolio Value (₹)")
    plt.xlabel("Trading Days")
    plt.axhline(INITIAL_INV, color='black', linestyle=':', label='Break Even')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    save_file = OUTPUT_PATH / "fat_tail_stress_test_fixed.png"
    plt.savefig(save_file, bbox_inches='tight', dpi=150)
    print(f"Comparison Chart saved to {save_file}")

if __name__ == "__main__":
    main()
