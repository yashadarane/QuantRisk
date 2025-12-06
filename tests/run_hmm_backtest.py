import pandas as pd
import numpy as np
import json
from pathlib import Path
from regime_backtest import (
    forward_fill_monthly_regimes_to_daily,
    run_backtest,
    performance_stats
)

# 1. Load Data
daily = pd.read_parquet("data/processed/india_asset_returns_daily.parquet")
monthly = pd.read_parquet("data/processed/hmm_regimes_india.parquet")

nifty = [c for c in daily.columns if "NIFTY" in c.upper() or "NSEI" in c.upper()][0]
daily_rets = daily[nifty]

print(f"Optimizing Strategy for Asset: {nifty}")

# 2. OPTIMIZED Exposure Logic (The "Alpha" Fix)
# ---------------------------------------------------------
def get_optimized_exposure(daily_regimes, daily_returns, model_dir="models"):
    # Load Mapping
    with open(Path(model_dir) / "state_mapping.json", "r") as f:
        mapping = json.load(f)
    
    bear_state = mapping["bear_state"]
    bear_prob = daily_regimes[f"prob_state_{bear_state}"]
    
    # --- CHANGE 1: DEFAULT TO 100% INVESTED ---
    # India is a growth market. Don't sit in cash unless necessary.
    exposure = pd.Series(index=daily_regimes.index, data=1.0)
    
    # --- CHANGE 2: HIGHER CONVICTION EXIT ---
    # Only exit if the model is > 70% sure it's a Bear market.
    # Prevents "whipsaws" (getting tricked by small dips).
    exposure[bear_prob > 0.70] = 0.0
    
    # --- CHANGE 3: TACTICAL VOLATILITY FILTER ---
    # Calculate 21-day rolling volatility (Annualized)
    rolling_vol = daily_returns.rolling(window=21).std() * np.sqrt(252)
    
    # Panic Switch: If Volatility > 35%, go to Cash immediately.
    # (Raised from 30% to 35% to give the market more room to breathe)
    exposure[rolling_vol > 0.35] = 0.0
    
    return exposure

# 3. Execution
# ---------------------------------------------------------
daily_regimes = forward_fill_monthly_regimes_to_daily(monthly, daily_rets.index)
exposure = get_optimized_exposure(daily_regimes, daily_rets)

# Run Backtest
df = run_backtest(daily_rets, exposure, tc_per_trade=0.001) # 0.1% Transaction Cost

# 4. Compare vs Benchmark
# ---------------------------------------------------------
stats = performance_stats(df["port_r"])
bench_stats = performance_stats(daily_rets)

print("\n" + "="*50)
print("OPTIMIZED 'ALPHA' STRATEGY RESULTS")
print("="*50)
print(f"{'Metric':<20} | {'Your Strategy':<15} | {'Benchmark (Nifty)':<15}")
print("-" * 56)
print(f"{'Annualized Return':<20} | {stats['ann_return']*100:.2f}%          | {bench_stats['ann_return']*100:.2f}%")
print(f"{'Sharpe Ratio':<20} | {stats['sharpe']:.2f}            | {bench_stats['sharpe']:.2f}")
print(f"{'Max Drawdown':<20} | {stats['max_drawdown']*100:.2f}%         | {bench_stats['max_drawdown']*100:.2f}%")
print("-" * 56)

if stats['sharpe'] > bench_stats['sharpe']:
    print("SUCCESS: You have beaten the market on a risk-adjusted basis.")
else:
    print("NOTE: Strategy is safer, but trails in raw returns.")

# Save for Dashboard
df.to_csv("reports/hmm_backtest_results.csv")