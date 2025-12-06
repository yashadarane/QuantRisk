from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import json

def forward_fill_monthly_regimes_to_daily(monthly_regimes: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    if "regime" not in monthly_regimes.columns:
        raise ValueError("monthly_regimes must contain a 'regime' column")

    # Ensure monthly index is sorted
    monthly = monthly_regimes.sort_index()
    # Reindex + forward fill
    daily = monthly.reindex(daily_index, method="ffill")
    return daily


import json

def exposure_from_posteriors(daily_regimes: pd.DataFrame, daily_returns: pd.Series, model_dir="models") -> pd.Series:
    # 1. Load HMM Mapping (The Strategic View)
    with open(Path(model_dir) / "state_mapping.json", "r") as f:
        mapping = json.load(f)
    
    bull_state = mapping["bull_state"]
    bear_state = mapping["bear_state"]
    
    # 2. Base Strategy from HMM
    bull_prob = daily_regimes[f"prob_state_{bull_state}"]
    bear_prob = daily_regimes[f"prob_state_{bear_state}"]
    
    exposure = pd.Series(index=daily_regimes.index, data=0.5) # Neutral start
    exposure[bull_prob > 0.6] = 1.0  # High conviction Bull
    exposure[bear_prob > 0.6] = 0.0  # High conviction Bear
    
    # ---------------------------------------------------------
    # 3. TACTICAL OVERLAY (The "Circuit Breaker")
    # ---------------------------------------------------------
    # Calculate realized volatility (21-day rolling standard deviation)
    # 21 days ~ 1 trading month
    rolling_vol = daily_returns.rolling(window=21).std() * np.sqrt(252)
    
    # VOLATILITY FILTER:
    # If volatility is extreme (> 30%), force exposure to 0 (Cash)
    # This protects you from fast crashes that the Monthly HMM misses.
    # 0.30 is a standard "Panic Threshold" for Nifty.
    
    exposure[rolling_vol > 0.30] = 0.0
    
    return exposure
def run_backtest(daily_returns: pd.Series,
                 exposure: pd.Series,
                 tc_per_trade: float = 0.0005) -> pd.DataFrame:
    if not daily_returns.index.equals(exposure.index):
        # try to align
        exposure = exposure.reindex(daily_returns.index, method='ffill').fillna(0)

    df = pd.DataFrame(index=daily_returns.index)
    df['asset_r'] = daily_returns
    df['exposure'] = exposure
    # transaction cost applied on abs change in exposure
    df['exposure_change'] = df['exposure'].diff().abs().fillna(0)
    df['tc'] = df['exposure_change'] * tc_per_trade
    # portfolio log-return: exposure * asset log-return - tc
    df['port_r'] = df['exposure'] * df['asset_r'] - df['tc']
    df['cum_log'] = df['port_r'].cumsum()
    df['cum_simple'] = np.exp(df['cum_log']) - 1

    return df


def performance_stats(port_r: pd.Series, ann_factor: int = 252) -> Dict[str, float]:
    # assume input is daily log-returns
    mean_daily = port_r.mean()
    vol_daily = port_r.std()
    ann_ret = mean_daily * ann_factor
    ann_vol = vol_daily * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    # max drawdown on cumulative simple returns
    cum = np.exp(port_r.cumsum())
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    return {
        'ann_return': float(ann_ret),
        'ann_vol': float(ann_vol),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd)
    }


def _make_synthetic_data() -> Tuple[pd.Series, pd.DataFrame]:
    # 30 days daily returns (log returns) with a simple trend
    dates = pd.date_range('2020-01-01', periods=30, freq='B')
    rng = np.random.RandomState(0)
    # 0.1% mean daily log-return, small noise
    daily_rets = pd.Series(rng.normal(0.001, 0.002, size=len(dates)), index=dates)

    # monthly regimes: 2 rows, covering calendar months corresponding to dates
    months = pd.DatetimeIndex([pd.Timestamp('2020-01-31'), pd.Timestamp('2020-02-29')])
    # define 2 regimes: regime 0 (bear) and regime 1 (bull)
    monthly = pd.DataFrame(index=months)
    monthly['regime'] = [0, 1]
    # create posterior prob columns
    monthly['prob_state_0'] = [0.9, 0.1]
    monthly['prob_state_1'] = [0.1, 0.9]

    return daily_rets, monthly


# Pytest-compatible test functions
def test_forward_fill_and_exposure():
    daily_rets, monthly = _make_synthetic_data()
    daily_regimes = forward_fill_monthly_regimes_to_daily(monthly, daily_rets.index)
    assert daily_regimes.shape[0] == len(daily_rets)
    # exposure should be equal to posterior of bull (column with higher mean is prob_state_1)
    exposure = exposure_from_posteriors(daily_regimes)
    assert exposure.max() <= 1.0 and exposure.min() >= 0.0
    # because second month is bull with 0.9, last days should show exposures near 0.9
    assert exposure.iloc[-1] > 0.5


def test_run_backtest_and_stats():
    daily_rets, monthly = _make_synthetic_data()
    daily_regimes = forward_fill_monthly_regimes_to_daily(monthly, daily_rets.index)
    exposure = exposure_from_posteriors(daily_regimes)
    df = run_backtest(daily_rets, exposure, tc_per_trade=0.001)
    # basic shape and columns
    for c in ['asset_r','exposure','tc','port_r','cum_log','cum_simple']:
        assert c in df.columns
    stats = performance_stats(df['port_r'])
    assert 'ann_return' in stats and 'sharpe' in stats
    # cum_simple should be numeric and finite
    assert np.isfinite(df['cum_simple'].iloc[-1])


if __name__ == '__main__':
    # Quick demo run (not a test). Use when running python regime_backtest.py
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic data')
    args = parser.parse_args()
    if args.demo:
        daily_rets, monthly = _make_synthetic_data()
        daily_regimes = forward_fill_monthly_regimes_to_daily(monthly, daily_rets.index)
        exposure = exposure_from_posteriors(daily_regimes)
        df = run_backtest(daily_rets, exposure, tc_per_trade=0.0005)
        print(df.head())
        print('Stats:', performance_stats(df['port_r']))
    else:
        print('This module provides functions for regime backtesting. Run with --demo for a quick example.')
