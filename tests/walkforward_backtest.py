
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import json
import cvxpy as cp

def solve_cvar_local(scenarios, alpha=0.01, max_w=1.0, min_w=0.0):
    N, n = scenarios.shape
    w = cp.Variable(n)
    v = cp.Variable()
    z = cp.Variable(N)
    losses = - scenarios @ w
    constraints = [z >= 0, z >= losses - v, cp.sum(w) == 1, w >= min_w, w <= max_w]
    obj = cp.Minimize(v + (1.0 / (alpha * N)) * cp.sum(z))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver="ECOS", verbose=False)
    return np.array(w.value, dtype=float)

def build_scenarios_from_history(returns_df, train_start, train_end, horizon, n_scen):
    # returns_df: DataFrame indexed by period end (monthly), columns assets
    idx = returns_df.loc[train_start:train_end].index
    possible_starts = []
    for i in range(len(idx)):
        if i + horizon <= len(idx) - 1:
            possible_starts.append(idx[i])
    if not possible_starts:
        raise RuntimeError("No possible start positions for scenario generation")
    scenarios = []
    for _ in range(n_scen):
        s = np.random.choice(possible_starts)
        start_pos = returns_df.index.get_loc(s)
        block = returns_df.iloc[start_pos:start_pos+horizon].values
        cum = np.prod(1 + block, axis=0) - 1
        scenarios.append(cum)
    return np.vstack(scenarios)

def main(args):
    RETURNS = Path(args.returns)
    if not RETURNS.exists():
        raise FileNotFoundError(RETURNS)
    df = pd.read_parquet(RETURNS)
    df = df.sort_index()
    assets = df.columns.tolist()

    results = []
    rebalance_dates = pd.date_range(start=args.start_date or df.index[0] + pd.DateOffset(months=args.train_window),
                                    end=args.end_date or df.index[-1] - pd.DateOffset(months=args.horizon),
                                    freq=pd.DateOffset(months=args.rebalance_freq))

    for rd in rebalance_dates:
        train_start = rd - pd.DateOffset(months=args.train_window)
        train_end = rd
        test_start = rd
        test_end = rd + pd.DateOffset(months=args.horizon)
        train = df.loc[train_start:train_end]
        test = df.loc[test_start:test_end]
        if train.shape[0] < args.train_window or test.shape[0] < args.horizon:
            continue
        # build scenarios
        scenarios = build_scenarios_from_history(df, train_start, train_end, args.horizon, args.n_scen)
        # solve cvar
        w = solve_cvar_local(scenarios, alpha=args.alpha, max_w=args.max_weight, min_w=args.min_weight)
        # compute realized multi-month return for the test period per asset
        realized = np.prod(1 + test.values, axis=0) - 1
        port_return = float((w * realized).sum())
        results.append({"rebalance_date": rd, "train_start": train_start, "train_end": train_end, "oos_return": port_return})
        print("rebalance", rd.date(), "oos_return", port_return)
    out = pd.DataFrame(results)
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path / "walkforward_results.csv", index=False)
    # cum plot
    out = out.sort_values("rebalance_date")
    cum = (1 + out["oos_return"]).cumprod()
    plt.figure(figsize=(8,4))
    plt.plot(out["rebalance_date"], cum, marker="o")
    plt.title("Walk-forward cumulative OOS return (CVaR optimized)")
    plt.xlabel("Rebalance date")
    plt.ylabel("Cumulative return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path / "walkforward_cumret.png", dpi=150)
    print("Saved walkforward results and plot to", out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--returns", default="data/processed/india_asset_returns_monthly.parquet")
    p.add_argument("--train_window", type=int, default=60)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--rebalance_freq", type=int, default=1)
    p.add_argument("--n_scen", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--out_dir", default="reports")
    p.add_argument("--start_date", default=None)
    p.add_argument("--end_date", default=None)
    p.add_argument("--max_weight", type=float, default=1.0)
    p.add_argument("--min_weight", type=float, default=0.0)
    args = p.parse_args()
    main(args)
