import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json

def compute_mcvar(returns_sim, w, alpha=0.01):
    # returns_sim: (n_sim, T, n_assets)
    # compute portfolio final P&L or final returns; assume returns multiplicative: final portfolio value = (1 + r1)*(1 + r2)*... but easier: use cumulative return additive approx
    # For simplicity assume returns_sim are T-day returns (not path). We'll use final-period returns: returns_sim[:, -1, :]
    ret_T = returns_sim[:, -1, :]  # (n_sim, n_assets)
    port_vals = (ret_T * w).sum(axis=1)  # portfolio return contributions (can be P&L scaled)
    var = np.quantile(port_vals, alpha)
    tail_idx = port_vals <= var
    if tail_idx.sum() == 0:
        return None
    # average asset returns in tail
    avg_asset_ret_tail = ret_T[tail_idx].mean(axis=0)
    # marginal contribution in P&L = weight * avg_asset_ret_tail
    mcvar = w * avg_asset_ret_tail
    # contribution percentages
    total_tail_loss = (w * avg_asset_ret_tail).sum()
    pct = mcvar / total_tail_loss if total_tail_loss != 0 else np.zeros_like(mcvar)
    return mcvar, pct, var, port_vals[tail_idx].mean()

def main(args):
    inp = Path(args.inp)
    out_csv = Path(args.out_dir) / "mcvar_contributions.csv"
    if not inp.exists():
        raise FileNotFoundError(inp)
    d = np.load(inp)
    if 'returns' not in d:
        raise KeyError("returns array not found in npz (expected key 'returns')")
    returns = d['returns']  # shape n_sim, T, n_assets
    n_sim, T, n_assets = returns.shape
    print("Loaded returns sims:", returns.shape)
    # load weights
    if args.weights and Path(args.weights).exists():
        wdict = json.load(open(args.weights))
        # assume weights provided as list under "weights" and asset names under "assets"
        w = np.array(wdict.get("weights"))
        assets = wdict.get("assets")
    else:
        # equal weight
        w = np.ones(n_assets) / n_assets
        assets = [f"asset_{i}" for i in range(n_assets)]
    mcvar_res = compute_mcvar(returns, w, alpha=args.alpha)
    if mcvar_res is None:
        print("No tail observations found for given alpha.")
        return
    mcvar, pct, var, tail_mean = mcvar_res
    df = pd.DataFrame({
        "asset": assets,
        "weight": w,
        "mcvar": mcvar,
        "pct_of_total_tail": pct
    }).sort_values("mcvar")
    df.to_csv(out_csv, index=False)
    print("Saved MCVaR contributions to", out_csv)
    print(df)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="data/sims/returns_sims.npz")
    p.add_argument("--weights", default=None, help="JSON file {assets:[], weights:[]}")
    p.add_argument("--out_dir", default="reports")
    p.add_argument("--alpha", type=float, default=0.01)
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    main(args)
