# src/risk/marginal_contributions_final.py
#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json

def load_returns_sims(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    arr = np.load(p)
    key = list(arr.keys())[0]
    return arr[key]   # shape (n_sim, horizon, n_assets)

def compute_mcvar_from_final_scenarios(scenarios, w, alpha=0.01):
    # scenarios: (n_sim, n_assets) final cumulative returns
    port = (scenarios * w).sum(axis=1)
    var = np.quantile(port, alpha)
    tail_idx = port <= var
    if tail_idx.sum() == 0:
        # fallback: use quantile approximate
        tail_idx = port <= var
    avg_asset_ret_tail = scenarios[tail_idx].mean(axis=0) if tail_idx.sum()>0 else scenarios.mean(axis=0)
    mcvar = w * avg_asset_ret_tail
    total_loss = (w * avg_asset_ret_tail).sum()
    pct = mcvar / total_loss if total_loss != 0 else np.zeros_like(mcvar)
    return mcvar, pct, var, port[tail_idx].mean() if tail_idx.sum()>0 else var

def main(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # Prefer final-scenarios (2D). If not found, try returns_sims (3D) and compute final cumulative returns.
    if Path(args.scenarios).exists():
        data = np.load(args.scenarios)
        if isinstance(data, np.lib.npyio.NpzFile):
            # may contain multiple arrays; try to pick one named 'returns' or first key
            keys = list(data.keys())
            arr = data[keys[0]]
        else:
            arr = data
        arr = np.asarray(arr)
        if arr.ndim == 3:
            # compute final cumulative returns
            scenarios = np.prod(1.0 + arr, axis=1) - 1.0
        elif arr.ndim == 2:
            scenarios = arr
        else:
            raise ValueError("Unsupported scenarios shape: " + str(arr.shape))
    else:
        raise FileNotFoundError(args.scenarios)

    n_sim, n_assets = scenarios.shape
    # weights
    if args.weights:
        wdict = json.load(open(args.weights))
        w = np.array(wdict["weights"])
        assets = wdict.get("assets", [f"asset_{i}" for i in range(n_assets)])
    else:
        w = np.ones(n_assets)/n_assets
        assets = [f"asset_{i}" for i in range(n_assets)]

    mcvar, pct, var, tail_mean = compute_mcvar_from_final_scenarios(scenarios, w, alpha=args.alpha)
    df = pd.DataFrame({"asset": assets, "weight": w, "mcvar": mcvar, "pct_of_total_tail": pct})
    df = df.sort_values("mcvar")  # worst (most negative) first
    out_csv = out_dir / "mcvar_contributions.csv"
    df.to_csv(out_csv, index=False)
    print("Saved MCVaR contributions to", out_csv)
    print(df)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", default="data/sims/returns_sims.npz", help=".npz with 'returns' 3D or .npy 2D scenarios")
    p.add_argument("--weights", default=None, help="optional JSON {assets:[], weights:[]}")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--out_dir", default="reports")
    args = p.parse_args()
    main(args)
