import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def summarize_sims(sim_array, alphas=(0.01, 0.05)):
    res = {}
    res['median'] = np.median(sim_array, axis=0)
    res['mean'] = np.mean(sim_array, axis=0)
    for a in alphas:
        q = np.quantile(sim_array, a, axis=0)
        res[f'VaR_{int(a*100)}'] = q
        # CVaR: mean of draws <= VaR for each t
        cvar = np.empty(sim_array.shape[1])
        for t in range(sim_array.shape[1]):
            tail = sim_array[:, t][sim_array[:, t] <= q[t]]
            cvar[t] = tail.mean() if len(tail) > 0 else q[t]
        res[f'CVaR_{int(a*100)}'] = cvar
    return pd.DataFrame(res)

def main(args):
    inp = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    data = np.load(inp)
    # expected keys
    keys = [k for k in data.keys()]
    print("Found simulation keys:", keys)

    for key in keys:
        sims = data[key]  
        df = summarize_sims(sims, alphas=(0.01, 0.05))
        out_csv = out_dir / f"stress_summary_{key}.csv"
        df.to_csv(out_csv, index_label="horizon_step")
        print(f"Saved summary for {key} to {out_csv}")
        T = sims.shape[1] - 1
        snapshot = {
            "scenario": key,
            "horizon_steps": T,
            "median": float(df['median'].iloc[T]),
            "VaR_1%": float(df['VaR_1'].iloc[T]),
            "CVaR_1%": float(df['CVaR_1'].iloc[T]),
            "VaR_5%": float(df['VaR_5'].iloc[T]),
            "CVaR_5%": float(df['CVaR_5'].iloc[T])
        }
        print(snapshot)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="data/sims/simulations.npz", help=".npz file with simulation arrays")
    p.add_argument("--out_dir", default="reports", help="output folder for CSVs")
    args = p.parse_args()
    main(args)
