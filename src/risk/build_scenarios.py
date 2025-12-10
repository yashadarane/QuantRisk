# src/data/build_scenario_returns.py
#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import json
import argparse

def main(inp="data/sims/returns_sims.npz", out_dir="data/scenarios"):
    INP = Path(inp)
    OUT_DIR = Path(out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT = OUT_DIR / "scenario_returns.npy"
    META = OUT_DIR / "scenario_returns_meta.json"

    if not INP.exists():
        raise FileNotFoundError(f"Input file not found: {INP}")

    data = np.load(INP)
    key = list(data.keys())[0]
    arr = np.asarray(data[key])   # expected (n_sim, horizon, n_assets)
    print("Loaded", INP, "key:", key, "shape:", arr.shape)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (n_sim,horizon,n_assets). Got {arr.shape}")

    # compound returns across horizon: (1+r1)*(1+r2)*... - 1
    cum = np.prod(1.0 + arr, axis=1) - 1.0   # shape (n_sim, n_assets)
    np.save(OUT, cum)
    meta = {"source": str(INP), "key_used": key, "input_shape": arr.shape, "output_shape": cum.shape}
    with open(META, "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved scenario returns to:", OUT)
    print("Saved metadata to:", META)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="data/sims/returns_sims.npz")
    p.add_argument("--out_dir", default="data/scenarios")
    args = p.parse_args()
    main(args.inp, args.out_dir)
