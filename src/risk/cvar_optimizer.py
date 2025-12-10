
"""
CVaR optimizer (minimize CVaR at level alpha) using cvxpy.

Features:
 - If --scenarios not provided, auto-detects the newest .npy/.npz under data/scenarios/ or data/sims/
 - Graceful error messages and solver fallbacks
 - Saves results to models/cvar_weights.csv
 - Accepts optional --assets list

Usage:
  # Auto-detect scenario file:
  python src/portfolio/cvar_optimizer.py

  # Or explicitly:
  python src/portfolio/cvar_optimizer.py --scenarios data/scenarios/scenario_returns.npy --alpha 0.01
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import cvxpy as cp
import sys
import math
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cvar_optimizer")


def find_latest_scenarios(dirs=("data/scenarios", "data/sims", "data")):
    """Search directories for .npy/.npz files and return the newest path, or None."""
    candidates = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        for ext in ("*.npy", "*.npz"):
            for f in p.glob(ext):
                candidates.append(f)
    if not candidates:
        return None
    # pick newest by modified time
    latest = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return latest


def load_scenarios(path: Path):
    """Load scenarios from .npy or .npz. Return 2D numpy array (N_scen, n_assets)."""
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    if path.suffix == ".npz":
        data = np.load(path)
        # pick the first array key
        keys = list(data.keys())
        if not keys:
            raise ValueError(f"No arrays found inside {path}")
        arr = data[keys[0]]
        logger.info("Loaded keys %s from %s (using '%s')", keys, path, keys[0])
    elif path.suffix == ".npy":
        arr = np.load(path)
        logger.info("Loaded %s", path)
    else:
        raise ValueError("Scenario file must be .npy or .npz")
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Scenarios array must be 2D (N_scen, n_assets). Got shape {arr.shape}")
    return arr


def solve_cvar(scenarios: np.ndarray, alpha=0.01, max_weight=1.0, min_weight=0.0, solver_preference=("ECOS", "OSQP", "SCS")):
    """
    Solve min CVaR:
        minimize v + (1/(alpha*N)) sum z_i
        s.t. z_i >= loss_i - v, z_i >= 0
             loss_i = - r_i @ w
             sum(w) == 1, min_weight <= w <= max_weight

    scenarios: (N, n_assets)
    returns: w_opt (n,), v_opt, status, solver_used
    """
    N, n = scenarios.shape
    logger.info("Solving CVaR: scenarios=%s, assets=%d, alpha=%s", scenarios.shape, n, alpha)

    w = cp.Variable(n)
    v = cp.Variable()
    z = cp.Variable(N)

    losses = -scenarios @ w  # shape (N,)

    constraints = [z >= 0, z >= losses - v, cp.sum(w) == 1, w >= min_weight, w <= max_weight]

    obj = cp.Minimize(v + (1.0 / (alpha * N)) * cp.sum(z))
    prob = cp.Problem(obj, constraints)

    last_err = None
    for solver_name in solver_preference:
        try:
            logger.info("Attempting solve with %s...", solver_name)
            prob.solve(solver=solver_name, verbose=False)
            status = prob.status
            logger.info("Solver %s finished with status: %s", solver_name, status)
            if status in ("optimal", "optimal_inaccurate"):
                return np.array(w.value, dtype=float), float(v.value), status, solver_name
            else:
                # solver ran but not optimal; return whatever it produced
                return np.array(w.value if w.value is not None else np.zeros(n), dtype=float), float(v.value if v.value is not None else math.nan), status, solver_name
        except Exception as e:
            last_err = e
            logger.warning("Solver %s failed: %s", solver_name, str(e))
            continue

    # If we reach here, all solvers failed
    raise RuntimeError(f"All solvers failed. Last error: {last_err}")


def save_weights(weights: np.ndarray, assets, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if assets:
        if len(assets) != len(weights):
            logger.warning("Provided assets list length doesn't match weights; falling back to generic names.")
            assets = [f"asset_{i}" for i in range(len(weights))]
    else:
        assets = [f"asset_{i}" for i in range(len(weights))]

    df = pd.DataFrame({"asset": assets, "weight": weights})
    out_path = out_dir / "cvar_weights.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved weights to %s", out_path)
    return out_path


def main(args):
    # 1) find scenarios if not provided
    scenarios_path = Path(args.scenarios) if args.scenarios else None
    if scenarios_path is None:
        scenarios_path = find_latest_scenarios()
        if scenarios_path:
            logger.info("Auto-detected scenarios file: %s", scenarios_path)
        else:
            logger.error("No scenario files found under data/scenarios or data/sims. Please generate scenarios or pass --scenarios.")
            sys.exit(1)

    try:
        scenarios = load_scenarios(scenarios_path)
    except Exception as e:
        logger.exception("Failed to load scenarios: %s", e)
        sys.exit(1)

    # 2) solve
    try:
        w_opt, v_opt, status, solver_used = solve_cvar(scenarios, alpha=args.alpha,
                                                       max_weight=args.max_weight,
                                                       min_weight=args.min_weight,
                                                       solver_preference=(args.solver.split(",") if args.solver else ("ECOS", "OSQP", "SCS")))
    except Exception as e:
        logger.exception("Optimization failed: %s", e)
        sys.exit(1)

    logger.info("Optimization finished (solver=%s, status=%s). v_opt=%s", solver_used, status, v_opt)
    # 3) save results
    asset_list = [s.strip() for s in args.assets.split(",")] if args.assets else None
    out_path = save_weights(w_opt, asset_list, Path(args.out_dir))
    print("Done. Weights saved to:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default=None, help="Path to .npy or .npz scenarios file. If omitted, latest under data/scenarios or data/sims is used.")
    parser.add_argument("--alpha", type=float, default=0.01, help="CVaR alpha level")
    parser.add_argument("--assets", default=None, help="Comma-separated asset names in same column order as scenario file")
    parser.add_argument("--out_dir", default="models", help="Output directory for weights CSV")
    parser.add_argument("--no_long_only", action="store_true", help="Allow shorting (not recommended).")
    parser.add_argument("--max_weight", type=float, default=1.0, help="Maximum weight per asset (use 0.5 for 50%)")
    parser.add_argument("--min_weight", type=float, default=0.0, help="Minimum weight per asset (>=0 for long-only)")
    parser.add_argument("--solver", default=None, help="Comma-separated solvers to try (ECOS,OSQP,SCS). Default tries ECOS, OSQP, SCS.")
    args = parser.parse_args()

    # If user sets --no_long_only, allow negative min_weight
    if args.no_long_only and args.min_weight >= 0:
        args.min_weight = -1.0 * args.max_weight

    main(args)
