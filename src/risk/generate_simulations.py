import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t

# CONFIG
RETURNS_PATH = Path("data/processed/india_asset_returns_monthly.parquet")
OUT_DIR = Path("data/sims")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SIM = 1000   # number of paths
HORIZON = 30   # 30 trading days ~ 1.5 months
DOF = 4        # Degrees of freedom for fat-tailed student-t


def generate_gaussian(mu, sigma, n_sim, horizon):
    return np.random.normal(mu, sigma, size=(n_sim, horizon))


def generate_student_t(mu, sigma, dof, n_sim, horizon):
    # Student-t with scaling to match variance
    samples = t.rvs(df=dof, size=(n_sim, horizon))
    samples = samples / np.sqrt(dof / (dof - 2))  # variance correction
    return mu + sigma * samples


def generate_doomsday(mu, sigma, n_sim, horizon):
    """
    Heavy left-tail scenario:
       - 80% normal
       - 20% huge negative shock
    """
    base = np.random.normal(mu, sigma, size=(n_sim, horizon))
    shock_mask = np.random.rand(n_sim, horizon) < 0.2
    base[shock_mask] += np.random.normal(-3 * sigma, sigma, size=base[shock_mask].shape)
    return base


def main():
    if not RETURNS_PATH.exists():
        raise FileNotFoundError(RETURNS_PATH)

    df = pd.read_parquet(RETURNS_PATH)
    nifty_cols = [c for c in df.columns if "NSEI" in c.upper() or "NIFTY" in c.upper()]
    if not nifty_cols:
        raise RuntimeError("NIFTY column not found in returns file.")
    nifty = df[nifty_cols[0]]

    mu = nifty.mean()
    sigma = nifty.std()

    print("Using NIFTY monthly returns:")
    print("µ =", mu)
    print("σ =", sigma)

    gauss = generate_gaussian(mu, sigma, N_SIM, HORIZON)
    student = generate_student_t(mu, sigma, DOF, N_SIM, HORIZON)
    doom = generate_doomsday(mu, sigma, N_SIM, HORIZON)

    np.savez(OUT_DIR / "simulations.npz",
             gaussian=gauss,
             student_t=student,
             doomsday=doom)

    print("Saved simulation paths to data/sims/simulations.npz")

    # Asset-level simulations (for marginal contributions)
    returns = df.values  # historical returns
    n_assets = returns.shape[1]
    returns_sims = np.random.normal(df.mean(axis=0),
                                    df.std(axis=0),
                                    size=(N_SIM, HORIZON, n_assets))
    np.savez(OUT_DIR / "returns_sims.npz", returns=returns_sims)

    print("Saved asset-level simulations to data/sims/returns_sims.npz")


if __name__ == "__main__":
    main()

