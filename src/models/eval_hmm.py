# src/models/eval_hmm_india.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

REGIMES_PATH = Path("data/processed/hmm_regimes_india.parquet")
MODEL_FILE = Path("models/hmm_gmm_india.joblib")
PRICES_RET = Path("data/processed/india_asset_returns_log.parquet")  # to plot NIFTY

out_dir = Path("reports")
out_dir.mkdir(parents=True, exist_ok=True)

def average_state_duration(states):
    durations = []
    curr = states[0]
    length = 1
    for s in states[1:]:
        if s == curr:
            length += 1
        else:
            durations.append(length)
            curr = s
            length = 1
    durations.append(length)
    return np.mean(durations), np.median(durations)

def main():
    if not REGIMES_PATH.exists() or not MODEL_FILE.exists():
        raise FileNotFoundError("Run training first.")

    out = pd.read_parquet(REGIMES_PATH)
    model_bundle = joblib.load(MODEL_FILE)
    model = model_bundle["model"]
    n_states = model.n_components

    print("Transition matrix:")
    print(model.transmat_)

    # state durations
    mean_dur, med_dur = average_state_duration(out["regime"].values)
    print(f"Average regime spell length (months): mean={mean_dur:.2f}, median={med_dur}")

    # plot regimes over NIFTY cumulative returns
    rets = pd.read_parquet(PRICES_RET)
    nifty_candidates = [c for c in rets.columns if "NSEI" in c.upper() or "NIFTY" in c.upper()]
    if not nifty_candidates:
        nifty = rets.columns[0]
    else:
        nifty = nifty_candidates[0]

    df = out.join(rets[[nifty]], how="inner")
    df = df.dropna()
    df["cumret"] = df[nifty].cumsum()

    plt.figure(figsize=(12,5))
    plt.plot(df.index, df["cumret"], label="NIFTY cum log-returns")
    # shade regimes
    for s in sorted(df["regime"].unique()):
        mask = (df["regime"] == s)
        plt.fill_between(df.index, df["cumret"], where=mask, alpha=0.12, label=f"regime {s}")
    plt.legend()
    plt.title("NIFTY cumulative returns with regime shading")
    plt.tight_layout()
    out_png = out_dir / "regime_timeline_nifty.png"
    plt.savefig(out_png, dpi=150)
    print("Saved plot to", out_png)

    # Print regime counts and mean posterior prob
    print("\nRegime counts and mean posterior probabilities:")
    cols = [c for c in out.columns if c.startswith("prob_state_")]
    summary = pd.DataFrame({
        "count": out.groupby("regime").size(),
        "mean_posterior": out.groupby("regime")[cols].mean().max(axis=1)
    })
    print(summary)

if __name__ == "__main__":
    main()
