# eval_hmm.py  (replace relevant parts or entire file)
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

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

    out = pd.read_parquet(REGIMES_PATH)   # this should be monthly features + regime
    model_bundle = joblib.load(MODEL_FILE)
    model = model_bundle["model"]
    n_states = model.n_components

    print("Transition matrix:")
    print(model.transmat_)

    # state durations
    mean_dur, med_dur = average_state_duration(out["regime"].values)
    print(f"Average regime spell length (months): mean={mean_dur:.2f}, median={med_dur}")

    # load daily returns (log returns) for NIFTY so we can overlay cumulative daily returns
    rets = pd.read_parquet(PRICES_RET)  # ensure this is daily log-returns
    nifty_candidates = [c for c in rets.columns if "NSEI" in c.upper() or "NIFTY" in c.upper()]
    if not nifty_candidates:
        nifty = rets.columns[0]
    else:
        nifty = nifty_candidates[0]

    # Align regimes (monthly) with daily returns by forward filling regime labels to each day
    # out.index is monthly. create daily regime series by reindexing to daily index of returns
    monthly_regimes = out[["regime"]].copy()
    daily_index = rets.index
    # reindex monthly regimes to daily by forward filling
    regimes_daily = monthly_regimes.reindex(daily_index, method='ffill')
    # Join with daily NIFTY returns
    df = regimes_daily.join(rets[[nifty]], how="inner")
    df = df.dropna()
    df["cumret"] = df[nifty].cumsum()

    # Compute mean monthly return per regime from the monthly data (out contains monthly features/returns)
    # If your monthly returns column is e.g. 'NSEI' in out, adjust below. We'll compute using the same nifty col if present:
    if nifty in out.columns:
        monthly_ret_col = nifty
    else:
        # fallback: try to find asset matching NIFTY pattern in monthly file
        monthly_candidates = [c for c in out.columns if "NSEI" in c.upper() or "NIFTY" in c.upper()]
        monthly_ret_col = monthly_candidates[0] if monthly_candidates else None

    regime_perf = {}
    if monthly_ret_col is not None:
        perf = out.groupby("regime")[monthly_ret_col].mean()
        for r in perf.index:
            regime_perf[r] = perf.loc[r]
    else:
        # If we don't have the monthly NIFTY returns column, compute proxy using mean of features or posteriors
        perf = out.groupby("regime").mean().mean(axis=1)
        for r in perf.index:
            regime_perf[r] = perf.loc[r]

    # Map regimes -> colors: sort by perf, worst -> red, best -> green, others -> orange/yellow
    sorted_regs = sorted(regime_perf.items(), key=lambda x: x[1])
    reg_order = [r for r, _ in sorted_regs]
    colors_for_order = {}
    # assign colors by rank
    if len(reg_order) == 1:
        colors_for_order[reg_order[0]] = "lightgray"
    elif len(reg_order) == 2:
        colors_for_order[reg_order[0]] = "red"
        colors_for_order[reg_order[1]] = "green"
    else:
        # worst -> red, best -> green, middle(s) -> orange
        colors_for_order[reg_order[0]] = "red"
        for r in reg_order[1:-1]:
            colors_for_order[r] = "orange"
        colors_for_order[reg_order[-1]] = "green"

    # Now plot daily cumulative returns and shade by regime color
    plt.figure(figsize=(14,6))
    plt.plot(df.index, df["cumret"], label="NIFTY cum log-returns", color="black", linewidth=1.5)

    # iterate over contiguous regions where regime is constant and shade using the mapped color
    regimes_arr = df["regime"].values
    dates = df.index
    start_idx = 0
    curr_reg = regimes_arr[0]
    for i in range(1, len(regimes_arr)):
        if regimes_arr[i] != curr_reg:
            # shade from start_idx to i-1
            start_date = dates[start_idx]
            end_date = dates[i-1]
            mask = (dates >= start_date) & (dates <= end_date)
            plt.fill_between(dates[mask], df["cumret"][mask], color=colors_for_order.get(curr_reg, "lightgray"), alpha=0.18)
            start_idx = i
            curr_reg = regimes_arr[i]
    # last block
    start_date = dates[start_idx]
    mask = (dates >= start_date)
    plt.fill_between(dates[mask], df["cumret"][mask], color=colors_for_order.get(curr_reg, "lightgray"), alpha=0.18)

    # create a legend entry for each regime showing color + perf
    import matplotlib.patches as mpatches
    patches = []
    for r in sorted(regime_perf.keys()):
        label = f"regime {r} (mean={regime_perf[r]:.4f})"
        patches.append(mpatches.Patch(color=colors_for_order.get(r, "lightgray"), label=label))
    plt.legend(handles=[plt.Line2D([0],[0], color='black', label='NIFTY cum log-returns')] + patches, loc='upper left')
    plt.title("NIFTY cumulative returns with regime shading (colored by regime performance)")
    plt.tight_layout()
    out_png = out_dir / "regime_timeline_nifty_colored.png"
    plt.savefig(out_png, dpi=150)
    print("Saved plot to", out_png)

    # Print regime counts and mean posterior prob (unchanged)
    cols = [c for c in out.columns if c.startswith("prob_state_")]
    summary = pd.DataFrame({
        "count": out.groupby("regime").size(),
        "mean_posterior": out.groupby("regime")[cols].mean().max(axis=1)
    })
    print("\nRegime counts and mean posterior probabilities:")
    print(summary)

if __name__ == "__main__":
    main()