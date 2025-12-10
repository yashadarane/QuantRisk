# src/dashboard/app.py
"""
QuantRisk Dashboard (Streamlit)
- view regime timeline (if produced)
- inspect simulation distributions and numeric VaR/CVaR
- view MCVaR table
- run CVaR optimizer interactively and download weights
- quick compare: equal-weight vs optimized CVaR on scenario set
"""

import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import io
import json
import cvxpy as cp
import math
import glob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quantrisk_streamlit")

# -----------------------
# Helper functions
# -----------------------
@st.cache_data(show_spinner=False)
def find_latest_file(patterns):
    """Return newest file matching any of glob patterns (list)."""
    candidates = []
    for pat in patterns:
        for f in glob.glob(pat):
            candidates.append(Path(f))
    if not candidates:
        return None
    latest = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return latest

@st.cache_data(show_spinner=False)
def load_npz_or_npy(path):
    p = Path(path)
    if p.suffix == ".npz":
        d = np.load(p)
        key = list(d.keys())[0]
        arr = d[key]
    else:
        arr = np.load(p)
    return np.asarray(arr)

@st.cache_data(show_spinner=False)
def load_mcvar(csv_path):
    p = Path(csv_path)
    if not p.exists():
        return None
    return pd.read_csv(p)

@st.cache_data(show_spinner=False)
def compute_scenario_from_returns_sims(npz_path):
    a = np.load(npz_path)
    key = list(a.keys())[0]
    arr = np.asarray(a[key])  # (n_sim, horizon, n_assets)
    if arr.ndim == 3:
        cum = np.prod(1.0 + arr, axis=1) - 1.0
        return cum
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError("Unsupported returns shape: " + str(arr.shape))

def compute_var_cvar_from_scenarios(scenarios, alpha=0.01):
    # scenarios: (n_sim, n_assets) final cumulative returns
    port = scenarios.sum(axis=1) if scenarios.ndim==2 and scenarios.shape[1]==1 else None
    # Usually we need portfolio weights to get port returns; for distribution of a specific weight vector pass them in
    # Here compute per-asset distribution stats and simple portfolio stats for equal weight
    return None

def solve_cvar_inprocess(scenarios, alpha=0.01, max_weight=1.0, min_weight=0.0):
    """
    scenarios: (N, n_assets) final cumulative returns
    Returns: optimal weights (n_assets,)
    """
    N, n = scenarios.shape
    w = cp.Variable(n)
    v = cp.Variable()
    z = cp.Variable(N)
    losses = -scenarios @ w
    constraints = [z >= 0, z >= losses - v, cp.sum(w) == 1, w >= min_weight, w <= max_weight]
    obj = cp.Minimize(v + (1.0 / (alpha * N)) * cp.sum(z))
    prob = cp.Problem(obj, constraints)
    # solver fallback built-in: try a quick loop
    solvers = [cp.ECOS, cp.OSQP, cp.SCS]
    last_exc = None
    for s in solvers:
        try:
            prob.solve(solver=s, verbose=False)
            status = prob.status
            break
        except Exception as e:
            last_exc = e
            continue
    if w.value is None:
        raise RuntimeError(f"No solver succeeded. Last error: {last_exc}")
    return np.array(w.value, dtype=float), status

def format_weights_df(weights, asset_names=None):
    if asset_names is None:
        asset_names = [f"asset_{i}" for i in range(len(weights))]
    df = pd.DataFrame({"asset": asset_names, "weight": weights})
    df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    return df

def download_csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="QuantRisk Dashboard", layout="wide")
st.title("QuantRisk — Regime-Aware Stress & CVaR Dashboard (India)")

# Sidebar controls
st.sidebar.header("Inputs & Controls")

# Scenario file selector (auto-detect latest)
latest_scn = find_latest_file(["data/scenarios/*.npy", "data/sims/returns_sims.npz", "data/sims/*.npz"])
scn_path = st.sidebar.text_input("Scenario file (.npy or .npz)", value=str(latest_scn) if latest_scn else "")
if scn_path:
    scn_path = Path(scn_path)
else:
    st.sidebar.warning("No scenario file found; generate simulations first.")
alpha = st.sidebar.slider("CVaR alpha", min_value=0.005, max_value=0.2, value=0.01, step=0.005)
max_w = st.sidebar.slider("Max weight per asset", 0.1, 1.0, 1.0, step=0.05)
min_w = st.sidebar.number_input("Min weight per asset", value=0.0, format="%.3f")
assets_input = st.sidebar.text_input("Asset names (comma-separated)", value="")  # optional

st.sidebar.markdown("---")
st.sidebar.header("Files & Diagnostics")
mcvar_latest = find_latest_file(["reports/mcvar_contributions.csv"])
if mcvar_latest:
    st.sidebar.write("MCVaR:", mcvar_latest.name)
else:
    st.sidebar.write("MCVaR: (not found)")

hmm_img = find_latest_file(["reports/regime_timeline_*.png", "reports/regime_timeline_nifty.png"])
if hmm_img:
    st.sidebar.write("Regime plot:", hmm_img.name)

st.sidebar.markdown("---")
st.sidebar.write("Run CVaR optimize or inspect scenario distributions below.")

# Main layout: three columns
col1, col2 = st.columns((2,3))

with col1:
    st.subheader("1) Scenario inspector")
    if scn_path and scn_path.exists():
        try:
            arr = load_npz_or_npy(scn_path)
            # If 3D, convert to final-period cumulative returns
            if arr.ndim == 3:
                st.info(f"Loaded returns-sims (n_sim={arr.shape[0]}, horizon={arr.shape[1]}, n_assets={arr.shape[2]})")
                scenarios = np.prod(1.0 + arr, axis=1) - 1.0
            elif arr.ndim == 2:
                st.info(f"Loaded scenario matrix (n_sim={arr.shape[0]}, n_assets={arr.shape[1]})")
                scenarios = arr
            else:
                st.error(f"Unsupported scenario array shape: {arr.shape}")
                scenarios = None

            if scenarios is not None:
                n_sim, n_assets = scenarios.shape
                st.metric("N scenarios", n_sim)
                st.metric("N assets", n_assets)
                st.write("Sample scenario rows (first 5):")
                st.dataframe(pd.DataFrame(scenarios[:5,:], columns=[f"asset_{i}" for i in range(n_assets)]))

                # distribution plot of portfolio (equal-weight) final returns
                ew = np.ones(n_assets)/n_assets
                port = (scenarios * ew).sum(axis=1)
                fig = px.histogram(pd.DataFrame({"port_return": port}), x="port_return", nbins=80, title="Equal-weight portfolio final returns")
                st.plotly_chart(fig, use_container_width=True)

                # show numeric VaR/CVaR
                var1 = np.quantile(port, alpha)
                cvar1 = port[port <= var1].mean()
                st.write(f"Equal-weight portfolio: VaR {int(alpha*100)}% = {var1:.4f}, CVaR {int(alpha*100)}% = {cvar1:.4f}")

                # per-asset boxplot
                df_assets = pd.DataFrame(scenarios, columns=[f"asset_{i}" for i in range(n_assets)])
                fig2 = px.box(df_assets.melt(var_name="asset", value_name="return"), x="asset", y="return", title="Per-asset final return distribution")
                st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error("Failed to load scenarios: " + str(e))
    else:
        st.info("No scenario file selected or found. Generate simulations first.")

    st.markdown("---")
    st.subheader("MCVaR (Tail Attribution)")
    mc = load_mcvar("reports/mcvar_contributions.csv") if Path("reports/mcvar_contributions.csv").exists() else None
    if mc is not None:
        st.dataframe(mc)
        st.download_button("Download MCVaR CSV", data=mc.to_csv(index=False).encode(), file_name="mcvar_contributions.csv")
    else:
        st.info("MCVaR file not found. Run `marginal_contributions` script.")

with col2:
    st.subheader("2) CVaR Optimizer (interactive)")
    st.write("Alpha:", alpha, "Max weight:", max_w, "Min weight:", min_w)
    run_opt = st.button("Run CVaR optimizer on selected scenarios")
    if run_opt:
        if not (scn_path and scn_path.exists()):
            st.error("No valid scenarios file selected.")
        else:
            try:
                # ensure 'scenarios' var exists from earlier block
                # if not, reload
                if 'scenarios' not in locals():
                    arr = load_npz_or_npy(scn_path)
                    if arr.ndim == 3:
                        scenarios = np.prod(1.0 + arr, axis=1) - 1.0
                    else:
                        scenarios = arr
                    n_sim, n_assets = scenarios.shape
                # parse asset names if provided
                assets = None
                if assets_input:
                    assets = [a.strip() for a in assets_input.split(",")]
                    if len(assets) != n_assets:
                        st.warning("Provided asset list length doesn't match scenario columns; using generic names.")
                        assets = None
                w_opt, status = solve_cvar_inprocess(scenarios, alpha=alpha, max_weight=max_w, min_weight=min_w)
                dfw = format_weights_df(w_opt, asset_names=assets)
                st.success(f"Solver status: {status}")
                st.dataframe(dfw)
                st.download_button("Download weights CSV", data=download_csv_bytes(dfw), file_name="cvar_weights.csv")
                # quick compare stats
                ew = np.ones(n_assets)/n_assets
                port_ew = (scenarios * ew).sum(axis=1)
                port_opt = (scenarios * w_opt).sum(axis=1)
                df_compare = pd.DataFrame({
                    "strategy": ["equal_weight", "cvar_opt"],
                    "VaR": [np.quantile(port_ew, alpha), np.quantile(port_opt, alpha)],
                    "CVaR": [port_ew[port_ew <= np.quantile(port_ew, alpha)].mean(), port_opt[port_opt <= np.quantile(port_opt, alpha)].mean()],
                    "median": [np.median(port_ew), np.median(port_opt)]
                })
                st.write("Quick strategy comparison (final-horizon):")
                st.dataframe(df_compare)
            except Exception as e:
                st.exception(e)

    st.markdown("---")
    st.subheader("3) Regime Timeline")
    if hmm_img and Path(hmm_img).exists():
        st.image(str(hmm_img), caption="Regime timeline", use_column_width=True)
    else:
        st.info("No regime timeline image found (reports/regime_timeline_*.png). Run your HMM eval script to generate it.")


