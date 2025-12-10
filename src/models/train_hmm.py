# src/models/train_hmm_india.py
"""
Robust HMM training for QuantRisk (India)

- KMeans smart init
- sticky transition initialization
- multiple restarts to avoid local optima
- full/diag covariance handling
- saves model bundle (including scaler + feature order) and regimes parquet
- small regularization for covariance matrices to avoid singularities
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import joblib
import math
import json

# --- CONFIGURATION ---
FEATURES_PATH = Path("data/processed/features_hmm_india.parquet")
# try both common returns file names (monthly log returns or generic monthly)
RETURNS_PATHS = [
    Path("data/processed/india_asset_returns_monthly.parquet"),
    Path("data/processed/india_asset_returns_log.parquet"),
    Path("data/processed/india_asset_returns.parquet")
]
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "hmm_gmm_india.joblib"
REGIMES_OUT = Path("data/processed/hmm_regimes_india.parquet")
SUMMARY_FILE = MODEL_DIR / "hmm_train_summary.json"
MAPPING_FILE = MODEL_DIR / "state_mapping.json"

# Tuned Hyperparameters (adjustable)
N_STATES_TO_TRY = [2, 3]    # try a small grid first
N_RESTARTS = 5                 # number of seeds / restarts per n_states (set <= len(RANDOM_SEEDS))
COV_TYPE = "diag"              # 'diag' or 'full'
N_ITER = 500                   # iterations for EM (reduce for quick dev)
TOLERANCE = 1e-4
RANDOM_SEEDS = [42, 7, 0, 123, 2023, 11, 13, 17, 99, 101]

# Preprocessing Config
CLIP_OUTLIERS = False
CLIP_Q = (0.01, 0.99)

def num_params_gaussian_hmm(n_states, n_features, cov_type="full"):
    start_params = n_states - 1
    trans_params = n_states * (n_states - 1)
    mean_params = n_states * n_features
    if cov_type == "diag":
        cov_params = n_states * n_features
    else:
        cov_params = n_states * (n_features * (n_features + 1) / 2)
    return int(start_params + trans_params + mean_params + cov_params)


def compute_aic_bic(logL, n_params, n_obs):
    aic = 2 * n_params - 2 * logL
    bic = n_params * math.log(n_obs) - 2 * logL
    return aic, bic


def preprocess_data(df):
    X = df.values.astype(float).copy()

    # handle NaN / Inf
    if np.isnan(X).any() or np.isinf(X).any():
        # Fill with column median (more robust than 0)
        col_med = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_med, inds[1])
        # replace inf with large finite value
        X[np.isinf(X)] = np.nan
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_med, inds[1])
        print("Filled NaN/Inf in features using column medians.")

    # optional winsorize / clipping
    if CLIP_OUTLIERS:
        lower = np.percentile(X, CLIP_Q[0] * 100, axis=0)
        upper = np.percentile(X, CLIP_Q[1] * 100, axis=0)
        X = np.clip(X, lower, upper)
        print("Applied outlier clipping to features.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def init_from_kmeans(X, n_states, seed):
    """
    KMeans init that returns startprob, means, covars_init, labels
    covars_init has shape:
     - diag: (n_states, n_features)
     - full: (n_states, n_features, n_features)
    """
    km = KMeans(n_clusters=n_states, random_state=seed, n_init=10).fit(X)
    labels = km.labels_
    startprob = np.bincount(labels, minlength=n_states) / len(labels)
    means = km.cluster_centers_

    n_features = X.shape[1]
    if COV_TYPE == "diag":
        covars = np.zeros((n_states, n_features))
        for s in range(n_states):
            members = X[labels == s]
            if len(members) > 1:
                covars[s, :] = np.var(members, axis=0) + 1e-6
            else:
                covars[s, :] = np.var(X, axis=0) + 1e-6
    else:  # full
        covars = np.zeros((n_states, n_features, n_features))
        for s in range(n_states):
            members = X[labels == s]
            if len(members) > 2:
                cov = np.cov(members.T)
                # regularize
                cov += np.eye(n_features) * 1e-6
                covars[s] = cov
            else:
                # fallback: global covariance scaled down (more stable)
                cov = np.cov(X.T)
                cov += np.eye(n_features) * 1e-6
                covars[s] = cov
    return startprob, means, covars, labels


def train_one_model(X, n_states, seed):
    """Initialize HMM from KMeans, sticky transmat, and fit. Returns model or None."""
    startprob_init, means_init, covars_init, labels = init_from_kmeans(X, n_states, seed)

    model = GaussianHMM(n_components=n_states,
                        covariance_type=COV_TYPE,
                        n_iter=N_ITER,
                        tol=TOLERANCE,
                        random_state=seed,
                        init_params="")  # prevent internal re-init

    # Sticky transition init (high self-transition)
    stickiness = 0.90
    trans_init = np.ones((n_states, n_states)) * ((1.0 - stickiness) / max(1, (n_states - 1)))
    np.fill_diagonal(trans_init, stickiness)
    model.transmat_ = trans_init

    # set initial parameters
    model.startprob_ = startprob_init
    model.means_ = means_init
    model.covars_ = covars_init

    try:
        model.fit(X)
        return model
    except Exception as e:
        # fitting may fail (singular cov etc.)
        return None


def find_returns_file():
    for p in RETURNS_PATHS:
        if p.exists():
            return p
    return None


def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")

    # 1) Load features and preprocess
    Xdf = pd.read_parquet(FEATURES_PATH)
    print(f"Loaded features {FEATURES_PATH} -> shape {Xdf.shape}")

    X, scaler = preprocess_data(Xdf)
    n_obs, n_features = X.shape
    print(f"Preprocessed features -> shape {X.shape}")

    results = []

    for n_states in N_STATES_TO_TRY:
        print(f"\n--- Training HMM with n_states={n_states} ---")
        best_model_for_n = None
        best_logL_for_n = -np.inf

        # run multiple restarts (seeds)
        seeds = RANDOM_SEEDS[:N_RESTARTS]
        for seed in seeds:
            model = train_one_model(X, n_states, seed)
            if model is None:
                continue
            try:
                logL = model.score(X)   # total log-likelihood of observed sequence
            except Exception:
                continue
            if logL > best_logL_for_n:
                best_logL_for_n = logL
                best_model_for_n = model

        if best_model_for_n is None:
            print(f"  Failed to fit any model for n_states={n_states}")
            continue

        n_params = num_params_gaussian_hmm(n_states, n_features, COV_TYPE)
        aic, bic = compute_aic_bic(best_logL_for_n, n_params, n_obs)
        print(f"  >> Best for n={n_states}: LogL={best_logL_for_n:.2f} | AIC={aic:.2f} | BIC={bic:.2f}")

        results.append({
            "n_states": n_states,
            "logL": float(best_logL_for_n),
            "n_params": n_params,
            "AIC": float(aic),
            "BIC": float(bic),
            "model": best_model_for_n
        })

    if not results:
        raise RuntimeError("No models were successfully fitted.")

    # pick best by BIC
    best_result = min(results, key=lambda r: r["BIC"])
    best_states = best_result["n_states"]
    best_model = best_result["model"]
    print(f"\nSELECTED BEST MODEL: n_states={best_states} (BIC={best_result['BIC']:.2f})")

    # Save model bundle (model + scaler + feature names)
    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "feature_columns": Xdf.columns.tolist(),
        "n_states": best_states,
        "cov_type": COV_TYPE
    }, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")

    # Predict regimes & posterior probs on the training data
    hidden_states = best_model.predict(X)
    posteriors = best_model.predict_proba(X)

    out_df = Xdf.copy()
    out_df["regime"] = hidden_states
    for i in range(best_states):
        out_df[f"prob_state_{i}"] = posteriors[:, i]

    out_df.to_parquet(REGIMES_OUT)
    print(f"Saved regimes to {REGIMES_OUT}")

    # Save training summary (without model object)
    summary_json = [{k: v for k, v in r.items() if k != "model"} for r in results]
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved training summary to {SUMMARY_FILE}")

    # Interpret regimes if returns file exists
    returns_file = find_returns_file()
    if returns_file:
        print("\n--- STATE INTERPRETATION ---")
        m_ret = pd.read_parquet(returns_file)
        # detect nifty column case-insensitive
        nifty_cols = [c for c in m_ret.columns if "NSEI" in c.upper() or "NIFTY" in c.upper()]
        if nifty_cols:
            target_col = nifty_cols[0]
            analysis = out_df[["regime"]].join(m_ret[[target_col]], how="inner")
            stats = analysis.groupby("regime")[target_col].agg(["mean", "std", "count"])
            stats["ann_ret"] = stats["mean"] * 12
            stats["ann_vol"] = stats["std"] * np.sqrt(12)
            stats["sharpe"] = stats["ann_ret"] / (stats["ann_vol"] + 1e-9)
            print(stats)

            # Robust detection rules
            bear_state = int(stats["ann_ret"].idxmin())
            positive_states = stats[stats["ann_ret"] > 0]
            if not positive_states.empty:
                bull_state = int(positive_states["sharpe"].idxmax())
            else:
                bull_state = int(stats["ann_ret"].idxmax())

            mapping = {
                "bull_state": bull_state,
                "bear_state": bear_state,
                "n_states": best_states,
                "stats": stats.to_dict()
            }

            with open(MAPPING_FILE, "w") as f:
                json.dump(mapping, f, indent=2)
            print(f"Saved state mapping to {MAPPING_FILE}")
        else:
            print("Could not detect NIFTY column in returns file; skipping interpretation.")
    else:
        print("No returns file found; skipping regime interpretation.")


if __name__ == "__main__":
    main()
