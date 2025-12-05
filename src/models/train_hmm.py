# src/models/train_hmm_india.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM
import joblib
import math
import json

#config
FEATURES_PATH = Path("data/processed/features_hmm_india.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE = MODEL_DIR / "hmm_gmm_india.joblib"
REGIMES_OUT = Path("data/processed/hmm_regimes_india.parquet")

N_STATES_TO_TRY = [3]   
COV_TYPE = "diag"
N_ITER = 200
RANDOM_STATE = 42

def num_params_gaussian_hmm(n_states, n_features):
    # free params:
    # startprob:n_states-1
    # transmat:n_states*(n_states - 1)
    # means:n_states*n_features
    # covars (diag):n_states*n_features
    return (n_states - 1) + n_states * (n_states - 1) + 2 * n_states * n_features

def compute_aic_bic(logL, n_params, n_obs):
    aic = 2 * n_params - 2 * logL
    bic = n_params * math.log(n_obs) - 2 * logL
    return aic, bic

def smart_init_kmeans(X, n_states):
    km = KMeans(n_clusters=n_states, random_state=RANDOM_STATE).fit(X)
    startprob_init = np.bincount(km.labels_) / len(km.labels_)
    means_init = km.cluster_centers_
    return startprob_init, means_init

def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")

    Xdf = pd.read_parquet(FEATURES_PATH)
    X = Xdf.values
    n_obs, n_features = X.shape
    print(f"Loaded features {FEATURES_PATH} -> shape {X.shape}")

    results = []
    for n_states in N_STATES_TO_TRY:
        print(f"\nFitting HMM with n_states={n_states}")
        #init
        startprob_init, means_init = smart_init_kmeans(X, n_states)
        model = GaussianHMM(n_components=n_states, covariance_type=COV_TYPE,
                            n_iter=N_ITER, random_state=RANDOM_STATE, verbose=False)
        #initial params
        model.startprob_ = startprob_init
        model.means_ = means_init
        try:
            model.fit(X)
        except Exception as e:
            print("Fit failed:", e)
            continue
        logL = model.score(X) * n_obs  #hmmlearn score returns avg logL per sample
        #hmmlearn.score returns total log likelihood of sequence
        logL = model.score(X)
        n_params = num_params_gaussian_hmm(n_states, n_features)
        aic, bic = compute_aic_bic(logL, n_params, n_obs)
        print(f"n_states={n_states} logL={logL:.2f} n_params={n_params} AIC={aic:.2f} BIC={bic:.2f}")
        results.append({"n_states":n_states, "logL":float(logL), "n_params":n_params, "AIC":aic, "BIC":bic, "model":model})

    #choose best by BIC
    best = min(results, key=lambda r: r["BIC"])
    best_states = best["n_states"]
    best_model = best["model"]
    print(f"\nSelected n_states={best_states} by BIC")

    #predicting regimes and posterior probs
    states = best_model.predict(X)
    post = best_model.predict_proba(X)

    #save model+metadata
    joblib.dump({
        "model": best_model,
        "feature_columns": Xdf.columns.tolist(),
        "n_states": best_states
    }, MODEL_FILE)
    print(f"Saved HMM model to {MODEL_FILE}")

    #save regimes as dataframe
    out_df = Xdf.copy()
    out_df["regime"] = states
    #also save posterior probabilities as separate columns
    for i in range(best_states):
        out_df[f"prob_state_{i}"] = post[:, i]
    out_df.to_parquet(REGIMES_OUT)
    print(f"Saved regime assignments to {REGIMES_OUT}")

    #save summary of results
    summary_path = MODEL_DIR / "hmm_train_summary.json"
    summary = [{"n_states": r["n_states"], "logL": r["logL"], "n_params": r["n_params"], "AIC": r["AIC"], "BIC": r["BIC"]} for r in results]
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training summary to {summary_path}")

if __name__ == "__main__":
    main()
