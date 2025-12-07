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
RETURNS_PATH = Path("data/processed/india_asset_returns_monthly.parquet") # Needed for interpretation
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "hmm_gmm_india.joblib"
REGIMES_OUT = Path("data/processed/hmm_regimes_india.parquet")
SUMMARY_FILE = MODEL_DIR / "hmm_train_summary.json"
MAPPING_FILE = MODEL_DIR / "state_mapping.json"

# Tuned Hyperparameters
N_STATES_TO_TRY = [2, 3, 4]
N_RESTARTS = 10            # CRITICAL: Run fit 10 times per state count to find global optimum
COV_TYPE = "full"          # 'full' captures correlation between features (Better for Finance)
N_ITER = 1000              # More iterations for convergence
TOLERANCE = 1e-4
RANDOM_SEEDS = [42, 7, 0, 123, 2023, 11, 13, 17, 99, 101] # Seeds for restarts

# Preprocessing Config
CLIP_OUTLIERS = True       # Winsorize data to remove extreme spikes (e.g., Covid crash)
CLIP_Q = (0.01, 0.99)      # Percentiles for clipping

def num_params_gaussian_hmm(n_states, n_features, cov_type="full"):
    """Calculate the number of free parameters for BIC calculation."""
    # Start probs: n_states - 1
    start_params = n_states - 1
    # Trans mat: n_states * (n_states - 1)
    trans_params = n_states * (n_states - 1)
    # Means: n_states * n_features
    mean_params = n_states * n_features
    
    # Covars:
    if cov_type == "diag":
        cov_params = n_states * n_features
    else: # full
        cov_params = n_states * (n_features * (n_features + 1) / 2)
        
    return int(start_params + trans_params + mean_params + cov_params)

def compute_aic_bic(logL, n_params, n_obs):
    aic = 2 * n_params - 2 * logL
    bic = n_params * math.log(n_obs) - 2 * logL
    return aic, bic

def preprocess_data(df):
    """
    Standardize and robustly scale the data.
    HMMs fail if features have vastly different variances.
    """
    X = df.values.copy()
    
    # 1. Handle NaN/Inf
    if np.isnan(X).any() or np.isinf(X).any():
        print("Warning: NaNs or Infs found in features. Filling with 0.")
        X = np.nan_to_num(X)

    # 2. Winsorize (Clip Outliers)
    # This prevents one massive crash (like 2020) from creating its own 'state'
    if CLIP_OUTLIERS:
        lower = np.percentile(X, CLIP_Q[0] * 100, axis=0)
        upper = np.percentile(X, CLIP_Q[1] * 100, axis=0)
        X = np.clip(X, lower, upper)

    # 3. Standardize (Mean=0, Std=1)
    # HMMs converge much faster on scaled data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

def train_one_model(X, n_states, seed):
    """
    Fits a single HMM model with specific initialization logic.
    """
    # 1. Smart Initialization using KMeans
    # We use KMeans to find rough clusters first
    km = KMeans(n_clusters=n_states, random_state=seed, n_init=10).fit(X)
    
    # 2. Define Model
    # init_params="" prevents the model from overwriting our custom initialization
    model = GaussianHMM(n_components=n_states, covariance_type=COV_TYPE,
                        n_iter=N_ITER, tol=TOLERANCE, 
                        random_state=seed, verbose=False,
                        init_params="") 

    # 3. "Sticky" Transition Matrix Initialization
    # We want regimes to be stable. Initialize diagonal with high probability (0.9)
    # This assumes markets stay in the same state 90% of the time.
    trans_init = np.ones((n_states, n_states)) * (1.0 - 0.90) / (n_states - 1)
    np.fill_diagonal(trans_init, 0.90)
    model.transmat_ = trans_init
    
    # 4. Initialize Means from KMeans
    model.means_ = km.cluster_centers_
    
    # 5. Initialize Start Probability
    model.startprob_ = np.bincount(km.labels_, minlength=n_states) / len(km.labels_)

    # 6. Initialize Covariances
    # For 'full', calculate empirical covariance of clusters
    n_features = X.shape[1]
    cv = np.zeros((n_states, n_features, n_features))
    for s in range(n_states):
        cluster_data = X[km.labels_ == s]
        if len(cluster_data) > 2:
            # Add small regularization (1e-6) to diagonal to prevent singular matrices
            cv[s] = np.cov(cluster_data.T) + np.eye(n_features) * 1e-6
        else:
            cv[s] = np.eye(n_features) # Fallback
    model.covars_ = cv

    # 7. Fit
    try:
        model.fit(X)
        return model
    except Exception as e:
        # print(f"Fit failed for seed {seed}: {e}")
        return None

def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")

    # --- 1. Load and Preprocess ---
    Xdf = pd.read_parquet(FEATURES_PATH)
    print(f"Loaded features {FEATURES_PATH} -> shape {Xdf.shape}")
    
    # Apply robust scaling and outlier clipping
    X, scaler = preprocess_data(Xdf)
    n_obs, n_features = X.shape
    
    # --- 2. Training Loop (Model Selection) ---
    results = []
    
    for n_states in N_STATES_TO_TRY:
        print(f"\n--- Training HMM with n_states={n_states} ---")
        
        best_model_n = None
        best_logL_n = -np.inf
        
        # RESTART LOOP: Try different seeds to avoid local minima
        for i, seed in enumerate(RANDOM_SEEDS[:N_RESTARTS]):
            model = train_one_model(X, n_states, seed)
            
            if model is not None:
                score = model.score(X)
                
                # Keep track of the best model for this n_state
                if score > best_logL_n:
                    best_logL_n = score
                    best_model_n = model
        
        if best_model_n is None:
            print(f"  Failed to fit any model for n_states={n_states}")
            continue

        # Compute Metrics
        n_params = num_params_gaussian_hmm(n_states, n_features, COV_TYPE)
        aic, bic = compute_aic_bic(best_logL_n, n_params, n_obs)
        
        print(f"  >> Best for n={n_states}: LogL={best_logL_n:.2f} | BIC={bic:.2f}")
        
        results.append({
            "n_states": n_states,
            "logL": best_logL_n,
            "n_params": n_params,
            "AIC": aic,
            "BIC": bic,
            "model": best_model_n
        })

    # --- 3. Select Best Model ---
    # We prefer lower BIC (Bayesian Information Criterion)
    if not results:
        print("No models converged!")
        return

    best_result = min(results, key=lambda r: r["BIC"])
    best_states = best_result["n_states"]
    best_model = best_result["model"]
    
    print(f"\nSELECTED BEST MODEL: n_states={best_states} (BIC={best_result['BIC']:.2f})")

    # --- 4. Save Model ---
    joblib.dump({
        "model": best_model,
        "scaler": scaler, # SAVE THE SCALER! We need it for inference later
        "feature_columns": Xdf.columns.tolist(),
        "n_states": best_states,
        "cov_type": COV_TYPE
    }, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")

    # --- 5. Generate Regimes ---
    hidden_states = best_model.predict(X)
    posteriors = best_model.predict_proba(X)
    
    out_df = Xdf.copy() # Use original DF for indices
    out_df["regime"] = hidden_states
    for i in range(best_states):
        out_df[f"prob_state_{i}"] = posteriors[:, i]
        
    out_df.to_parquet(REGIMES_OUT)
    print(f"Saved regimes to {REGIMES_OUT}")
    
    # Save Summary
    summary_json = [{k: v for k, v in r.items() if k != "model"} for r in results]
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary_json, f, indent=2)

    # --- 6. Interpret Regimes (Bull/Bear) ---
    if RETURNS_PATH.exists():
        print("\n--- STATE INTERPRETATION ---")
        m_ret = pd.read_parquet(RETURNS_PATH)
        # Find the Nifty column dynamically
        nifty_col = [c for c in m_ret.columns if "NSEI" in c or "NIFTY" in c]
        
        if nifty_col:
            target_col = nifty_col[0]
            # Join regimes with returns
            # Ensure indices align (usually datetime)
            analysis = out_df[["regime"]].join(m_ret[target_col], how="inner")
            
            stats = analysis.groupby("regime")[target_col].agg(["mean", "std", "count"])
            stats["ann_ret"] = stats["mean"] * 12
            stats["ann_vol"] = stats["std"] * np.sqrt(12)
            # Add small epsilon to vol to avoid div by zero
            stats["sharpe"] = stats["ann_ret"] / (stats["ann_vol"] + 1e-6)
            
            print(stats)
            
            # --- ROBUST LOGIC ---
            # Bear = Lowest Return (often high vol)
            bear_state = int(stats["ann_ret"].idxmin())
            
            # Bull = High Sharpe OR Highest Return
            # Note: Sometimes the highest return state is a "Rebound" state (very high vol).
            # We prefer a "Steady Bull" (High Sharpe).
            
            # Filter for positive return states only
            positive_states = stats[stats["ann_ret"] > 0]
            
            if not positive_states.empty:
                # If we have positive states, pick the one with best risk-adjusted return
                bull_state = int(positive_states["sharpe"].idxmax())
            else:
                # If everything is negative (rare), pick the least negative
                bull_state = int(stats["ann_ret"].idxmax())

            print(f"\nDetected Bull State: {bull_state}")
            print(f"Detected Bear State: {bear_state}")
            
            mapping = {
                "bull_state": bull_state,
                "bear_state": bear_state,
                "n_states": best_states
            }
            
            # If we have 3+ states, the leftover is usually "Sideways" or "High Volatility"
            if best_states > 2:
                all_states = set(range(best_states))
                others = list(all_states - {bull_state, bear_state})
                mapping["other_states"] = others
                
            with open(MAPPING_FILE, "w") as f:
                json.dump(mapping, f, indent=2)
            print(f"Saved state mapping to {MAPPING_FILE}")
        else:
            print("Could not find NIFTY column in returns file.")
    else:
        print("Returns file not found. Skipping interpretation.")

if __name__ == "__main__":
    main()