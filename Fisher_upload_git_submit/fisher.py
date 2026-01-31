#===================================================================================================
#  Import and  Device Set up
#===================================================================================================
from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path

import numpy as np

# Ensure parent folder is importable (contains All_Schemes.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from All_Schemes import (
    DistName,
    get_unit_variance_pdf,
    fisher_continuous,
    compute_t_fx,
    compute_nonadaptive_lower_bound,
    compute_adaptive_lower_bound,
)

#===================================================================================================
#    Reproducibility Utilities
#===================================================================================================
REPRO_MODE = True
GLOBAL_BASE_SEED = 42

if REPRO_MODE:
    np.random.seed(GLOBAL_BASE_SEED)

# ============================================================
#  Distributions 
# ============================================================
DIST_LIST: tuple[DistName, ...] = ("gaussian", "logistic", "hypsecant", "sin2")

# ============================================================
#  Directory for results
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[0]
DATA_DIR = ROOT_DIR / "Fisher_Data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# n-grid for MSE curves
n_vals = np.arange(50, 20001, 50, dtype=np.int64)

# sigma fixed for this Fisher experiment
SIGMA = 1.0   # do not change; paper uses unit-variance family

# ============================================================
#  Main loop
# ============================================================
for dist in DIST_LIST:
    print("\n==========================================")
    print(f"Running distribution: {dist}")
    print("==========================================")

    # f(0)
    pdf = get_unit_variance_pdf(dist)
    fx0 = float(pdf(0.0))

    # Fisher information for location of X = mu + sigma Z
    # (Your fisher_continuous returns I_X = I_Z / sigma^2)
    I_cont = float(fisher_continuous(dist, sigma=SIGMA))

    if I_cont <= 0.0 or not np.isfinite(I_cont):
        raise ValueError(f"Non-positive or invalid Fisher information for dist={dist}: I_cont={I_cont}")

    # Lower bounds
    T_val = float(compute_t_fx(dist))
    C_non = float(compute_nonadaptive_lower_bound(dist, SIGMA, override_t=T_val))
    C_adp = float(compute_adaptive_lower_bound(dist, SIGMA))

    # Print summary
    print(f"f_X(0)               = {fx0:.6f}")
    print(f"T(f_X)               = {T_val:.6f}")
    print(f"I_cont               = {I_cont:.6f}")
    print(f"C_non                = {C_non:.6f}")
    print(f"C_adp                = {C_adp:.6f}")
    print(f"C_non / C_adp        = {C_non / C_adp:.6f}")

    # MSE curves (asymptotic  constant / n)
    mse_cont = (1.0 / I_cont) / n_vals
    mse_non  = C_non / n_vals
    mse_adp  = C_adp / n_vals

    # Save everything
    out = {
        "dist": dist,
        "sigma": SIGMA,

        "f0": fx0,
        "T_fx": T_val,
        "I_cont": I_cont,
        "C_non": C_non,
        "C_adp": C_adp,

        "n_vals": n_vals,
        "mse_cont": mse_cont,
        "mse_non": mse_non,
        "mse_adp": mse_adp,
    }

    fname = DATA_DIR / f"fisher_{dist}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved:", str(fname))
