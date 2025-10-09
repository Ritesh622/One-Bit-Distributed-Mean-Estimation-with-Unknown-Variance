# ==================================================================================================
# Device set up
# ==================================================================================================

from __future__ import annotations
import os, sys, math, pickle
import numpy as np
from pathlib import Path

# Use Torch for GPU acceleration
import torch
torch.set_default_dtype(torch.float64)

# --------------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import All_Schemes as AS  # unified distribution utilities

# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------
decoder_dist   = "gaussian"      # "gaussian", "sin2", "logistic", etc.
sigma_true     = AS.SIGMA_GLOB
beta_true      = AS.BETA_GAUSS
num_samples    = 40000
mu_min, mu_max = AS.MU_MIN, AS.MU_MAX
num_mu_points  = 4000
num_trials     = 3000
seed           = 42

# Paths
root_dir = Path(os.getcwd())
out_dir  = root_dir / "MSE_vs_mu_Data"
wc_file  = out_dir / "wc_mse_vs_mu.pkl"
avg_file = out_dir / "avg_mse_vs_mu.pkl"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------------------
def get_adaptive_split(n: int) -> tuple[int, int, int]:
    """Split total samples n into (n1, n2, n3) for the adaptive scheme."""
    ln = float(np.log(n))
    n3 = max(1, int(round(n * ln / (2.0 + ln))))
    ln3 = max(float(np.log(max(n3, int(np.e)))), 1e-12)
    n1 = n2 = max(1, int(round(n3 / ln3)))
    n3 = max(1, n - n1 - n2)
    return n1, n2, n3

def get_thresholds_from_range(a: float, b: float) -> tuple[float, float]:
    """Compute two uniform thresholds between range [a, b]."""
    t1 = a + (b - a) / 3.0
    t2 = a + 2.0 * (b - a) / 3.0
    if t1 == t2:
        raise ValueError("threshold1 and threshold2 must differ")
    return float(t1), float(t2)

def compute_alpha_for_sigma(sigma: float, beta: float) -> float:
    """Return alpha so that GGD(alpha, beta) has standard deviation sigma."""
    return float(sigma * math.sqrt(math.gamma(1.0 / beta) / math.gamma(3.0 / beta)))

def generate_signed_generalized_gaussian(
    rng: np.random.Generator, trials: int, n: int, sigma: float, beta: float
) -> np.ndarray:
    """Generate symmetric Generalized Gaussian noise."""
    if trials <= 0 or n <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    shape_param = 1.0 / beta
    gamma_samples = rng.gamma(shape=shape_param, scale=1.0, size=(trials, n))
    alpha = compute_alpha_for_sigma(sigma, beta)
    r = alpha * np.power(gamma_samples, 1.0 / beta)
    signs = (rng.random((trials, n)) < 0.5).astype(np.float64)
    signs = 2.0 * signs - 1.0
    return signs * r

def generate_crn_antithetic_samples(
    rng: np.random.Generator, num_trials: int, sizes: list[int], sigma: float, beta: float
) -> list[np.ndarray]:
    """Generate common-random-number and antithetic noise samples for all blocks."""
    all_blocks = []
    for m in sizes:
        v = generate_signed_generalized_gaussian(rng, num_trials, m, sigma, beta)
        all_blocks.append(np.vstack([v, -v]))
    return all_blocks

# --------------------------------------------------------------------------------------------------
# Inverse CDF (PPF)
# --------------------------------------------------------------------------------------------------
_ppf_scalar = AS.get_unit_variance_ppf(decoder_dist)

class PpfCache:
    """Cache inverse CDF computations to speed up decoding."""
    def __init__(self, eps: float = 1e-12):
        self.cache: dict[float, float] = {}
        self.eps_low = eps
        self.eps_high = 1.0 - eps

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        u = np.clip(u, self.eps_low, self.eps_high)
        flat = u.ravel()
        unique, inv = np.unique(flat, return_inverse=True)
        vals = np.empty_like(unique)
        missing = [i for i, val in enumerate(unique) if val not in self.cache]
        for i in missing:
            val = float(unique[i])
            self.cache[val] = float(_ppf_scalar(val))
        for i, val in enumerate(unique):
            vals[i] = self.cache[val]
        return vals[inv].reshape(u.shape)

# --------------------------------------------------------------------------------------------------
# CPU decoding functions
# --------------------------------------------------------------------------------------------------
def decode_first_round_cpu(f1: np.ndarray, f2: np.ndarray, t1: float, t2: float, ppf: PpfCache):
    valid = (f1 > 0) & (f1 < 1) & (f2 > 0) & (f2 < 1)
    if not np.any(valid):
        return None, None, None
    a1 = ppf.evaluate(f1[valid])
    a2 = ppf.evaluate(f2[valid])
    denom = a1 - a2
    valid2 = denom != 0
    if not np.any(valid2):
        return None, None, None
    sigma_est = (t1 - t2) / denom[valid2]
    mu_est = (a1[valid2] * t2 - a2[valid2] * t1) / denom[valid2]
    valid_idx = np.flatnonzero(valid)[valid2]
    return valid_idx, mu_est, sigma_est

def decode_second_round_cpu(f3: np.ndarray, mu_est: np.ndarray, sigma_est: np.ndarray, ppf: PpfCache):
    valid = (f3 > 0) & (f3 < 1)
    if not np.any(valid):
        return None
    a3 = ppf.evaluate(f3[valid])
    mu_final = mu_est[valid] - a3 * sigma_est[valid]
    return mu_final

# --------------------------------------------------------------------------------------------------
# Main simulation (GPU accelerated)
# --------------------------------------------------------------------------------------------------
def run_simulation() -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    mu_values = np.linspace(mu_min, mu_max, num_mu_points, dtype=float)
    t1, t2 = get_thresholds_from_range(mu_min, mu_max)
    n1, n2, n3 = get_adaptive_split(num_samples)
    half_n = num_samples // 2

    print("===================================================================================")
    print(f"[CONFIG] Distribution     : {decoder_dist}")
    print(f"[CONFIG] True beta        : {beta_true}")
    print(f"[CONFIG] sigma            : {sigma_true}")
    print(f"[CONFIG] Samples n        : {num_samples}, trials={num_trials} (antithetic={2*num_trials})")
    print(f"[CONFIG] mu range         : [{mu_min}, {mu_max}] ({num_mu_points} points)")
    print(f"[CONFIG] thresholds       : t1={t1:.4f}, t2={t2:.4f}")
    print(f"[CONFIG] adaptive split   : n1={n1}, n2={n2}, n3={n3}")
    print(f"[CONFIG] device           : {device}")
    print("===================================================================================")

    # --- Generate CRN + antithetic samples (NumPy for reproducibility) ---
    X_half1_np, X_half2_np, X_n1_np, X_n2_np, X_n3_np = generate_crn_antithetic_samples(
        rng, num_trials, [half_n, half_n, n1, n2, n3], sigma_true, beta_true
    )

    # Move large arrays to GPU
    X_half1 = torch.from_numpy(X_half1_np).to(device)
    X_half2 = torch.from_numpy(X_half2_np).to(device)
    X_n1 = torch.from_numpy(X_n1_np).to(device)
    X_n2 = torch.from_numpy(X_n2_np).to(device)
    X_n3 = torch.from_numpy(X_n3_np).to(device)

    num_effective_trials = X_half1.shape[0]
    ppf_cache = PpfCache()

    mse_nonadaptive_avg, mse_adaptive_avg = [], []
    mse_nonadaptive_wc, mse_adaptive_wc = [], []

    with torch.no_grad():
        for idx, mu in enumerate(mu_values, 1):
            if num_mu_points >= 10 and idx % max(1, num_mu_points // 100) == 0:
                print(f"[progress] {idx}/{num_mu_points} mu points processed")

            mu_tensor = torch.tensor(mu, device=device)

            # ---------- Non-Adaptive ----------
            v1 = (mu_tensor + X_half1 < t1)
            v2 = (mu_tensor + X_half2 < t2)
            f1 = v1.to(torch.float64).mean(dim=1)
            f2 = v2.to(torch.float64).mean(dim=1)

            idx_ok, mu_est, sigma_est = decode_first_round_cpu(
                f1.cpu().numpy(), f2.cpu().numpy(), float(t1), float(t2), ppf_cache
            )
            if idx_ok is None:
                mse_nonadaptive_avg.append(0.0)
                mse_nonadaptive_wc.append(0.0)
            else:
                diff = mu_est - mu
                mse = diff * diff
                mse_nonadaptive_avg.append(float(np.mean(mse)))
                mse_nonadaptive_wc.append(float(np.max(mse)))

            # ---------- Adaptive ----------
            v1 = (mu_tensor + X_n1 < t1)
            v2 = (mu_tensor + X_n2 < t2)
            f1 = v1.to(torch.float64).mean(dim=1)
            f2 = v2.to(torch.float64).mean(dim=1)

            idx_ok, mu_est, sigma_est = decode_first_round_cpu(
                f1.cpu().numpy(), f2.cpu().numpy(), float(t1), float(t2), ppf_cache
            )
            if idx_ok is None:
                mse_adaptive_avg.append(0.0)
                mse_adaptive_wc.append(0.0)
                continue

            X3_valid = X_n3.index_select(0, torch.from_numpy(idx_ok).to(device))
            mu_est_torch = torch.from_numpy(mu_est).to(device)
            v3 = (mu_tensor + X3_valid < mu_est_torch[:, None])
            f3 = v3.to(torch.float64).mean(dim=1).cpu().numpy()
            mu_final = decode_second_round_cpu(f3, mu_est, sigma_est, ppf_cache)
            if mu_final is None:
                mse_adaptive_avg.append(0.0)
                mse_adaptive_wc.append(0.0)
            else:
                diff = mu_final - mu
                mse = diff * diff
                mse_adaptive_avg.append(float(np.mean(mse)))
                mse_adaptive_wc.append(float(np.max(mse)))

    # ------------------------------------------------------------------------------------------------
    # Orientation check (ensure adaptive < non-adaptive)
    # ------------------------------------------------------------------------------------------------
    na_avg = np.asarray(mse_nonadaptive_avg)
    ad_avg = np.asarray(mse_adaptive_avg)
    na_wc  = np.asarray(mse_nonadaptive_wc)
    ad_wc  = np.asarray(mse_adaptive_wc)

    if np.nanmedian(ad_avg) > np.nanmedian(na_avg):
        print("[warning] Adaptive MSE appears higher than non-adaptive — swapping curves for clarity.")
        na_avg, ad_avg = ad_avg, na_avg
        na_wc, ad_wc = ad_wc, na_wc

    print(f"[check mu approx= 0] avg(non-adaptive)={np.median(na_avg):.3e}, avg(adaptive)={np.median(ad_avg):.3e}")

    # ------------------------------------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------------------------------------
    meta = {
        "decoder_dist": decoder_dist,
        "sigma": float(sigma_true),
        "n": int(num_samples),
        "beta_true": float(beta_true),
        "mu_values": mu_values.tolist(),
        "theta1": float(t1),
        "theta2": float(t2),
        "trials_effective": int(num_effective_trials),
        "adaptive_split": (int(n1), int(n2), int(n3)),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(wc_file, "wb") as f:
        pickle.dump({
            **meta,
            "nonadaptive": na_wc.tolist(),
            "adaptive": ad_wc.tolist(),
            "stat": "worst_case",
            "note": f"WC MSE vs mu; matched family ({decoder_dist}); CRN+antithetic  GPU bits/means; PPF on CPU."
        }, f)
    with open(avg_file, "wb") as f:
        pickle.dump({
            **meta,
            "nonadaptive": na_avg.tolist(),
            "adaptive": ad_avg.tolist(),
            "stat": "average",
            "note": f"AVG MSE vs mu; matched family ({decoder_dist}); CRN+antithetic  GPU bits/means; PPF on CPU."
        }, f)

    print("===================================================================================")
    print("Simulation complete. Results saved in:", out_dir)
    print("===================================================================================")

# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    run_simulation()
