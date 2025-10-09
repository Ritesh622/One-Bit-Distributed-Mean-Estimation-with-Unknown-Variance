# ===================================================================================================
# Imports and Device Setup
# ===================================================================================================
from __future__ import annotations
import numpy as np
import random
import os
import pickle
import pathlib
import sys
import shutil
from tqdm import tqdm
from hashlib import sha256
from pathlib import Path

# --- Path hack: ensure parent folder (Global_All_Schemes) is importable ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import All_Schemes as AS  # uses GGD(beta=1.5) for "gaussian" and p=1.5 tuned sin2

# ===================================================================================================
# Reproducibility Utilities
# ===================================================================================================
REPRO_MODE = True
GLOBAL_BASE_SEED = 42

def stable_seed(*xs) -> int:
    s = "|".join(map(str, xs)).encode("utf-8")
    return int.from_bytes(sha256(s).digest()[:4], "little")

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

# Avoid torch entirely in the inner loop; determinism via numpy RNG is enough.
if REPRO_MODE:
    pass  # keep placeholder to mirror previous structure

# ===================================================================================================
# Utility Functions 
# ===================================================================================================
def bernoulli_mean(dist: str, q: float, mu: float, sigma: float, n: int) -> float:
    """
    Draw f = (1/n) * sum 1[X<q] by sampling directly from Bernoulli(n, p),
    where p = F((q - mu)/sigma).  Faster than sampling X.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    F = AS.get_unit_variance_cdf(dist)  # unit-variance CDF
    z = (q - mu) / sigma
    p = float(np.clip(F(z), 1e-12, 1.0 - 1e-12))
    # crucial: simulate count once; no arrays
    k = np.random.binomial(int(n), p)
    return k / float(n)

def decode_first_round_from_means(dist: str, f1: float, f2: float, q1: float, q2: float):
    """
    Same math as AS.decode_adaptive_first_round but uses means directly.
    """
    if not (0.0 < f1 < 1.0 and 0.0 < f2 < 1.0):
        return None
    ppf = AS.get_unit_variance_ppf(dist)
    a1 = float(ppf(f1))
    a2 = float(ppf(f2))
    if a1 == a2:
        return None
    sigma_hat = (q1 - q2) / (a1 - a2)
    mu_hat = (a1 * q2 - a2 * q1) / (a1 - a2)
    return float(mu_hat), float(sigma_hat)

def decode_second_round_from_mean(dist: str, f3: float, mu_hat: float, sigma_hat: float):
    """
    Same math as AS.decode_adaptive_second_round but uses mean directly.
    """
    if not (0.0 < f3 < 1.0):
        return None
    ppf = AS.get_unit_variance_ppf(dist)
    a3 = float(ppf(f3))
    return float(mu_hat - a3 * sigma_hat)

def calculate_thresholds(a: float, b: float) -> tuple[float, float]:
    """Two evenly spaced thresholds in [a,b]: a+(b-a)/3 and a+2(b-a)/3."""
    return a + (b - a) / 3.0, a + 2.0 * (b - a) / 3.0

def calculate_adaptive_sample_sizes(n: int) -> tuple[int, int, int]:
    """
    Split n into (n1, n2, n3) for 2 coarse rounds + 1 refine round.
    """
    if n < 3:
        raise ValueError("Total samples n must be at least 3")
    log_n = np.log(n)
    n3 = max(1, int(np.round(n * log_n / (2 + log_n))))
    n1 = n2 = max(1, int(np.round(n3 / np.log(max(n3, np.e)))))
    total = n1 + n2 + n3
    if total != n:
        n3 = n - n1 - n2
    return n1, n2, n3

def save_results(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)

# ===================================================================================================
# Directory Setup
# ===================================================================================================
current_path = pathlib.Path(__file__).parent.resolve()
parent_dir = os.getcwd()

mse_data_dir = os.path.join(parent_dir, 'Avg_Worst_MSE_data')
worst_case_dir = os.path.join(mse_data_dir, 'Worst_Case')
avg_case_dir = os.path.join(mse_data_dir, 'Average_Case')
benchmark_dir = os.path.join(mse_data_dir, 'Benchmark')
upper_bound_dir = os.path.join(mse_data_dir, 'Upper_Bound')

if os.path.exists(mse_data_dir):
    print(f"Flushing existing directory: {mse_data_dir}")
    shutil.rmtree(mse_data_dir)

for d in [worst_case_dir, avg_case_dir, benchmark_dir, upper_bound_dir]:
    os.makedirs(d, exist_ok=True)
    print(f"Created directory: {d}")

# ===================================================================================================
# Experiment Parameters
# ===================================================================================================
dist_set = ["gaussian","gaussian_b2", "hypsecant", "logistic", "sin2"]
#dist_set = ["gaussian"]  # for quick testing
n_max = 40000
step = 1000
total_samples = np.arange(200, n_max, step)
total_samples = np.append(total_samples, n_max)

n_trials = 2000
mu_range = AS.MU_MIN, AS.MU_MAX
true_sigma = AS.SIGMA_GLOB
q1, q2 = calculate_thresholds(*mu_range)
mu_values = np.linspace(mu_range[0], mu_range[1], 10)

# ===================================================================================================
# Main Experiment Loop
# ===================================================================================================
for decode_dist in dist_set:
    print(f"\nRunning for decode_dist = {decode_dist}")
    worst_mse_na, worst_mse_ad = [], []
    worst_mu_na, worst_mu_ad = [], []
    avg_case_mse_na, avg_case_mse_ad = [], []
    all_mu_data = {}

    for n in total_samples:
        max_mse_na = -np.inf
        max_mse_ad = -np.inf
        max_mu_na = max_mu_ad = None
        sum_mse_na = sum_mse_ad = 0.0
        cnt_mse_na = cnt_mse_ad = 0
        mu_data = {}

        for mu in tqdm(mu_values, desc=f"n={n}, dec={decode_dist}"):
            seed = stable_seed(mu, decode_dist, int(n))
            set_all_seeds(seed)

            mse_na_trials, mse_ad_trials = [], []

            # --- Non-adaptive (two thresholds once) ---
            half_n = int(n // 2)
            for _ in range(n_trials):
                f1 = bernoulli_mean(decode_dist, q1, mu, true_sigma, half_n)
                f2 = bernoulli_mean(decode_dist, q2, mu, true_sigma, half_n)
                res = decode_first_round_from_means(decode_dist, f1, f2, q1, q2)
                if res is None:
                    continue
                est_mu, _ = res
                mse_na_trials.append((est_mu - mu) ** 2)
            avg_mse_na = float(np.mean(mse_na_trials)) if mse_na_trials else float('nan')

            if np.isfinite(avg_mse_na):
                if avg_mse_na > max_mse_na:
                    max_mse_na, max_mu_na = avg_mse_na, float(mu)
                sum_mse_na += avg_mse_na
                cnt_mse_na += 1

            # --- Adaptive (n1, n2, n3) ---
            try:
                n1, n2, n3 = calculate_adaptive_sample_sizes(int(n))
            except ValueError:
                avg_mse_ad = float('nan')
            else:
                for _ in range(n_trials):
                    f1 = bernoulli_mean(decode_dist, q1, mu, true_sigma, n1)
                    f2 = bernoulli_mean(decode_dist, q2, mu, true_sigma, n2)
                    res = decode_first_round_from_means(decode_dist, f1, f2, q1, q2)
                    if res is None:
                        continue
                    mu_hat, sigma_hat = res
                    # refine: threshold at mu_hat
                    f3 = bernoulli_mean(decode_dist, mu_hat, mu, true_sigma, n3)
                    est_mu_final = decode_second_round_from_mean(decode_dist, f3, mu_hat, sigma_hat)
                    if est_mu_final is not None:
                        mse_ad_trials.append((est_mu_final - mu) ** 2)
                avg_mse_ad = float(np.mean(mse_ad_trials)) if mse_ad_trials else float('nan')

            if np.isfinite(avg_mse_ad):
                if avg_mse_ad > max_mse_ad:
                    max_mse_ad, max_mu_ad = avg_mse_ad, float(mu)
                sum_mse_ad += avg_mse_ad
                cnt_mse_ad += 1

            mu_data[(float(mu), decode_dist)] = {
                'nonadaptive_avg_mse': avg_mse_na,
                'adaptive_avg_mse': avg_mse_ad
            }

        # Worst-case across mean
        worst_mse_na.append(max_mse_na if np.isfinite(max_mse_na) else np.nan)
        worst_mu_na.append(max_mu_na)
        worst_mse_ad.append(max_mse_ad if np.isfinite(max_mse_ad) else np.nan)
        worst_mu_ad.append(max_mu_ad)

        # Average-case across mean
        avg_case_mse_na.append(sum_mse_na / cnt_mse_na if cnt_mse_na > 0 else np.nan)
        avg_case_mse_ad.append(sum_mse_ad / cnt_mse_ad if cnt_mse_ad > 0 else np.nan)
        all_mu_data[int(n)] = mu_data

    # Save worst-case results
    save_results({
        'samples': total_samples.tolist(),
        'nonadaptive_worst_mse': worst_mse_na,
        'adaptive_worst_mse': worst_mse_ad,
        'nonadaptive_worst_mu': worst_mu_na,
        'adaptive_worst_mu': worst_mu_ad,
        'all_mu_encode_data': all_mu_data
    }, os.path.join(worst_case_dir, f"{decode_dist}_worst_case.pkl"))

    # Save average-case results
    save_results({
        'samples': total_samples.tolist(),
        'nonadaptive_average_mse': avg_case_mse_na,
        'adaptive_average_mse': avg_case_mse_ad,
        'all_mu_encode_data': all_mu_data
    }, os.path.join(avg_case_dir, f"{decode_dist}_average_case.pkl"))

    # --- Save UB curve (max over mu in [mu_min, mu_max]) ---
    best_mu, C_ub = AS.compute_nonadaptive_upper_bound(
        dist=decode_dist,
        n=1.0,  # UB constant
        sigma=true_sigma,
        theta1=q1,
        theta2=q2,
        mu_min=float(mu_range[0]),
        mu_max=float(mu_range[1]),
        k1=0.5,
        k2=0.5,
        # n_mu=101,  
    )

    save_results({
        "samples": total_samples.tolist(),
        "mse_upper_bound": [float(C_ub) / float(n) for n in total_samples],
        "C_nonadaptive_ub": float(C_ub),
        "best_mu": float(best_mu),
        "theta1": float(q1), "theta2": float(q2),
        "mu_range": (float(mu_range[0]), float(mu_range[1])),
        "K1": 0.5, "K2": 0.5,
        "dist": decode_dist,
    }, os.path.join(upper_bound_dir, f"{decode_dist}_nonadaptive_ub.pkl"))

# ===================================================================================================
# Benchmarks (lower bounds)  — compute T(f) with a coarser dx for speed
# ===================================================================================================
for encode_dist in dist_set:
    t_fast = AS.compute_t_fx(encode_dist, dx=1e-3)  # coarser grid \geq  big speedup
    C_adapt = AS.compute_adaptive_lower_bound(encode_dist, true_sigma)
    C_nonad = AS.compute_nonadaptive_lower_bound(encode_dist, true_sigma, override_t=t_fast)

    save_results({
        "samples": total_samples.tolist(),
        "mse": [C_adapt / n for n in total_samples],
        "C_adapt": C_adapt
    }, os.path.join(benchmark_dir, f"{encode_dist}_benchmark.pkl"))

    save_results({
        "samples": total_samples.tolist(),
        "mse": [C_nonad / n for n in total_samples],
        "C_nonadaptive": C_nonad
    }, os.path.join(benchmark_dir, f"{encode_dist}_nonadaptive_lb.pkl"))
