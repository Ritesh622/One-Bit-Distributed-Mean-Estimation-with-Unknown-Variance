#===================================================================================================
#  Import and  Device Set up
#===================================================================================================
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

# Ensure parent folder is importable (contains All_Schemes.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import All_Schemes as AS

#===================================================================================================
#    Reproducibility Utilities
#===================================================================================================
REPRO_MODE = True
GLOBAL_BASE_SEED = 42

def stable_seed(*xs) -> int:
    s = "|".join(map(str, xs)).encode("utf-8")
    return int.from_bytes(sha256(s).digest()[:4], "little")

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

if REPRO_MODE:
    # Note: this experiment is mostly deterministic except for binomial draws,
    # and we reseed per (mu, dist, K, n) below. So global seed is optional.
    set_all_seeds(GLOBAL_BASE_SEED)

#===================================================================================================
#    Helper Functions
#===================================================================================================
def get_cdf(dist: str):
    """Unit-variance CDF callable from All_Schemes."""
    return AS.get_unit_variance_cdf(dist)

def get_ppf(dist: str):
    """Unit-variance PPF callable from All_Schemes."""
    return AS.get_unit_variance_ppf(dist)

def bernoulli_mean(dist: str, q: float, mu: float, sigma: float, n: int, F=None) -> float:
    """Single Binomial draw for mean of 1[X<q], with p = F((q-mu)/sigma)."""
    if n <= 0:
        return np.nan
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if F is None:
        F = get_cdf(dist)
    z = (q - mu) / sigma
    p = float(np.clip(F(z), 1e-12, 1.0 - 1e-12))
    return np.random.binomial(int(n), p) / float(n)

def decode_first_from_means(dist: str, f1: float, f2: float, q1: float, q2: float, PPF=None):
    """Same algebra as AS.decode_adaptive_first_round but uses means directly."""
    if not (0.0 < f1 < 1.0 and 0.0 < f2 < 1.0):
        return None
    if PPF is None:
        PPF = get_ppf(dist)
    a1 = float(PPF(f1))
    a2 = float(PPF(f2))
    if a1 == a2:
        return None
    sigma_hat = (q1 - q2) / (a1 - a2)
    mu_hat    = (a1 * q2 - a2 * q1) / (a1 - a2)
    return float(mu_hat), float(sigma_hat)

def decode_second_from_mean(dist: str, f3: float, mu_hat: float, sigma_hat: float, PPF=None):
    """Same algebra as AS.decode_adaptive_second_round but uses mean directly."""
    if not (0.0 < f3 < 1.0):
        return None
    if PPF is None:
        PPF = get_ppf(dist)
    a3 = float(PPF(f3))
    return float(mu_hat - a3 * sigma_hat)

def compute_thresholds(mu_min: float, mu_max: float) -> tuple[float, float]:
    """Two evenly spaced thresholds in [mu_min, mu_max]."""
    theta1 = mu_min + (mu_max - mu_min) / 3.0
    theta2 = mu_min + 2.0 * (mu_max - mu_min) / 3.0
    return float(theta1), float(theta2)

def save_results(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def ksuffix(prefix: str, K1: float | None = None, K2: float | None = None) -> str:
    if K1 is None or K2 is None:
        return prefix
    return f"{prefix}_K1_{K1:.2f}_K2_{K2:.2f}".replace('.', '_')

def solve_n3_for_total_n(n: int) -> int:
    """n1=n2=n3/log n3; choose n3 so total is n."""
    if n <= 10:
        return max(3, n - 2)
    lo, hi = 3.0, float(max(4, n - 3))
    def f(x): return x * (1.0 + 2.0 / np.log(x)) - n
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < 1e-7:
            break
    n3 = int(round(0.5 * (lo + hi)))
    return max(3, n3)

#===================================================================================================
#    Directories
#===================================================================================================
current_path = pathlib.Path(__file__).parent.resolve()

# Write outputs relative to this script (NOT current working directory)
mse_data_dir    = current_path / "Avg_Worst_MSE_data"
worst_case_dir  = mse_data_dir / "Worst_Case"
avg_case_dir    = mse_data_dir / "Average_Case"
benchmark_dir   = mse_data_dir / "Benchmark"
upper_bound_dir = mse_data_dir / "Upper_Bound"

if mse_data_dir.exists():
    print(f"Flushing existing directory: {mse_data_dir}")
    shutil.rmtree(mse_data_dir)

for d in [worst_case_dir, avg_case_dir, benchmark_dir, upper_bound_dir]:
    d.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {d}")

#===================================================================================================
#    Experiment Settings
#===================================================================================================
USE_SPECIAL_ADAPTIVE = True   # n1 = n2 = n3 / log n3
dist_set = ["gaussian", "gaussian_b2", "logistic", "hypsecant", "sin2"]
#dist_set = ["gaussian"]    # for quick testing

n_max = 40000
step = 1000
total_samples = np.append(np.arange(200, n_max, step), n_max)

n_trials =   2000 #2000
mu_range = [AS.MU_MIN, AS.MU_MAX]
true_std = AS.SIGMA_GLOB
q1, q2 = compute_thresholds(*mu_range)
mu_values = np.linspace(mu_range[0], mu_range[1], 15)

#===================================================================================================
#    Non-Adaptive Block (K1 + K2 = 1) + Upper Bound
#===================================================================================================
def run_nonadaptive_block(K1: float, K2: float) -> None:
    if not np.isclose(K1 + K2, 1.0, rtol=1e-9):
        print(f"[SKIP NON-ADAPT] K1+K2 must equal 1. Got {K1}+{K2}={K1+K2:.2f}")
        return

    tag = ksuffix("NONADAPT", K1, K2)

    for decode_dist in dist_set:
        print(f"\n[NON-ADAPT] {decode_dist}  ({tag})")

        worst_mse_na, worst_mu_na = [], []
        avg_case_mse_na = []
        all_mu_data = {}

        # UB constant over mu \in [mu_min,mu_max]; n=1.0 -> returned mse == constant C
        ub_best_mu, C_ub_const = AS.compute_nonadaptive_upper_bound(
            dist=decode_dist,
            n=1.0,
            sigma=true_std,
            theta1=q1,
            theta2=q2,
            mu_min=float(mu_range[0]),
            mu_max=float(mu_range[1]),
            k1=float(K1),
            k2=float(K2),
            # n_mu=801,  # optional resolution
        )

        # cache per-dist callables once
        F = get_cdf(decode_dist)
        PPF = get_ppf(decode_dist)

        for n in total_samples:
            n1 = int(round(K1 * n))
            n2 = int(n - n1)

            if n1 <= 0 or n2 <= 0:
                print(f"[WARN NON-ADAPT] n too small at n={n}")
                worst_mse_na.append(np.nan)
                worst_mu_na.append(None)
                avg_case_mse_na.append(np.nan)
                all_mu_data[int(n)] = {}
                continue

            max_mse_na, max_mu_na_i = -np.inf, None
            sum_mse_na, cnt_mse_na = 0.0, 0
            mu_data = {}

            for mu in tqdm(mu_values, desc=f"n={n}, dec={decode_dist}, {tag}"):
                seed = stable_seed(mu, decode_dist, tag, n)
                set_all_seeds(seed)

                mse_na_trials = []
                for _ in range(n_trials):
                    f1 = bernoulli_mean(decode_dist, q1, mu, true_std, n1, F)
                    f2 = bernoulli_mean(decode_dist, q2, mu, true_std, n2, F)
                    res = decode_first_from_means(decode_dist, f1, f2, q1, q2, PPF)
                    if res:
                        est_mu, _ = res
                        mse_na_trials.append((est_mu - mu) ** 2)

                avg_mse_na = float(np.mean(mse_na_trials)) if mse_na_trials else float('nan')
                if np.isfinite(avg_mse_na):
                    if avg_mse_na > max_mse_na:
                        max_mse_na, max_mu_na_i = avg_mse_na, float(mu)
                    sum_mse_na += avg_mse_na
                    cnt_mse_na += 1

                mu_data[(float(mu), decode_dist)] = {'nonadaptive_avg_mse': avg_mse_na}

            if not np.isfinite(max_mse_na):
                max_mse_na, max_mu_na_i = np.nan, None

            worst_mse_na.append(max_mse_na)
            worst_mu_na.append(max_mu_na_i)
            avg_case_mse_na.append((sum_mse_na / cnt_mse_na) if cnt_mse_na > 0 else np.nan)
            all_mu_data[int(n)] = mu_data

        # Save empirical results
        save_results({
            'samples': total_samples.tolist(),
            'nonadaptive_worst_mse': worst_mse_na,
            'nonadaptive_worst_mu':  worst_mu_na,
            'K1': K1, 'K2': K2, 'all_mu_encode_data': all_mu_data
        }, worst_case_dir / f"{decode_dist}_{tag}_worst_case.pkl")

        save_results({
            'samples': total_samples.tolist(),
            'nonadaptive_average_mse': avg_case_mse_na,
            'K1': K1, 'K2': K2, 'all_mu_encode_data': all_mu_data
        }, avg_case_dir / f"{decode_dist}_{tag}_average_case.pkl")

        # Save UB curve as C/n and metadata
        save_results({
            'samples': total_samples.tolist(),
            'mse_upper_bound': [float(C_ub_const) / float(n) for n in total_samples],
            'C_nonadaptive_ub': float(C_ub_const),
            'best_mu_ub': float(ub_best_mu),
            'K1': K1, 'K2': K2,
            'theta1': float(q1), 'theta2': float(q2),
            'mu_range': (float(mu_range[0]), float(mu_range[1])),
            'dist': decode_dist
        }, upper_bound_dir / f"{decode_dist}_{tag}_nonadaptive_ub.pkl")

        print(f"[OK NON-ADAPT] UB saved (C={C_ub_const:.6e} at mu*={ub_best_mu:.3f}); empirical curves saved.")

#===================================================================================================
#    Adaptive Block (K1 + K2 < 1, n3 = n - n1 - n2)
#===================================================================================================
def run_adaptive_block(K1: float, K2: float) -> None:
    if (K1 <= 0) or (K2 <= 0) or (K1 + K2 >= 1.0):
        print(f"[SKIP ADAPT] Need K1>0, K2>0, K1+K2<1. Got K1={K1}, K2={K2}")
        return

    tag = ksuffix("ADAPT", K1, K2)

    for decode_dist in dist_set:
        print(f"\n[ADAPT] {decode_dist}  ({tag})")

        worst_mse_ad, worst_mu_ad = [], []
        avg_case_mse_ad = []
        all_mu_data = {}

        # cache per-dist callables once
        F = get_cdf(decode_dist)
        PPF = get_ppf(decode_dist)

        for n in total_samples:
            n1 = int(round(K1 * n))
            n2 = int(round(K2 * n))
            n3 = int(n - n1 - n2)

            if n1 <= 0 or n2 <= 0 or n3 <= 0:
                print(f"[WARN ADAPT] n too small at n={n}")
                worst_mse_ad.append(np.nan); worst_mu_ad.append(None)
                avg_case_mse_ad.append(np.nan); all_mu_data[int(n)] = {}
                continue

            max_mse_ad, max_mu_ad_i = -np.inf, None
            sum_mse_ad, cnt_mse_ad = 0.0, 0
            mu_data = {}

            for mu in tqdm(mu_values, desc=f"n={n}, dec={decode_dist}, {tag}"):
                seed = stable_seed(mu, decode_dist, tag, n)
                set_all_seeds(seed)

                mse_ad_trials = []
                for _ in range(n_trials):
                    f1 = bernoulli_mean(decode_dist, q1, mu, true_std, n1, F)
                    f2 = bernoulli_mean(decode_dist, q2, mu, true_std, n2, F)
                    res = decode_first_from_means(decode_dist, f1, f2, q1, q2, PPF)
                    if res is None:
                        continue
                    mu_hat, sigma_hat = res
                    f3 = bernoulli_mean(decode_dist, mu_hat, mu, true_std, n3, F)  # refine at q=mu_hat
                    est_mu_final = decode_second_from_mean(decode_dist, f3, mu_hat, sigma_hat, PPF)
                    if est_mu_final is not None:
                        mse_ad_trials.append((est_mu_final - mu) ** 2)

                avg_mse_ad = float(np.mean(mse_ad_trials)) if mse_ad_trials else float('nan')
                if np.isfinite(avg_mse_ad):
                    if avg_mse_ad > max_mse_ad:
                        max_mse_ad, max_mu_ad_i = avg_mse_ad, float(mu)
                    sum_mse_ad += avg_mse_ad
                    cnt_mse_ad += 1

                mu_data[(float(mu), decode_dist)] = {'adaptive_avg_mse': avg_mse_ad}

            if not np.isfinite(max_mse_ad):
                max_mse_ad, max_mu_ad_i = np.nan, None

            worst_mse_ad.append(max_mse_ad)
            worst_mu_ad.append(max_mu_ad_i)
            avg_case_mse_ad.append((sum_mse_ad / cnt_mse_ad) if cnt_mse_ad > 0 else np.nan)
            all_mu_data[int(n)] = mu_data

        save_results({
            'samples': total_samples.tolist(),
            'adaptive_worst_mse': worst_mse_ad,
            'adaptive_worst_mu':  worst_mu_ad,
            'K1': K1, 'K2': K2, 'all_mu_encode_data': all_mu_data
        }, worst_case_dir / f"{decode_dist}_{tag}_worst_case.pkl")

        save_results({
            'samples': total_samples.tolist(),
            'adaptive_average_mse': avg_case_mse_ad,
            'K1': K1, 'K2': K2, 'all_mu_encode_data': all_mu_data
        }, avg_case_dir / f"{decode_dist}_{tag}_average_case.pkl")

        print(f"[OK ADAPT] Saved for {decode_dist} ({tag})")

#===================================================================================================
#    Adaptive Special Schedule: n1 = n2 = n3 / log n3
#===================================================================================================
def run_adaptive_special_block() -> None:
    tag = "ADAPT_SPECIAL"

    for decode_dist in dist_set:
        print(f"\n[ADAPT SPECIAL] {decode_dist}  ({tag})")

        worst_mse_ad, worst_mu_ad = [], []
        avg_case_mse_ad = []
        all_mu_data = {}

        # cache per-dist callables once
        F = get_cdf(decode_dist)
        PPF = get_ppf(decode_dist)

        for n in total_samples:
            n3 = solve_n3_for_total_n(int(n))
            n1 = int(round(n3 / np.log(max(n3, 3))))
            n2 = n1

            total = n1 + n2 + n3
            if total != n:
                n3 = max(1, n3 + (n - total))

            if n1 <= 0 or n2 <= 0 or n3 <= 0:
                print(f"[WARN ADAPT SPECIAL] invalid split at n={n} -> ({n1},{n2},{n3})")
                worst_mse_ad.append(np.nan)
                worst_mu_ad.append(None)
                avg_case_mse_ad.append(np.nan)
                all_mu_data[int(n)] = {}
                continue

            max_mse_ad, max_mu_ad_i = -np.inf, None
            sum_mse_ad, cnt_mse_ad = 0.0, 0
            mu_data = {}

            for mu in tqdm(mu_values, desc=f"n={n}, dec={decode_dist}, {tag}"):
                seed = stable_seed(mu, decode_dist, tag, n)
                set_all_seeds(seed)

                mse_ad_trials = []
                for _ in range(n_trials):
                    f1 = bernoulli_mean(decode_dist, q1, mu, true_std, n1, F)
                    f2 = bernoulli_mean(decode_dist, q2, mu, true_std, n2, F)
                    res = decode_first_from_means(decode_dist, f1, f2, q1, q2, PPF)
                    if res is None:
                        continue
                    mu_hat, sigma_hat = res
                    f3 = bernoulli_mean(decode_dist, mu_hat, mu, true_std, n3, F)
                    est_mu_final = decode_second_from_mean(decode_dist, f3, mu_hat, sigma_hat, PPF)
                    if est_mu_final is not None:
                        mse_ad_trials.append((est_mu_final - mu) ** 2)

                avg_mse_ad = float(np.mean(mse_ad_trials)) if mse_ad_trials else float('nan')
                if np.isfinite(avg_mse_ad):
                    if avg_mse_ad > max_mse_ad:
                        max_mse_ad, max_mu_ad_i = avg_mse_ad, float(mu)
                    sum_mse_ad += avg_mse_ad
                    cnt_mse_ad += 1

                mu_data[(float(mu), decode_dist)] = {'adaptive_avg_mse': avg_mse_ad}

            if not np.isfinite(max_mse_ad):
                max_mse_ad, max_mu_ad_i = np.nan, None

            worst_mse_ad.append(max_mse_ad)
            worst_mu_ad.append(max_mu_ad_i)
            avg_case_mse_ad.append((sum_mse_ad / cnt_mse_ad) if cnt_mse_ad > 0 else np.nan)
            all_mu_data[int(n)] = mu_data

        save_results({
            'samples': total_samples.tolist(),
            'adaptive_worst_mse': worst_mse_ad,
            'adaptive_worst_mu':  worst_mu_ad,
            'special': True, 'all_mu_encode_data': all_mu_data
        }, worst_case_dir / f"{decode_dist}_{tag}_worst_case.pkl")

        save_results({
            'samples': total_samples.tolist(),
            'adaptive_average_mse': avg_case_mse_ad,
            'special': True, 'all_mu_encode_data': all_mu_data
        }, avg_case_dir / f"{decode_dist}_{tag}_average_case.pkl")

        print(f"[OK ADAPT SPECIAL] Saved for {decode_dist} ({tag})")

#===================================================================================================
#    Run All Schedules
#===================================================================================================
for K1, K2 in AS.K_CONFIGS_NONADAPTIVE:
    run_nonadaptive_block(K1, K2)

for K1, K2 in AS.K_CONFIGS_ADAPTIVE:
    run_adaptive_block(K1, K2)

if USE_SPECIAL_ADAPTIVE:
    run_adaptive_special_block()

#===================================================================================================
#    Lower Bounds (independent of K)
#===================================================================================================
CENTRAL_TRUE_STD = 2.0
for encode_dist in dist_set:
    C_adapt = AS.compute_adaptive_lower_bound(encode_dist, CENTRAL_TRUE_STD)
    save_results({
        "samples": total_samples.tolist(),
        "mse": [C_adapt / n for n in total_samples],
        "C_adapt": C_adapt
    }, benchmark_dir / f"{encode_dist}_benchmark.pkl")

    C_nonad = AS.compute_nonadaptive_lower_bound(encode_dist, CENTRAL_TRUE_STD)
    save_results({
        "samples": total_samples.tolist(),
        "mse": [C_nonad / n for n in total_samples],
        "C_nonadaptive": C_nonad
    }, benchmark_dir / f"{encode_dist}_nonadaptive_lb.pkl")

    rel = "<" if C_nonad < C_adapt else "≥"
    ratio = (C_nonad / C_adapt) if C_adapt > 0 else float('inf')
    print(f"[INFO] {encode_dist}: C_nonadaptive={C_nonad:.6f} {rel} C_adaptive={C_adapt:.6f}, C_non/C_adapt={ratio:.6f}")
