# =======================================================
# SIN2 Distribution Analysis and Summary Table 
# Import Dependencies
# =======================================================

from __future__ import annotations
import numpy as np
import csv
import sys
from pathlib import Path
np.random.seed(42)
# ========== Imports ==========
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import All_Schemes as AS  # unified distributions, constants, bounds, etc.
from All_Schemes import _sin2_phi_pp_unit  # explicit imports

# ========== Parameters ==========
a, b, d, omega = AS.A_SIN2, AS.B_SIN2, AS.D_SIN2, AS.W_SIN2
c_seed = AS.C_SIN2
NA_CONST = AS.NA_CONST

# Numerics
L = 12.0
N_int = 400_001
dx = 1e-4
nt = 200_001

# ========== Potential and Derivatives ==========
def compute_potential_phi(x, a, b, d, omega, c, p):
    z = x / c
    return a * (np.abs(z) ** p) + b * (z ** 4) + d * (np.sin(omega * z) ** 2) + 1.0

def compute_first_derivative_phi(x, a, b, d, omega, c, p):
    z = x / c
    term1 = (a * p / c) * np.sign(z) * (np.abs(z) ** (p - 1))
    term2 = (4 * b / (c ** 4)) * (x ** 3)
    term3 = d * (omega / c) * np.sin(2 * omega * z)
    return term1 + term2 + term3

def compute_second_derivative_phi(x, a, b, d, omega, c, p):
    z = x / c
    out = np.zeros_like(z, dtype=float)
    mask = np.abs(z) > 1e-8
    if p > 1:
        out[mask] = (a * p * (p - 1) / (c ** 2)) * (np.abs(z[mask]) ** (p - 2))
    out += (12 * b / (c ** 4)) * (x ** 2)
    out += (2 * d * (omega ** 2) / (c ** 2)) * np.cos(2 * omega * z)
    return out

# ========== Compute T of fX with x Star and h Star ==========
def compute_t_fx_with_maximizers(a, b, d, omega, c, p):
    xs = np.arange(0.0, L + dx, dx)
    fu = np.exp(-compute_potential_phi(xs, a, b, d, omega, c, p))
    Z = np.trapz(fu, xs) * 2.0
    f = fu / Z
    phi_prime = compute_first_derivative_phi(xs, a, b, d, omega, c, p)
    h = 2.0 * phi_prime * f
    h[h < 0] = 0.0

    # Find maximizer and value
    idx_star = int(np.argmax(h))
    x_star = float(xs[idx_star])
    h_star = float(h[idx_star])

    if h_star <= 0:
        return 0.0, np.nan, np.nan

    # Monotone envelope and integral for T(f)
    h_seg = h[:idx_star + 1]
    x_seg = xs[:idx_star + 1]
    h_mono = np.maximum.accumulate(h_seg)
    t_grid = np.linspace(0.0, h_star, nt)
    x_of_t = np.interp(t_grid, h_mono, x_seg)
    phi_prime_at = compute_first_derivative_phi(x_of_t, a, b, d, omega, c, p)
    T_val = float(np.trapz(phi_prime_at * x_of_t, t_grid))
    return T_val, x_star, h_star

# ========== PDF Normalization and Variance Correction ==========
def compute_unnormalized_density(x, a, b, d, omega, c, p):
    return np.exp(-compute_potential_phi(x, a, b, d, omega, c, p))

def get_normalized_density(a, b, d, omega, c, p):
    xs = np.linspace(-L, L, N_int)
    fu = compute_unnormalized_density(xs, a, b, d, omega, c, p)
    Z = np.trapz(fu, xs)
    return xs, fu / Z, Z

def compute_moments(a, b, d, omega, c, p):
    xs = np.linspace(-L, L, N_int)
    fu = compute_unnormalized_density(xs, a, b, d, omega, c, p)
    Z = np.trapz(fu, xs)
    f = fu / Z
    mean = np.trapz(xs * f, xs)
    var = np.trapz((xs - mean) ** 2 * f, xs)
    return Z, mean, var

def adjust_c_for_unit_variance(a, b, d, omega, c_seed, p):
    _, _, var0 = compute_moments(a, b, d, omega, c_seed, p)
    return c_seed / np.sqrt(max(var0, 1e-300))

# ========== Strict Log Concavity Check ==========
def check_strict_log_concavity(a, b, d, omega, c, p):
    xs = np.linspace(-L, L, N_int)
    xs = xs[np.abs(xs) > 1e-6]
    vals = compute_second_derivative_phi(xs, a, b, d, omega, c, p)
    ok_local = (np.isfinite(vals).all() and np.min(vals) > 0)
    min_local = float(np.min(vals))
    vals_as = _sin2_phi_pp_unit(xs)
    ok_as = (np.isfinite(vals_as).all() and np.min(vals_as) > 0)
    min_as = float(np.min(vals_as))
    return (ok_local and ok_as), min(min_local, min_as)

# ========== Sweep Over P Values ==========
p_values = [1.0, 1.2, 1.5, 1.7, 1.8, 2.0]
results = []

print("\n======================== Summary Table ========================")
hdr = (
    "{:<5} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} "
    "{:>10} {:>10} {:>10} {:>8} {:>20} {:>10} {:>12} {:>12}"
)
row = (
    "{:<5.1f} {:>8.4f} {:>10.6f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} "
    "{:>10.4f} {:>10.4f} {:>10.4f} {:>8} {:>20.4f} {:>10} {:>12.4f} {:>12.4f}"
)

print(
    hdr.format(
        "p", "c", "Z", "mean", "var", "f(0)", "C_adp",
        "T(f)", "C_non", "Ratio", "PDF?",
        "min phi''", "LogConc?", "x*", "h*"
    )
)
print("=" * 170)
for p in p_values:
    c = adjust_c_for_unit_variance(a, b, d, omega, c_seed, p)
    xs, fX, Z = get_normalized_density(a, b, d, omega, c, p)
    f0 = np.exp(-compute_potential_phi(0.0, a, b, d, omega, c, p)) / Z
    Zc, mean, var = compute_moments(a, b, d, omega, c, p)
    ok_slc, min_phi_pp = check_strict_log_concavity(a, b, d, omega, c, p)
    C_adp = 1 / (4 * f0 ** 2)

    # Compute T(fX), x star, h star
    T_val, x_star, h_star = compute_t_fx_with_maximizers(a, b, d, omega, c, p)
    C_non = NA_CONST / T_val if T_val > 0 else np.nan
    ratio = C_non / C_adp if C_adp > 0 else np.nan

    results.append({
        "p": p, "c": c, "Z": Z, "mean": mean, "var": var, "f0": f0,
        "C_adp": C_adp, "T_f": T_val, "C_non": C_non, "ratio": ratio,
        "PDF_valid": True, "SLC": ok_slc, "min_phi_pp": min_phi_pp,
        "x_star": x_star, "h_star": h_star
    })

    print(
        row.format(
            p, c, Z, mean, var, f0, C_adp, T_val, C_non, ratio,
            str(True), min_phi_pp, str(ok_slc), x_star, h_star
        )
    )

print("=" * 170)

