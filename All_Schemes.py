#===================================================================================================
#     Unified Distributions, Sampling, Encoding/Decoding, Bounds
#===================================================================================================
from __future__ import annotations

#===================================================================================================
#    Imports and Types
#===================================================================================================
from typing import Callable, Dict, Literal, Optional, Tuple, List, Union
import numpy as np
from scipy import special as sp
from scipy.stats import norm, logistic, hypsecant

#===================================================================================================
#    Constants and Global Settings
#===================================================================================================
DistName = Literal["gaussian", "gaussian_b2", "logistic", "hypsecant", "sin2"]
SIGMA_GLOB  = 2.0
NA_CONST = 0.1034                        # Non-adaptive constant
BETA_GAUSS = 1.5                         # Shape parameter for GGD-style "gaussian"
BETA_GAUSS_2 = 2
MU_MIN, MU_MAX = -2.5, 2.5
S_UNIT_LOGISTIC = float(np.sqrt(3.0) / np.pi)

# --- Tuned Sin2 parameters (unit-variance constructed via table scaling) ---
A_SIN2 = 1.48
B_SIN2 = 0.5
D_SIN2 = 0.0675
W_SIN2 = 4.0
C_SIN2 = 2.023076     # tuned scaling
P_SIN2 = 1.5          # exponent on |x/c|

#===================================================================================================
#    Generalized Gaussian (GGD, beta = 1.5 / 2.0) Utilities (unit variance)
#===================================================================================================
def compute_ggd_scale(beta: float = BETA_GAUSS) -> float:
    """Scale alpha for a unit-variance GGD (Var=1)."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    return float(np.sqrt(sp.gamma(1.0 / beta) / sp.gamma(3.0 / beta)))

def ggd_pdf(x: Union[np.ndarray, float], beta: float = BETA_GAUSS) -> Union[np.ndarray, float]:
    """Unit-variance GGD PDF with shape beta."""
    alpha = compute_ggd_scale(beta)
    coef = beta / (2.0 * alpha * sp.gamma(1.0 / beta))
    return coef * np.exp(-(np.abs(x) / alpha) ** beta)

def build_ggd_table(beta: float = BETA_GAUSS, x_max: float = 20.0, dx: float = 1e-4) -> Dict[str, np.ndarray]:
    """High-resolution lookup for GGD(beta) at unit variance."""
    if dx <= 0:
        raise ValueError("dx must be positive")
    x = np.arange(-x_max, x_max + dx, dx, dtype=np.float64)
    p = ggd_pdf(x, beta=beta)
    Z = float(np.trapz(p, x))
    if not np.isclose(Z, 1.0, rtol=1e-4, atol=1e-6):
        p = p / Z
    F = np.cumsum(p) * dx
    F[0], F[-1] = 0.0, 1.0
    return {"x": x, "pdf": p, "cdf": F}

#===============================================================
# --- Build tables for beta =1.5 and beta =2.0 ---
#================================================================
GGD_TABLE_B15 = build_ggd_table(beta=BETA_GAUSS)
GGD_TABLE_B2  = build_ggd_table(beta=BETA_GAUSS_2)

#============= beta =1.5 =========================
def ggd_unit_pdf_from_table():
    x, p = GGD_TABLE_B15["x"], GGD_TABLE_B15["pdf"]
    def pdf(y: float) -> float:
        return float(np.interp(float(y), x, p))
    return pdf

def ggd_unit_cdf_from_table():
    x, F = GGD_TABLE_B15["x"], GGD_TABLE_B15["cdf"]
    def cdf(y: float) -> float:
        return float(np.clip(np.interp(float(y), x, F), 0.0, 1.0))
    return cdf

def ggd_unit_ppf_from_table(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    if u <= 0.0:
        return -np.inf
    if u >= 1.0:
        return np.inf
    x, F = GGD_TABLE_B15["x"], GGD_TABLE_B15["cdf"]
    return float(np.interp(u, F, x))

# ===================== beta =2.0 =======================
def ggd_unit_pdf_from_table_beta2():
    x, p = GGD_TABLE_B2["x"], GGD_TABLE_B2["pdf"]
    def pdf(y: float) -> float:
        return float(np.interp(float(y), x, p))
    return pdf

def ggd_unit_cdf_from_table_beta2():
    x, F = GGD_TABLE_B2["x"], GGD_TABLE_B2["cdf"]
    def cdf(y: float) -> float:
        return float(np.clip(np.interp(float(y), x, F), 0.0, 1.0))
    return cdf

def ggd_unit_ppf_from_table_beta2(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    if u <= 0.0:
        return -np.inf
    if u >= 1.0:
        return np.inf
    x, F = GGD_TABLE_B2["x"], GGD_TABLE_B2["cdf"]
    return float(np.interp(u, F, x))

#===================================================================================================
#    Sin2 Distribution (p = 1.5): Base Table (PDF/CDF) and Unit-Variance Wrappers
#===================================================================================================
def sin2_log_unnorm_pdf_base(x: np.ndarray) -> np.ndarray:
    """
    Unnormalized log PDF for the base sin2 distribution:
    """
    z = np.asarray(x, dtype=np.float64) / C_SIN2
    phi = (A_SIN2 * (np.abs(z) ** P_SIN2)
           + B_SIN2 * (z ** 4)
           + D_SIN2 * (np.sin(W_SIN2 * z) ** 2)
           + 1.0)
    return -phi

def build_sin2_base_table(x_max: float = 50.0, dx: float = 1e-4) -> Dict[str, np.ndarray]:
    """Generate base sin2 PDF/CDF and compute base std s_base."""
    if dx <= 0:
        raise ValueError("dx must be positive")
    x = np.arange(-x_max, x_max + dx, dx, dtype=np.float64)
    logp = sin2_log_unnorm_pdf_base(x)
    p_unnorm = np.exp(logp - np.max(logp))       # stability
    Z = float(np.trapz(p_unnorm, x))
    p = p_unnorm / Z
    F = np.cumsum(p) * dx
    F[0], F[-1] = 0.0, 1.0
    mean = float(np.trapz(x * p, x))
    var = float(np.trapz(((x - mean) ** 2) * p, x))
    s_base = float(np.sqrt(var))
    return {"x": x, "pdf": p, "cdf": F, "s_base": np.array([s_base], dtype=np.float64)}

SIN2_TABLE = build_sin2_base_table()

def get_sin2_base_std() -> float:
    """Return s_base (std of the base sin2 before unit-variance scaling)."""
    return float(SIN2_TABLE["s_base"][0])

def sin2_unit_pdf() -> Callable[[float], float]:
    """
    Unit-variance sin2 PDF via table scaling:
      if X_base ~ f_base, define Y = X_base / s_base and  Var(Y)=1.
      Then f_unit(y) = s_base · f_base(s_base · y).
    """
    x = SIN2_TABLE["x"]
    p = SIN2_TABLE["pdf"]
    s = get_sin2_base_std()
    def pdf(y: float) -> float:
        return float(s * np.interp(s * float(y), x, p))
    return pdf

def sin2_unit_cdf() -> Callable[[float], float]:
    """Unit-variance sin2 CDF via table scaling."""
    s = get_sin2_base_std()
    x = SIN2_TABLE["x"] / s
    F = SIN2_TABLE["cdf"]
    def cdf(y: float) -> float:
        return float(np.clip(np.interp(float(y), x, F), 0.0, 1.0))
    return cdf

def sin2_unit_ppf(u: float) -> float:
    """Unit-variance sin2 inverse CDF via interpolation."""
    u = float(np.clip(u, 0.0, 1.0))
    if u <= 0.0:
        return -np.inf
    if u >= 1.0:
        return np.inf
    s = get_sin2_base_std()
    x = SIN2_TABLE["x"] / s
    F = SIN2_TABLE["cdf"]
    return float(np.interp(u, F, x))

SIN2_UNITVAR_PDF = sin2_unit_pdf()
SIN2_UNITVAR_CDF = sin2_unit_cdf()

#===================================================================================================
#    Unit-Variance Distribution Functions (PDF/CDF/PPF)
#===================================================================================================
def get_unit_variance_pdf(dist: DistName) -> Callable[[float], float]:
    """Unit-variance PDF for each supported distribution."""
    if dist == "gaussian":
        return ggd_unit_pdf_from_table()
    if dist == "gaussian_b2":
        return ggd_unit_pdf_from_table_beta2()

    if dist == "logistic":
        s = S_UNIT_LOGISTIC
        return lambda x: float(np.exp(-float(x) / s) / (s * (1.0 + np.exp(-float(x) / s)) ** 2))
    if dist == "hypsecant":
        return lambda x: float(0.5 / np.cosh(np.pi * float(x) / 2.0))
    if dist == "sin2":
        return SIN2_UNITVAR_PDF
    raise ValueError("Unsupported distribution")

def get_unit_variance_cdf(dist: DistName) -> Callable[[float], float]:
    """Unit-variance CDF for each supported distribution."""
    if dist == "gaussian":
        return ggd_unit_cdf_from_table()
    if dist == "gaussian_b2":
        return ggd_unit_cdf_from_table_beta2()
    if dist == "logistic":
        return lambda x: float(logistic.cdf(float(x), loc=0, scale=S_UNIT_LOGISTIC))
    if dist == "hypsecant":
        return lambda x: float(hypsecant.cdf(float(x)))
    if dist == "sin2":
        return SIN2_UNITVAR_CDF
    raise ValueError("Unsupported distribution")

def get_unit_variance_ppf(dist: DistName) -> Callable[[float], float]:
    """Unit-variance inverse CDF (PPF) for each supported distribution."""
    if dist == "gaussian":
        return ggd_unit_ppf_from_table
    if dist == "gaussian_b2":
        return ggd_unit_ppf_from_table_beta2
    if dist == "logistic":
        return lambda u: float(logistic.ppf(float(u), loc=0, scale=S_UNIT_LOGISTIC))
    if dist == "hypsecant":
        return lambda u: float(hypsecant.ppf(float(u)))
    if dist == "sin2":
        return sin2_unit_ppf
    raise ValueError("Unsupported distribution")

def evaluate_pdf_at_zero(dist: DistName) -> float:
    """f_X(0) for unit-variance distribution."""
    return float(get_unit_variance_pdf(dist)(0.0))

#===================================================================================================
#    Sampling (unit-variance or (mu sigma)-scaled)
#===================================================================================================
def sample_unit_variance(dist: DistName, n: int) -> np.ndarray:
    """n IID samples from the unit-variance distribution using inverse-CDF sampling."""
    if n <= 0:
        raise ValueError("n must be positive")
    u = np.random.rand(n)
    ppf = get_unit_variance_ppf(dist)
    return np.array([ppf(ui) for ui in u], dtype=np.float64)

def sample_scaled_distribution(dist: DistName, mu: float, sigma: float, n: int) -> np.ndarray:
    """n samples from X = mu  + sigma·Z, Z ~ unit-variance dist."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    base = sample_unit_variance(dist, n)
    return mu + sigma * base

#===================================================================================================
#    Encoding / Decoding
#===================================================================================================
def encode_nonadaptive(dist: DistName, q1: float, mu: float, sigma: float, n: int) -> np.ndarray:
    """One-threshold encoder: bit = 1[x < q1]."""
    x = sample_scaled_distribution(dist, mu, sigma, n)
    return (x < q1).astype(np.int8)

def decode_nonadaptive(dist: DistName, bits: np.ndarray, q1: float, sigma: float) -> Optional[float]:
    """One-threshold decoder: mu_hat = q1 -  \alpha·F^{-1}(f1)."""
    f1 = float(np.mean(bits))
    if not (0.0 < f1 < 1.0):
        return None
    a1 = get_unit_variance_ppf(dist)(f1)
    return float(q1 - sigma * a1)

def decode_adaptive_first_round(
    dist: DistName,
    set1: np.ndarray,
    set2: np.ndarray,
    q1: float,
    q2: float
) -> Optional[Tuple[float, float]]:
    """Two-threshold coarse decode: returns (mu hat sigma_hat)."""
    f1, f2 = float(np.mean(set1)), float(np.mean(set2))
    if not (0.0 < f1 < 1.0 and 0.0 < f2 < 1.0):
        return None
    a1 = get_unit_variance_ppf(dist)(f1)
    a2 = get_unit_variance_ppf(dist)(f2)
    if a1 == a2:
        return None
    sigma_hat = (q1 - q2) / (a1 - a2)
    mu_hat = (a1 * q2 - a2 * q1) / (a1 - a2)
    return float(mu_hat), float(sigma_hat)

def decode_adaptive_second_round(
    dist: DistName,
    set3: np.ndarray,
    mu_hat_c: float,
    sigma_hat_c: float
) -> Optional[float]:
    """Refined mu_hat using threshold at hat_mu_c."""
    f3 = float(np.mean(set3))
    if not (0.0 < f3 < 1.0):
        return None
    a3 = get_unit_variance_ppf(dist)(f3)
    return float(mu_hat_c - a3 * sigma_hat_c)

#===================================================================================================
#    Bounds  ( Fisher info, T(f) via monotone envelope and stationary point for sin2)
#===================================================================================================
def fisher_continuous(dist: DistName, sigma: float = 1.0, z_max: float = 12.0, dz: float = 1e-4) -> float:
    """
    Fisher information for the location parameter in X = mu + sigma Z, where Z ~ unit-variance dist.

    Computes I_Z = integration (f'(z)^2 / f(z)) dz on [-z_max, z_max] numerically,
    and returns I_X = I_Z / sigma^2.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if dz <= 0:
        raise ValueError("dz must be positive")
    if z_max <= 0:
        raise ValueError("z_max must be positive")

    z = np.arange(-z_max, z_max + dz, dz, dtype=np.float64)
    f_fun = get_unit_variance_pdf(dist)
    f = np.array([f_fun(float(zi)) for zi in z], dtype=np.float64)

    # robust floor to avoid 0/0 in tails due to numerical interpolation
    f = np.maximum(f, 1e-300)

    f_prime = np.gradient(f, dz, edge_order=2)
    integrand = (f_prime ** 2) / f
    I_Z = float(np.trapz(integrand, z))
    return I_Z / (sigma ** 2)

def sin2_phi_prime_base(x: np.ndarray) -> np.ndarray:
    """
     sin2 potential (with exponent P_SIN2):
    """
    x = np.asarray(x, dtype=np.float64)
    z = x / C_SIN2
    term1 = (A_SIN2 * P_SIN2 / C_SIN2) * np.sign(z) * (np.abs(z) ** (P_SIN2 - 1))
    term2 = (4.0 * B_SIN2 / (C_SIN2 ** 4)) * (x ** 3)
    term3 = D_SIN2 * (W_SIN2 / C_SIN2) * np.sin(2.0 * W_SIN2 * z)
    return term1 + term2 + term3

def sin2_phi_prime_unit(x: np.ndarray) -> np.ndarray:
    """
    for unit-variance sin2: y = x, base variable = s*x.
    """
    s = get_sin2_base_std()
    return s * sin2_phi_prime_base(s * np.asarray(x, dtype=np.float64))

def sin2_phi_pp_unit(x: np.ndarray) -> np.ndarray:
    """
    phi''_unit(x) for unit-variance sin2.
    Base second derivative (with z = s x / C):
    """
    s = get_sin2_base_std()
    x = np.asarray(x, dtype=np.float64)
    z = (s * x) / C_SIN2

    out = np.zeros_like(z, dtype=float)
    mask = np.abs(z) > 1e-12
    if P_SIN2 > 1:
        out[mask] += (A_SIN2 * P_SIN2 * (P_SIN2 - 1) / (C_SIN2 ** 2)) * (np.abs(z[mask]) ** (P_SIN2 - 2))
    out += (12.0 * B_SIN2 / (C_SIN2 ** 2)) * (z ** 2)
    out += (2.0 * D_SIN2 * (W_SIN2 ** 2) / (C_SIN2 ** 2)) * np.cos(2.0 * W_SIN2 * z)
    return out

def t_fx_generic(dist: DistName, x_max: float, dx: float) -> float:
    """
    T(f) for (gaussian/logistic/hypsecant/gaussian_b2):
      h(x) = 2 phi'(x) f(x),  with  phi' = -f'(x)/f(x).
    Use the monotone envelope of h and invert it to integrate integral phi'(x(t)) x(t) dt from t=0..h*.
    """
    x = np.arange(0.0, x_max + dx, dx, dtype=np.float64)
    f_fun = get_unit_variance_pdf(dist)
    f = np.array([f_fun(float(xi)) for xi in x], dtype=np.float64)

    # numerical phi' = -f'/f
    f_prime = np.gradient(f, dx, edge_order=2)
    phi_prime = -f_prime / np.maximum(f, 1e-300)

    h = 2.0 * phi_prime * f
    h[h < 0.0] = 0.0
    h_mono = np.maximum.accumulate(h)
    h_star = float(h_mono[-1])
    if not np.isfinite(h_star) or h_star <= 0.0:
        return 0.0

    # invert monotone envelope
    t_grid = np.linspace(0.0, h_star, 200_001)
    x_of_t = np.interp(t_grid, h_mono, x)

    phi_prime_at = np.interp(x_of_t, x, phi_prime)
    integrand = phi_prime_at * x_of_t
    return float(np.trapz(integrand, t_grid))

def t_fx_sin2(x_max: float, dx: float) -> float:
    """
    Careful T(f) for sin2 (unit variance):
      h(x) = 2 phi'(x) f(x), with analytic phi'_unit and stationarity at phi'' - (phi')^2 = 0.
      Build monotone envelope only up to the first stationary point x* (or argmax h).
    """
    x = np.arange(0.0, x_max + dx, dx, dtype=np.float64)
    f_fun = SIN2_UNITVAR_PDF
    f = np.array([f_fun(float(xi)) for xi in x], dtype=np.float64)
    phi_prime = sin2_phi_prime_unit(x)

    h = 2.0 * phi_prime * f
    h[h < 0.0] = 0.0

    deriv = sin2_phi_pp_unit(x) - phi_prime ** 2
    idx_cross = np.where(np.diff(np.sign(deriv)) != 0)[0]
    idx_star = int(idx_cross[0]) if len(idx_cross) > 0 else int(np.argmax(h))

    h_star = float(h[idx_star])
    if not np.isfinite(h_star) or h_star <= 0.0:
        return 0.0

    h_seg = h[: idx_star + 1]
    x_seg = x[: idx_star + 1]
    h_mono = np.maximum.accumulate(h_seg)

    t_grid = np.linspace(0.0, h_star, 300_001)
    x_of_t = np.interp(t_grid, h_mono, x_seg)
    phi_prime_at = sin2_phi_prime_unit(x_of_t)
    integrand = phi_prime_at * x_of_t
    return float(np.trapz(integrand, t_grid))

def compute_t_fx(dist: DistName, x_max: Optional[float] = None, dx: float = 1e-4) -> float:
   # wrapper for T(f).

    if dist not in ["gaussian", "gaussian_b2", "logistic", "hypsecant", "sin2"]:
        raise ValueError("Unsupported distribution")

    if x_max is None:
        x_max = 50.0 if dist == "sin2" else 10.0

    if dist == "sin2":
        return t_fx_sin2(x_max, dx)
    else:
        return t_fx_generic(dist, x_max, dx)

def compute_nonadaptive_lower_bound(dist: DistName, sigma: float, override_t: Optional[float] = None) -> float:
    """Non-adaptive lower bound: (NA_CONST / T(f_X)) * \sigma^2 """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    t_val = float(override_t) if override_t is not None else compute_t_fx(dist)
    if t_val <= 0.0 or not np.isfinite(t_val):
        raise ZeroDivisionError("T(f_X) is non-positive; cannot compute lower bound.")
    return (NA_CONST / t_val) * (sigma ** 2)

#=============================================================================================
def compute_adaptive_lower_bound(dist: DistName, sigma: float) -> float:
    """Adaptive lower bound: \sigma^2 / (4 f_X(0)^2)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    fx0 = evaluate_pdf_at_zero(dist)
    return (sigma ** 2) / (4.0 * (fx0 ** 2))

#=======================================================================
def compute_nonadaptive_upper_bound(
    dist: str, n: float, sigma: float,
    theta1: float, theta2: float,
    mu_min: float, mu_max: float,
    k1: float, k2: float,
    n_mu: int = 50,
) -> tuple[float, float]:
    if dist not in ["gaussian", "gaussian_b2", "logistic", "hypsecant", "sin2"]:
        raise ValueError(f"Unsupported distribution: {dist}")
    if sigma <= 0:
        raise ValueError("Standard deviation sigma must be positive")
    if theta1 == theta2:
        raise ValueError("θ₁ and θ₂ must be different")
    if not np.isclose(k1 + k2, 1.0, rtol=1e-6):
        raise ValueError("k₁ + k₂ must equal 1")
    if k1 <= 0 or k2 <= 0:
        raise ValueError("k₁ and k₂ must be positive")

    f = get_unit_variance_pdf(dist)   # type: ignore[arg-type]
    F = get_unit_variance_cdf(dist)   # type: ignore[arg-type]

    mus = np.linspace(mu_min, mu_max, n_mu, dtype=np.float64)
    denom = (theta1 - theta2) ** 2
    if denom <= 0:
        raise ValueError("Denominator (theta1 - theta2)^2 must be positive")

    best_C = -np.inf
    best_mu = float(mu_min)

    for mu in mus:
        z1 = (theta1 - mu) / sigma
        z2 = (theta2 - mu) / sigma

        p1 = float(np.clip(F(z1), 1e-12, 1.0 - 1e-12))
        p2 = float(np.clip(F(z2), 1e-12, 1.0 - 1e-12))
        sig1_sq = p1 * (1.0 - p1)
        sig2_sq = p2 * (1.0 - p2)

        f1 = float(max(f(z1), 1e-300))
        f2 = float(max(f(z2), 1e-300))

        term1 = ((theta2 - mu) ** 2)*(k1 * sig1_sq )/ (f1 ** 2)
        term2 = ((theta1 - mu) ** 2)*(k2 * sig2_sq)/ (f2 ** 2)

        C = (sigma ** 2 / denom) * (term1 + term2)  # UB constant

        if np.isfinite(C) and C > best_C:
            best_C = float(C)
            best_mu = float(mu)

    print(f"Computed non-adaptive UB: mse={best_C} at mu={best_mu} for dist={dist}, n={n}")
    return best_mu, best_C




#===================================================================================================
#    K Configurations
#===================================================================================================
K_CONFIGS_NONADAPTIVE: List[Tuple[float, float]] = [
    (0.10, 0.90),
    (0.20, 0.80),
    (0.30, 0.70),
    (0.40, 0.60),
    (0.50, 0.50)
]
K_CONFIGS_ADAPTIVE: List[Tuple[float, float, float]] = [
    (0.05, 0.05),
    (0.10, 0.10),
    (0.15, 0.15),
]

#===================================================================================================
#    __all__
#===================================================================================================
__all__ = [
    # Types and K-configs
    "DistName",
    "K_CONFIGS_NONADAPTIVE", "K_CONFIGS_ADAPTIVE",

    # Unit-variance fns
    "get_unit_variance_pdf", "get_unit_variance_cdf", "get_unit_variance_ppf", "evaluate_pdf_at_zero",
    "sample_unit_variance", "sample_scaled_distribution",

    # Sin2 base std (useful externally)
    "get_sin2_base_std",

    # Encode/Decode
    "encode_nonadaptive", "decode_nonadaptive", "decode_adaptive_first_round", "decode_adaptive_second_round",

    # Fisher information
    "fisher_continuous",

    # Bounds
    "compute_t_fx",
    "compute_nonadaptive_lower_bound", "compute_adaptive_lower_bound",
    "compute_nonadaptive_upper_bound",
]

#===================================================================================================
#    Sanity Check (executed if run as main)
#===================================================================================================
if __name__ == "__main__":
    std_list = [1.0, 2.0, 3.0]
    dist_set = ["gaussian", "gaussian_b2", "logistic", "hypsecant", "sin2"]

    print("================================================================================")
    print(f"Distribution Constants: Unit Variance and Sigma-Scaled with beta = {BETA_GAUSS} and {BETA_GAUSS_2},\n"
          f"and p = {P_SIN2} for the SIN2 distribution.")
    print("================================================================================")
    print("Dist        Sigma   f_unit(0)    T(f)      C_non      C_adp     Ratio")
    print("------------------------------------------------------------------")

    for dist in dist_set:
        fx0 = evaluate_pdf_at_zero(dist)   # type: ignore[arg-type]
        t_val = compute_t_fx(dist)         # type: ignore[arg-type]

        for sigma in std_list:
            C_non = compute_nonadaptive_lower_bound(dist, sigma, override_t=t_val)  # type: ignore[arg-type]
            C_adp = compute_adaptive_lower_bound(dist, sigma)                       # type: ignore[arg-type]
            ratio = C_non / C_adp if C_adp > 0 else np.inf
            print(f"{dist:<10} {sigma:<5.1f} {fx0:10.4f} {t_val:8.4f} "
                  f"{C_non:9.4f} {C_adp:9.4f} {ratio:9.4f}")

    print("================================================================================")
    print(" Done.")
    print("================================================================================")
