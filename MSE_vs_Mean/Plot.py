# ============================================================
# Import and Device Set up
# ============================================================
from __future__ import annotations
import os, shutil, pickle
from pathlib import Path
from typing import Dict, Any
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# =============================================================================
# Format set up
# =============================================================================
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 30,
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "legend.fontsize": 24,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "lines.linewidth": 3.0,  
    "lines.markersize": 8, 
})

def enable_latex():
    try:
        matplotlib.rcParams.update({"text.usetex": True})
        plt.figure()
        plt.close()
    except Exception:
        matplotlib.rcParams["text.usetex"] = False
enable_latex()

# =============================================================================
# Directory Setup
# =============================================================================
HERE = Path(__file__).resolve().parent
IN_DIR = HERE / "MSE_vs_mu"
OUT_DIR = HERE / "MSE_vs_mu_Plots"
AVG_FILE = IN_DIR / "avg_mse_vs_mu.pkl"
WC_FILE = IN_DIR / "wc_mse_vs_mu.pkl"

if os.path.exists(OUT_DIR):
    print(f"Removing existing directory: {OUT_DIR}")
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Created directory: {OUT_DIR}")

def fallback_find(name: str) -> Path | None:
    cands = list(HERE.rglob(name))
    return cands[0] if cands else None

# =============================================================================
# Centralized Style
# =============================================================================
SMOOTH_WINDOW = 7
Y_EPS = 1e-14
N_MARKERS = 12  # number of evenly spaced markers to draw manually

MARKERS = {"nonadaptive": "o", "adaptive": "s"}
LINESTYLES = {"nonadaptive": "-", "adaptive": "-"}
COLORS = {"nonadaptive": "tab:orange", "adaptive": "tab:blue", "threshold": "tab:red"}

# =============================================================================
# Helpers
# =============================================================================
def load_pickle(p: Path) -> Dict[str, Any]:
    with open(p, "rb") as f:
        return pickle.load(f)

def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size < 3:
        return y
    w = int(window) | 1
    pad = w // 2
    y_pad = np.pad(y, pad_width=pad, mode="reflect")
    kernel = np.ones(w) / w
    return np.convolve(y_pad, kernel, mode="valid")

def prep(y: np.ndarray, eps: float) -> np.ndarray:
    y = np.asarray(y, float)
    y[~np.isfinite(y)] = np.nan
    y = np.nan_to_num(y, nan=np.nanmedian(y))
    return np.maximum(y, eps)

def normalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)
    if "nonadaptive" not in out:
        for k in ["nonadaptive_avg_curve", "nonadaptive_wc_curve"]:
            if k in out:
                out["nonadaptive"] = out[k]
                break
    if "adaptive" not in out:
        for k in ["adaptive_avg_curve", "adaptive_wc_curve"]:
            if k in out:
                out["adaptive"] = out[k]
                break
    return out

def sparse_indices(x: np.ndarray, n_points: int) -> np.ndarray:
    """Return ~n_points evenly spaced indices for markers."""
    if x.size <= n_points:
        return np.arange(x.size)
    step = max(1, x.size // n_points)
    return np.arange(0, x.size, step)

def print_mse_at_zero(payload: Dict[str, Any], tag: str):
    d = normalize_keys(payload)
    mu = np.asarray(d.get("mu_values", []), float)
    if mu.size == 0:
        return
    idx = int(np.argmin(np.abs(mu)))
    na = np.asarray(d.get("nonadaptive", []), float)
    ad = np.asarray(d.get("adaptive", []), float)
    print(f"[mu ≈0] {tag}: mu ={mu[idx]:.3f}, Non-adaptive={na[idx]:.6g}, Adaptive={ad[idx]:.6g}")

# =============================================================================
# Plot Function
# =============================================================================
def plot_one(payload: Dict[str, Any], tag: str, out_pdf: Path):
    d = normalize_keys(payload)
    mu = np.asarray(d.get("mu_values", []), float)
    na = np.asarray(d.get("nonadaptive", []), float)
    ad = np.asarray(d.get("adaptive", []), float)
    if mu.size == 0 or na.size == 0 or ad.size == 0:
        print(f"[skip] missing keys for {tag}")
        return

    na, ad = smooth(na, SMOOTH_WINDOW), smooth(ad, SMOOTH_WINDOW)
    na, ad = prep(na, Y_EPS), prep(ad, Y_EPS)

    plt.figure(figsize=(12, 8))
    # --- Main smooth curves ---
    plt.semilogy(mu, na, LINESTYLES["nonadaptive"], color=COLORS["nonadaptive"], label=r"Non-adaptive")
    plt.semilogy(mu, ad, LINESTYLES["adaptive"], color=COLORS["adaptive"], label=r"Adaptive")

    # --- Sparse markers (draw separately) ---
    idx_na = sparse_indices(mu, N_MARKERS)
    idx_ad = sparse_indices(mu, N_MARKERS)
    plt.semilogy(mu[idx_na], na[idx_na], linestyle="none", marker=MARKERS["nonadaptive"],
                 color=COLORS["nonadaptive"], markersize=8)
    plt.semilogy(mu[idx_ad], ad[idx_ad], linestyle="none", marker=MARKERS["adaptive"],
                 color=COLORS["adaptive"], markersize=8)

    # --- Thresholds ---
    for th, nm in [(d.get("theta1"), r"$\theta_1$"), (d.get("theta2"), r"$\theta_2$")]:
        if th is not None:
            plt.axvline(float(th), color=COLORS["threshold"], linestyle="-.", linewidth=2.0,
                        alpha=0.7, label=f"{nm}={float(th):.4f}")

    plt.xlabel(r"$\mu$")
    plt.ylabel(r"Mean Squared Error $(\mathrm{MSE})$")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="-", alpha=0.5)

    # --- Legend (top-center, stacked vertically) ---
    leg = plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        frameon=True,
        ncol=1,
    )
    leg.get_frame().set_alpha(0.95)
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_pdf}")
    print_mse_at_zero(d, tag)

# =============================================================================
# Main
# =============================================================================
if not AVG_FILE.exists():
    fb = fallback_find("avg_mse_vs_mu.pkl")
    if fb:
        AVG_FILE = fb
if not WC_FILE.exists():
    fb = fallback_find("wc_mse_vs_mu.pkl")
    if fb:
        WC_FILE = fb

if AVG_FILE and AVG_FILE.exists():
    avg_payload = load_pickle(AVG_FILE)
    plot_one(avg_payload, "avg", OUT_DIR / "avg_mse_vs_mu.pdf")
else:
    print(f"[skip] not found: {AVG_FILE}")

# Optional: uncomment for worst-case
# if WC_FILE and WC_FILE.exists():
#     wc_payload = load_pickle(WC_FILE)
#     plot_one(wc_payload, "worst", OUT_DIR / "wc_mse_vs_mu.pdf")
# else:
#     print(f"[skip] not found: {WC_FILE}")

print(f"\nAll requested plots saved in: {OUT_DIR}")
# =============================================================================
