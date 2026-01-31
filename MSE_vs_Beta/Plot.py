# ============================================================
# Import and Device Set up
# ============================================================
from __future__ import annotations

import pickle
import shutil
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, NullLocator, NullFormatter

# =============================================================================
# Project Root Import
# =============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    """LaTeX text rendering (fallback to non-TeX if unavailable)."""
    try:
        matplotlib.rcParams.update({"text.usetex": True})
        plt.figure()
        plt.xlabel(r"$\beta$")
        plt.ylabel(r"$C_{\mathrm{non}}$")
        plt.close()
    except Exception:
        matplotlib.rcParams["text.usetex"] = False

enable_latex()

# =============================================================================
# Directory Setup
# =============================================================================
script_dir = Path(__file__).resolve().parent

# Match Experiment.py output:
data_file = script_dir / "Beta_Data" / "ggd_beta_bounds.pkl"
if not data_file.exists():
    raise FileNotFoundError(f"Missing data file: {data_file}")

out_dir = script_dir / "Beta_Plots"

if out_dir.exists():
    print(f"Removing existing directory: {out_dir}")
    shutil.rmtree(out_dir)

out_dir.mkdir(parents=True, exist_ok=True)
print(f"Created directory: {out_dir}")

# =============================================================================
# Helper Functions
# =============================================================================
def ensure_positive(arr_like, eps: float = 1e-16):
    """Replace invalid or nonpositive values with eps."""
    arr = np.asarray(arr_like, dtype=float)
    arr[~np.isfinite(arr)] = eps
    arr[arr <= 0] = eps
    return arr

def style_axes(ax):
    """Apply consistent axes and grid styling."""
    ax.grid(True, which="both", linestyle="-", alpha=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

# =============================================================================
# Load Data
# =============================================================================
with open(data_file, "rb") as f:
    data = pickle.load(f)

betas = np.asarray(data["betas"], dtype=float)
C_non = ensure_positive(data["C_non"])
C_adapt = ensure_positive(data["C_adapt"])
ratio = ensure_positive(data["ratio"])
beta_star = float(data["beta_star"])

# =============================================================================
# Plot Constants vs Beta
# =============================================================================
print("\nGenerating plot: constants_vs_beta.pdf")
plt.figure(figsize=(12, 8))
plt.plot(betas, C_non, label=r"$C_{\mathrm{non}} = 0.1034/T(f_X)$", color="tab:blue")
plt.plot(betas, C_adapt, label=r"$C_{\mathrm{adapt}} = 1/(4f_X(0)^2)$", color="tab:orange")
plt.axvline(beta_star, linestyle="-.", color="tab:red", linewidth=2.5,
            label=rf"Crossing $\beta^* \approx {beta_star:.2f}$")

plt.xlabel(r"Shape parameter $(\beta)$")
plt.ylabel(r"Asymptotic MSE / $\sigma^2$")
style_axes(plt.gca())
plt.legend(loc="best", frameon=True)
plt.tight_layout()

plot_path = out_dir / "constants_vs_beta.pdf"
plt.savefig(plot_path, bbox_inches="tight")
plt.close()
print(f"Saved plot: {plot_path}")

# =============================================================================
#  Plot Ratio vs Beta
# =============================================================================
print("Generating plot: ratio_vs_beta.pdf")
plt.figure(figsize=(12, 8))
plt.semilogy(betas, ratio, color="tab:blue", label=r"$C_{\mathrm{non}}/C_{\mathrm{adapt}}$")
plt.axhline(1.0, color="purple", linestyle="--", linewidth=2.5, label=r"Ratio = 1")
plt.axvline(beta_star, color="tab:red", linestyle="-.", linewidth=2.5,
            label=rf"Crossing $\beta^* \approx {beta_star:.2f}$")

plt.xlabel(r"Shape parameter $(\beta)$")
plt.ylabel(r"$C_{\mathrm{non}}/C_{\mathrm{adapt}}$")
style_axes(plt.gca())
plt.legend(loc="best", frameon=True)
plt.tight_layout()

plot_path = out_dir / "ratio_vs_beta.pdf"
plt.savefig(plot_path, bbox_inches="tight")
plt.close()
print(f"Saved plot: {plot_path}")

print(f"\nAll plots saved in: {out_dir}")
# =============================================================================
