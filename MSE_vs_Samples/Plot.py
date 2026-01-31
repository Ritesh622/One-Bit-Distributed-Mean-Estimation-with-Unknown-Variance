# ============================================================
# Import andDevice Setup
# ============================================================
from __future__ import annotations
import os, shutil, pickle, sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, NullLocator, NullFormatter
from pathlib import Path

# =============================================================================
# Project Root Import
# =============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import All_Schemes as AS

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
    """ LaTeX text rendering."""
    try:
        matplotlib.rcParams.update({"text.usetex": True})
        plt.figure()
        plt.xlabel(r"$n$")
        plt.ylabel(r"$\mathrm{MSE}$")
        plt.close()
    except Exception:
        matplotlib.rcParams["text.usetex"] = False
enable_latex()

# =============================================================================
# Directory Setup
# =============================================================================
parent_dir = os.getcwd()
mse_data_dir = os.path.join(parent_dir, "Avg_Worst_MSE_data")
worst_case_dir = os.path.join(mse_data_dir, "Worst_Case")
avg_case_dir   = os.path.join(mse_data_dir, "Average_Case")
benchmark_dir  = os.path.join(mse_data_dir, "Benchmark")
upper_bound_dir = os.path.join(mse_data_dir, "Upper_Bound")
mse_plot_dir = os.path.join(parent_dir, "Worst_Average_MSE_Plots")

if os.path.exists(mse_plot_dir):
    print(f"Removing existing directory: {mse_plot_dir}")
    shutil.rmtree(mse_plot_dir)
os.makedirs(mse_plot_dir, exist_ok=True)
print(f"Created directory: {mse_plot_dir}")

# =============================================================================
# Plot  Configuration
# =============================================================================
dist_set = ["gaussian", "gaussian_b2", "hypsecant", "logistic", "sin2"]
# dist_set = ["gaussian"]  # for quick testing
true_sigma = AS.SIGMA_GLOB

MARKERS = {
    "adaptive_worst": "d",
    "adaptive_avg": "v",
    "nonadaptive_worst": "d",
    "nonadaptive_avg": "v",
    "adaptive_lb": "X",
    "nonadaptive_lb": "d",
    "nonadaptive_ub": "*",
}
LINESTYLES = {
    "adaptive_worst": "--",
    "adaptive_avg": ":",
    "nonadaptive_worst": "--",
    "nonadaptive_avg": ":",
    "adaptive_lb": "-",
    "nonadaptive_lb": "-",
    "nonadaptive_ub": "-.",
}

# =============================================================================
# Helper Functions
# =============================================================================
def load_curve_data(path: str):
    """Load MSE curve from pickle file."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    samples = data.get("samples")
    mse = data.get("mse")
    if samples is None or mse is None:
        return None
    return np.asarray(samples, dtype=int), np.asarray(mse, dtype=float)

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
# Main Plotting Loop
# =============================================================================
for decode_dist in dist_set:
    print(f"\nGenerating plot for {decode_dist}")

    worst_case_path = os.path.join(worst_case_dir, f"{decode_dist}_worst_case.pkl")
    avg_case_path   = os.path.join(avg_case_dir,   f"{decode_dist}_average_case.pkl")
    ub_path         = os.path.join(upper_bound_dir, f"{decode_dist}_nonadaptive_ub.pkl")

    if not (os.path.exists(worst_case_path) and os.path.exists(avg_case_path) and os.path.exists(ub_path)):
        print(f"[WARNING] Missing files for {decode_dist}, skipping.")
        continue

    with open(worst_case_path, "rb") as f:
        worst_data = pickle.load(f)
    with open(avg_case_path, "rb") as f:
        avg_data = pickle.load(f)
    with open(ub_path, "rb") as f:
        ub_data = pickle.load(f)

    total_samples = worst_data["samples"]

    # Adaptive / Nonadaptive data
    worst_ad_mse = ensure_positive(worst_data["adaptive_worst_mse"])
    avg_ad_mse   = ensure_positive(avg_data["adaptive_average_mse"])
    worst_na_mse = ensure_positive(worst_data["nonadaptive_worst_mse"])
    avg_na_mse   = ensure_positive(avg_data["nonadaptive_average_mse"])
    ub_na_mse    = ensure_positive(ub_data["mse_upper_bound"])

    # Lower bounds
    bench_samples, bench_mse = load_curve_data(os.path.join(benchmark_dir, f"{decode_dist}_benchmark.pkl"))
    nonad_samples, nonad_mse = load_curve_data(os.path.join(benchmark_dir, f"{decode_dist}_nonadaptive_lb.pkl"))

    bench_mse = ensure_positive(bench_mse)
    nonad_mse = ensure_positive(nonad_mse)

    # =============================================================================
    # Figure Creation
    # =============================================================================
    plt.figure(figsize=(12, 8))

    # Adaptive Curves
    plt.semilogy(total_samples, worst_ad_mse,
                 LINESTYLES["adaptive_worst"], marker=MARKERS["adaptive_worst"], color="tab:blue",
                 label=r"Worst-Case (Adaptive)")
    plt.semilogy(total_samples, avg_ad_mse,
                 LINESTYLES["adaptive_avg"], marker=MARKERS["adaptive_avg"], color="tab:purple",
                 label=r"Average-Case (Adaptive)")

    # Non-Adaptive Curves
    plt.semilogy(total_samples, worst_na_mse,
                 LINESTYLES["nonadaptive_worst"], marker=MARKERS["nonadaptive_worst"], color="tab:orange",
                 label=r"Worst-Case (Non-adaptive)")
    plt.semilogy(total_samples, avg_na_mse,
                 LINESTYLES["nonadaptive_avg"], marker=MARKERS["nonadaptive_avg"], color="tab:green",
                 label=r"Average-Case (Non-adaptive)")

    # Upper Bound
    plt.semilogy(total_samples, ub_na_mse,
                 LINESTYLES["nonadaptive_ub"], marker=MARKERS["nonadaptive_ub"], color="magenta",
                 label=r"Asymptotic (Non-adaptive)")

    # Lower Bounds (using centralized markers)
    plt.semilogy(bench_samples, bench_mse,
                 LINESTYLES["adaptive_lb"], marker=MARKERS["adaptive_lb"], color="tab:red",
                 label=r"Lower Bound (Adaptive)")
    plt.semilogy(nonad_samples, nonad_mse,
                 LINESTYLES["nonadaptive_lb"], marker=MARKERS["nonadaptive_lb"], color="hotpink",
                 label=r"Lower Bound (Non-adaptive)")

    # Axis configuration
    plt.xlabel(r"Total Number of Users $(n)$")
    plt.ylabel(r"Mean Squared Error $(\mathrm{MSE})$")
    style_axes(plt.gca())
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(mse_plot_dir, f"MSE_vs_Samples_{decode_dist}.pdf")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")

print(f"\nAll plots saved in: {mse_plot_dir}")
# ===============================================================================
