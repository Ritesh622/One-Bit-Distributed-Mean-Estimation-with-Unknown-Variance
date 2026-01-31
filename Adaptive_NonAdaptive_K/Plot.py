# ============================================================
# Device setup and Imports
#============================================================
from __future__ import annotations

import os
import shutil
import pickle
import sys
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
# Formatting Setup
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
    """Attempt to enable LaTeX text rendering."""
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
# Label and Style Setup
# =============================================================================
LABELS = {
    'nonad_worst': 'Worst (Non-adaptive)',
    'nonad_avg':   'Average (Non-adaptive)',
    'nonad_lb':    'Lower Bound (Non-adaptive)',
    'nonad_ub':    'Asymptotic (Non-Adaptive)',
    'adapt_worst': 'Worst (Adaptive)',
    'adapt_avg':   'Average (Adaptive)',
    'adapt_lb':    'Lower Bound (Adaptive)',
    'adapt_special_worst': r'Worst AD $(n_1=n_2=n_3/\log n_3)$',
    'adapt_special_avg':   r'Avg AD   $(n_1=n_2=n_3/\log n_3)$',
}

LINESTYLES = {
    'nonad_worst': '--', 'nonad_avg': ':', 'nonad_lb': '-', 'nonad_ub': '-.',
    'adapt_worst': '--', 'adapt_avg': ':', 'adapt_lb': '-.',
    'adapt_special_worst': '-', 'adapt_special_avg': '-',
}

MARKERS = {
    'nonad_worst': 'd', 'nonad_avg': 'v', 'nonad_ub': 'x',
    'adapt_worst': 'd', 'adapt_avg': 'o',
    'adapt_special_worst': '*', 'adapt_special_avg': '^',
    'nonad_lb': 'd', 'adapt_lb': 'X',
    'nonad_ub': '*'
}

PAIR_COLORS = ['tab:orange', 'tab:green', 'tab:blue', 'tab:purple', 'tab:red']

# =============================================================================
# Directory Setup  (IMPORTANT: use script location, not os.getcwd())
# =============================================================================
script_dir = Path(__file__).resolve().parent

mse_data_dir    = script_dir / "Avg_Worst_MSE_data"
worst_case_dir  = mse_data_dir / "Worst_Case"
avg_case_dir    = mse_data_dir / "Average_Case"
benchmark_dir   = mse_data_dir / "Benchmark"
upper_bound_dir = mse_data_dir / "Upper_Bound"

plots_nonad_dir = script_dir / "Plots_NonAdaptive"
plots_adapt_dir = script_dir / "Plots_Adaptive"
plots_combo_dir = script_dir / "Plots_Combined"

for p in [plots_nonad_dir, plots_adapt_dir, plots_combo_dir]:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Helper Functions
# =============================================================================
dist_set = ["gaussian", "gaussian_b2", "logistic", "hypsecant", "sin2"]
EPS = 1e-16

def ksuffix(prefix, K1=None, K2=None):
    """Append K1, K2 suffix to file names."""
    return prefix if (K1 is None or K2 is None) else f"{prefix}_K1_{K1:.2f}_K2_{K2:.2f}".replace('.', '_')

def load_mse_curve(path: Path):
    """Load simple {samples, mse} dictionary."""
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    samples = data.get('samples')
    mse = data.get('mse')
    if samples is None or mse is None:
        return None
    return np.asarray(samples, dtype=int), np.asarray(mse, dtype=float)

def load_data_dict(path: Path):
    """Load full pickle file as dict."""
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def floor_pos(x, eps: float = EPS):
    """Clamp non-positive or NaN values to eps."""
    a = np.asarray(x, dtype=float)
    a[~np.isfinite(a)] = eps
    a[a <= 0] = eps
    return a

def markevery_step(n, target=80):
    """Subsample markers to avoid clutter."""
    try:
        n = int(n)
    except Exception:
        return 1
    if n <= 0:
        return 1
    return max(1, int(np.ceil(n / float(target))))

def style_axes(ax):
    """Uniform axis and grid styling."""
    ax.grid(True, which="major", linestyle="-", alpha=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

# =============================================================================
# Non-Adaptive Per-K Plots
# =============================================================================
for decode_dist in dist_set:
    nonad_lb_path = benchmark_dir / f"{decode_dist}_nonadaptive_lb.pkl"
    nonad_loaded = load_mse_curve(nonad_lb_path)
    if nonad_loaded is None:
        print(f"[WARNING] Missing non-adaptive bound for {decode_dist}.")
        continue
    nonad_samples, nonad_mse = nonad_loaded
    nonad_mse = floor_pos(nonad_mse)

    for (K1, K2) in AS.K_CONFIGS_NONADAPTIVE:
        tag = ksuffix("NONADAPT", K1, K2)
        worst_path = worst_case_dir / f"{decode_dist}_{tag}_worst_case.pkl"
        avg_path   = avg_case_dir   / f"{decode_dist}_{tag}_average_case.pkl"
        if not (worst_path.exists() and avg_path.exists()):
            continue

        worst = load_data_dict(worst_path) or {}
        avg   = load_data_dict(avg_path) or {}

        total_samples = worst['samples']
        worst_na_mse  = floor_pos(worst['nonadaptive_worst_mse'])
        avg_na_mse    = floor_pos(avg['nonadaptive_average_mse'])

        plt.figure(figsize=(12, 8))
        plt.semilogy(
            total_samples, worst_na_mse,
            LINESTYLES['nonad_worst'],
            marker=MARKERS['nonad_worst'], color='tab:orange',
            label=fr"{LABELS['nonad_worst']} ($K_1={K1:.2f},\,K_2={K2:.2f}$)"
        )

        plt.semilogy(
            total_samples, avg_na_mse,
            LINESTYLES['nonad_avg'],
            marker=MARKERS['nonad_avg'], color='tab:green',
            label=fr"{LABELS['nonad_avg']} ($K_1={K1:.2f},\,K_2={K2:.2f}$)"
        )

        me_lb = markevery_step(len(nonad_samples))
        plt.semilogy(
            nonad_samples, nonad_mse,
            linestyle=LINESTYLES['nonad_lb'], marker=MARKERS['nonad_lb'],
            color='hotpink', label=LABELS['nonad_lb'],
            markevery=me_lb
        )

        plt.xlabel(r"Total Number of Users $(n)$")
        plt.ylabel(r"Mean Squared Error $(\mathrm{MSE})$")
        style_axes(plt.gca())
        plt.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        plt.savefig(plots_nonad_dir / f"MSE_NonAdapt_{decode_dist}_{tag}.pdf", bbox_inches="tight")
        plt.close()

# =============================================================================
# Non-Adaptive Combined Plots
# =============================================================================
for decode_dist in dist_set:
    nonad_lb_path = benchmark_dir / f"{decode_dist}_nonadaptive_lb.pkl"
    nonad_loaded = load_mse_curve(nonad_lb_path)
    if nonad_loaded is None:
        continue
    nonad_samples, nonad_mse = nonad_loaded
    nonad_mse = floor_pos(nonad_mse)

    plt.figure(figsize=(12, 8))
    for i, (K1, K2) in enumerate(AS.K_CONFIGS_NONADAPTIVE):
        tag = ksuffix("NONADAPT", K1, K2)
        worst_path = worst_case_dir / f"{decode_dist}_{tag}_worst_case.pkl"
        if not worst_path.exists():
            continue
        worst = load_data_dict(worst_path) or {}
        total_samples = worst['samples']
        worst_na_mse  = floor_pos(worst['nonadaptive_worst_mse'])
        color = PAIR_COLORS[i % len(PAIR_COLORS)]
        me = markevery_step(len(total_samples))
        plt.semilogy(total_samples, worst_na_mse,
                     label=fr'Worst NA ($K_1={K1:.2f},\,K_2={K2:.2f}$)',
                     color=color, linestyle=LINESTYLES['nonad_worst'],
                     marker=MARKERS['nonad_worst'], markevery=me)

    me_lb = markevery_step(len(nonad_samples))
    plt.semilogy(nonad_samples, nonad_mse,
                 linestyle=LINESTYLES['nonad_lb'], marker=MARKERS['nonad_lb'],
                 color='hotpink', label=LABELS['nonad_lb'], markevery=me_lb)
    plt.xlabel(r"Total Number of Users $(n)$")
    plt.ylabel(r"Mean Squared Error $(\mathrm{MSE})$")
    style_axes(plt.gca())
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(plots_combo_dir / f"MSE_NonAdapt_{decode_dist}_COMBINED_ALL.pdf", bbox_inches="tight")
    plt.close()

# =============================================================================
# Adaptive Per-K Plots + Special Case Plot
# =============================================================================
for decode_dist in dist_set:
    bench_path = benchmark_dir / f"{decode_dist}_benchmark.pkl"
    bench = load_mse_curve(bench_path)
    if bench is None:
        continue
    bench_samples, bench_mse = bench
    bench_mse = floor_pos(bench_mse)

    # ------------------ Regular (K1,K2) Adaptive Plots ------------------
    for (K1, K2) in AS.K_CONFIGS_ADAPTIVE:
        tag = ksuffix("ADAPT", K1, K2)
        worst_path = worst_case_dir / f"{decode_dist}_{tag}_worst_case.pkl"
        avg_path   = avg_case_dir   / f"{decode_dist}_{tag}_average_case.pkl"
        if not (worst_path.exists() and avg_path.exists()):
            continue

        worst = load_data_dict(worst_path) or {}
        avg   = load_data_dict(avg_path) or {}
        total_samples = worst["samples"]
        worst_ad_mse  = floor_pos(worst["adaptive_worst_mse"])
        avg_ad_mse    = floor_pos(avg["adaptive_average_mse"])

        plt.figure(figsize=(12, 8))
        plt.semilogy(
            total_samples, worst_ad_mse,
            LINESTYLES["adapt_worst"],
            marker=MARKERS["adapt_worst"], color="tab:blue",
            label=fr"{LABELS['adapt_worst']} ($K_1={K1:.2f},\,K_2={K2:.2f}$)"
        )
        plt.semilogy(
            total_samples, avg_ad_mse,
            LINESTYLES["adapt_avg"],
            marker=MARKERS["adapt_avg"], color="tab:purple",
            label=fr"{LABELS['adapt_avg']} ($K_1={K1:.2f},\,K_2={K2:.2f}$)"
        )

        me_lb = markevery_step(len(bench_samples))
        plt.semilogy(
            bench_samples, bench_mse,
            linestyle=LINESTYLES["adapt_lb"], marker=MARKERS["adapt_lb"],
            color="hotpink", label=LABELS["adapt_lb"], markevery=me_lb
        )

        plt.xlabel(r"Total Number of Users $(n)$")
        plt.ylabel(r"Mean Squared Error $(\mathrm{MSE})$")
        style_axes(plt.gca())
        plt.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        plt.savefig(plots_adapt_dir / f"MSE_Adapt_{decode_dist}_{tag}.pdf", bbox_inches="tight")
        plt.close()

    # ------------------ Special Adaptive Plot ------------------
    special_worst_path = worst_case_dir / f"{decode_dist}_ADAPT_SPECIAL_worst_case.pkl"
    special_avg_path   = avg_case_dir   / f"{decode_dist}_ADAPT_SPECIAL_average_case.pkl"
    if not (special_worst_path.exists() and special_avg_path.exists()):
        continue

    worst_s = load_data_dict(special_worst_path)
    avg_s   = load_data_dict(special_avg_path)
    total_samples = worst_s["samples"]
    worst_sp_mse  = floor_pos(worst_s["adaptive_worst_mse"])
    avg_sp_mse    = floor_pos(avg_s["adaptive_average_mse"])

    plt.figure(figsize=(12, 8))
    plt.semilogy(
        total_samples, worst_sp_mse,
        LINESTYLES["adapt_special_worst"],
        marker=MARKERS["adapt_special_worst"], color="tab:red",
        label=LABELS["adapt_special_worst"]
    )
    plt.semilogy(
        total_samples, avg_sp_mse,
        LINESTYLES["adapt_special_avg"],
        marker=MARKERS["adapt_special_avg"], color="tab:orange",
        label=LABELS["adapt_special_avg"]
    )

    me_lb = markevery_step(len(bench_samples))
    plt.semilogy(
        bench_samples, bench_mse,
        linestyle=LINESTYLES["adapt_lb"], marker=MARKERS["adapt_lb"],
        color="hotpink", label=LABELS["adapt_lb"], markevery=me_lb
    )

    plt.xlabel(r"Total Number of Users $(n)$")
    plt.ylabel(r"Mean Squared Error $(\mathrm{MSE})$")
    style_axes(plt.gca())
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(plots_adapt_dir / f"MSE_Adapt_{decode_dist}_ADAPT_SPECIAL.pdf", bbox_inches="tight")
    plt.close()

# =============================================================================
# Adaptive Special + Combined
# =============================================================================
for decode_dist in dist_set:
    bench_path = benchmark_dir / f"{decode_dist}_benchmark.pkl"
    bench = load_mse_curve(bench_path)
    if bench is None:
        continue
    bench_samples, bench_mse = bench
    bench_mse = floor_pos(bench_mse)

    plt.figure(figsize=(12, 8))
    for i, (K1, K2) in enumerate(AS.K_CONFIGS_ADAPTIVE):
        tag = ksuffix("ADAPT", K1, K2)
        worst_path = worst_case_dir / f"{decode_dist}_{tag}_worst_case.pkl"
        if not worst_path.exists():
            continue
        worst = load_data_dict(worst_path) or {}
        total_samples = worst["samples"]
        worst_ad_mse = floor_pos(worst["adaptive_worst_mse"])
        color = PAIR_COLORS[i % len(PAIR_COLORS)]
        me = markevery_step(len(total_samples))
        plt.semilogy(
            total_samples, worst_ad_mse,
            label=fr"Worst AD ($K_1={K1:.2f},\,K_2={K2:.2f}$)",
            color=color, linestyle=LINESTYLES["adapt_worst"],
            marker=MARKERS["adapt_worst"], markevery=me
        )

    special_path = worst_case_dir / f"{decode_dist}_ADAPT_SPECIAL_worst_case.pkl"
    if special_path.exists():
        special = load_data_dict(special_path)
        total_samples_sp = special["samples"]
        worst_sp_mse = floor_pos(special["adaptive_worst_mse"])
        plt.semilogy(
            total_samples_sp, worst_sp_mse,
            LINESTYLES["adapt_special_worst"],
            marker=MARKERS["adapt_special_worst"], color="tab:red",
            label=LABELS["adapt_special_worst"]
        )

    me_lb = markevery_step(len(bench_samples))
    plt.semilogy(
        bench_samples, bench_mse,
        linestyle=LINESTYLES["adapt_lb"], marker=MARKERS["adapt_lb"],
        color="hotpink", label=LABELS["adapt_lb"], markevery=me_lb
    )

    plt.xlabel(r"Total Number of Users $(n)$")
    plt.ylabel(r"Mean Squared Error $(\mathrm{MSE})$")
    style_axes(plt.gca())
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(plots_combo_dir / f"MSE_Adapt_{decode_dist}_COMBINED_ALL.pdf", bbox_inches="tight")
    plt.close()

# =============================================================================
# Completion Message
# =============================================================================
print("Saved plots in:")
print("  -", plots_nonad_dir.name)
print("  -", plots_adapt_dir.name)
print("  -", plots_combo_dir.name)
# =============================================================================
