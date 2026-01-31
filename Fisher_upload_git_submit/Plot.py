import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ============================================================
#  Global style configuration 
# ============================================================
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "axes.labelsize": 18,
    "font.size": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.titlesize": 18,
    "lines.linewidth": 2,
    "figure.figsize": (8, 6),

    # keeps text editable in PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ------------------------------------------------------------
# Markers, linestyles, colors
# ------------------------------------------------------------
MARKERS = {
    "cont": "*",
    "nonadp": "d",
    "adp": "^",
}

LINESTYLES = {
    "cont": "--",
    "nonadp": "-",
    "adp": "-.",
}

COLORS = {
    "cont": "blue",
    "nonadp": "#d62728",
    "adp": "#FF69B4",
}

# ------------------------------------------------------------
# Directory of fisher
# ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "Fisher_Data"
PLOT_DIR = ROOT_DIR / "Fisher_Plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# These are the 4 distributions used in the paper
DIST_LIST = ["gaussian", "logistic", "hypsecant", "sin2"]

# ============================================================
#  Plot for each distribution separately
# ============================================================
for dist in DIST_LIST:

    file = DATA_DIR / f"fisher_{dist}.pkl"
    if not file.exists():
        print(f"[WARNING] Missing file: {file}")
        continue

    with open(file, "rb") as f:
        D = pickle.load(f)

    # Extract
    n_vals   = np.asarray(D["n_vals"])
    mse_cont = np.asarray(D["mse_cont"])
    mse_non  = np.asarray(D["mse_non"])
    mse_adp  = np.asarray(D["mse_adp"])

    # Marker frequency
    total_points = len(n_vals)
    me = max(1, total_points // 40)

    plt.figure()

    # ---- Continuous Fisher bound ----
    plt.semilogy(
        n_vals, mse_cont,
        marker=MARKERS["cont"],
        linestyle=LINESTYLES["cont"],
        color=COLORS["cont"],
        label=r"Fisher Bound (No Quantization)",
        markevery=me, linewidth=2.5,
    )

    # ---- Non-adaptive LB ----
    plt.semilogy(
        n_vals, mse_non,
        marker=MARKERS["nonadp"],
        linestyle=LINESTYLES["nonadp"],
        color=COLORS["nonadp"],
        label=r"Non-adaptive LB",
        markevery=me, linewidth=2.5,
    )

    # ---- Adaptive LB ----
    plt.semilogy(
        n_vals, mse_adp,
        marker=MARKERS["adp"],
        linestyle=LINESTYLES["adp"],
        color=COLORS["adp"],
        label=r"Adaptive LB",
        markevery=me, linewidth=2.5,
    )

    plt.xlabel(r"Number of users $n$")
    plt.ylabel(r"Lower bound on MSE")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()

    # Save plot
    out_path = PLOT_DIR / f"fisher_plot_{dist}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print("Saved:", str(out_path))

print("\nAll plots saved in:", str(PLOT_DIR))
