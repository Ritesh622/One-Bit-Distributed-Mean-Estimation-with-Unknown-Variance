# ===================================================================================================
# Import and Device Set up
# ===================================================================================================
from __future__ import annotations
import numpy as np
import pickle
from pathlib import Path
import sys

# ---------------------------------------------------------------------------------------
# Ensure All_Schemes is importable
# ---------------------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import All_Schemes as AS

# ---------------------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "Beta_Data"
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUT_DIR / "ggd_beta_bounds.pkl"

BETA_MIN, BETA_MAX, BETA_STEP = 1.1, 2.5, 0.02

# ---------------------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------------------
def rebuild_ggd_lookup_table(beta: float) -> None:
    """Force All_Schemes to rebuild internal GGD lookup for current beta."""
    AS.BETA_GAUSS = beta
    AS._GGD = AS._build_ggd_table(beta=beta)

def compute_constants(beta: float) -> tuple[float, float, float]:
    """Compute (C_non, C_adapt, ratio) for GGD(beta)."""
    rebuild_ggd_lookup_table(beta)
    t_val = AS.compute_t_fx("gaussian")
    f0 = AS.ggd_pdf(0.0, beta=beta)
    c_non = AS.NA_CONST / t_val
    c_adapt = 1.0 / (4.0 * f0 * f0)
    return c_non, c_adapt, c_non / c_adapt

def find_crossing(beta_grid: np.ndarray, diff_grid: np.ndarray) -> float:
    """Simple bisection to find beta* where C_non = C_adapt."""
    for i in range(len(diff_grid) - 1):
        if diff_grid[i] * diff_grid[i + 1] < 0:
            L, U = beta_grid[i], beta_grid[i + 1]
            for _ in range(80):
                M = 0.5 * (L + U)
                c_non, c_adapt, _ = compute_constants(M)
                fM = c_non - c_adapt
                if np.sign(fM) == np.sign(diff_grid[i]):
                    L = M
                else:
                    U = M
                if abs(U - L) < 1e-6:
                    return M
    raise RuntimeError("No sign change found for crossing beta*.")

# ---------------------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------------------
def main() -> None:
    betas = np.arange(BETA_MIN, BETA_MAX + BETA_STEP, BETA_STEP)
    C_non, C_adapt, ratio = np.zeros_like(betas), np.zeros_like(betas), np.zeros_like(betas)

    for i, b in enumerate(betas):
        C_non[i], C_adapt[i], ratio[i] = compute_constants(b)
        print(f"beta={b:.2f}  C_non={C_non[i]:.6f}, C_adapt={C_adapt[i]:.6f}, ratio={ratio[i]:.3f}")

    diff = C_non - C_adapt
    beta_star = find_crossing(betas, diff)
    print(f"\n[crossing beta*] ≈ {beta_star:.4f}")

    # Save results
    results = {
        "betas": betas,
        "C_non": C_non,
        "C_adapt": C_adapt,
        "ratio": ratio,
        "beta_star": beta_star,
    }

    with open(OUT_FILE, "wb") as f:
        pickle.dump(results, f)
    print(f"[Saved]  {OUT_FILE}")

if __name__ == "__main__":
    main()
