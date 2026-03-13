"""
generate_dataset.py
===================
Generates training data for the hull MDO surrogate.

Workflow
--------
1. Sample N hull designs using Halton low-discrepancy sequence over h ∈ [0,1]^29
2. Evaluate each design with the canonical truth model  (truth_resistance_model.py)
3. Save the full dataset to  hull_dataset.csv

The saved CSV has columns:  h1 .. h29,  R,  V

Run once — the CSV is then loaded by hull_mdo_surrogate.py.
Re-running regenerates the data (same seed → identical results).

Usage
-----
    python generate_dataset.py              # default 600 samples
    python generate_dataset.py --n 1000    # custom sample count
"""

import argparse
import csv
import numpy as np
from True_model_test import compute_R_and_V, CONST


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_halton(n: int, d: int = 29, seed: int = 0) -> np.ndarray:
    """
    Return n samples in [0,1]^d using Halton low-discrepancy sequence.
    Falls back to Latin Hypercube Sampling if scipy is unavailable.
    """
    try:
        from scipy.stats import qmc
        sampler = qmc.Halton(d=d, scramble=True, seed=seed)
        return sampler.random(n=n).astype(float)
    except Exception:
        # Fallback: Latin Hypercube Sampling
        rng = np.random.default_rng(seed)
        cut = np.linspace(0.0, 1.0, n + 1)
        X   = np.zeros((n, d))
        for j in range(d):
            pts = cut[:n] + (cut[1:] - cut[:n]) * rng.random(n)
            rng.shuffle(pts)
            X[:, j] = pts
        return X


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_dataset(
    n_samples: int = 600,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample n_samples hull designs and evaluate the truth model on each.

    Returns
    -------
    H : (n, 29)  normalized design parameters
    R : (n,)     total resistance  [N]
    V : (n,)     displaced volume  [m³]
    """
    print(f"Sampling {n_samples} hull designs (Halton, seed={seed})...")
    H = sample_halton(n_samples, d=29, seed=seed)
    R = np.zeros(n_samples)
    V = np.zeros(n_samples)

    for i in range(n_samples):
        R[i], V[i] = compute_R_and_V(H[i], **CONST)
        if (i + 1) % 100 == 0:
            print(f"  Evaluated {i + 1}/{n_samples}")

    print(f"Done.  R ∈ [{R.min():.3f}, {R.max():.3f}] N   "
          f"V ∈ [{V.min():.4f}, {V.max():.4f}] m³")
    return H, R, V


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def save_csv(path: str, H: np.ndarray, R: np.ndarray, V: np.ndarray) -> None:
    """Write dataset to CSV with columns h1..h29, R, V."""
    n, d = H.shape
    header = [f"h{k+1}" for k in range(d)] + ["R", "V"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n):
            w.writerow([*H[i].tolist(), float(R[i]), float(V[i])])
    print(f"Saved {n} rows → {path}")


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a dataset CSV written by save_csv.

    Returns
    -------
    H : (n, 29)   design parameters
    R : (n,)      resistance [N]
    V : (n,)      volume [m³]
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    H = data[:, :29]
    R = data[:, 29]
    V = data[:, 30]
    print(f"Loaded {len(R)} rows from {path}   "
          f"R ∈ [{R.min():.3f}, {R.max():.3f}] N   "
          f"V ∈ [{V.min():.4f}, {V.max():.4f}] m³")
    return H, R, V


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hull MDO training dataset")
    parser.add_argument("--n",    type=int, default=600,
                        help="Number of samples (default: 600)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (default: 1)")
    parser.add_argument("--out",  type=str, default="hull_dataset.csv",
                        help="Output CSV path (default: hull_dataset.csv)")
    args = parser.parse_args()

    H, R, V = generate_dataset(n_samples=args.n, seed=args.seed)
    save_csv(args.out, H, R, V)