"""
hull_generate_data.py
=====================
Generates the training dataset for the hull MDO surrogate model.

ROLE OF THIS SCRIPT IN THE PIPELINE
─────────────────────────────────────
  hull_true_model.py   ← defines the physics: h → R_T, V
  hull_generate_data.py  ← THIS FILE: calls the true model at N sampled
                           designs, saves (h, R_T, V) to CSV
  hull_surrogate.py    ← trains DNN surrogate on the CSV
  hull_optimizer.py    ← optimizes using the surrogate

This script does NOT do any machine learning. It only:
  1. Samples N design vectors h ∈ [0,1]^10 using the leaped Halton sequence
  2. Evaluates each h through the true model
  3. Saves the results to a CSV file

CLARIFYING YOUR QUESTIONS
──────────────────────────
Q: "Does the true model output optimized parameters?"
A: No. The true model is a pure evaluator: given h, compute R_T and V.
   The design vector h is the INPUT, not an output. The optimizer (later)
   will search for the h that minimizes R_T. This script just calls the
   true model at many different h values to collect training data.

Q: "Is the surrogate a polynomial approximation?"
A: Not in our pipeline. Following Kim et al. (2024), we use a Deep Neural
   Network (DNN). A polynomial fails badly in 10 dimensions because it
   cannot capture cross-parameter interactions (e.g., how C_P and α_1E
   together affect resistance). A DNN handles this naturally.
   The data generation script is the same regardless of surrogate choice
   — the CSV format is surrogate-agnostic.

Q: "Can we incorporate the characteristic curve control points from Kim et al.?"
A: The key contribution of Kim et al. we incorporate here is their
   SAMPLING STRATEGY (leaped Halton sequence, Section 3.1) and their
   PIPELINE STRUCTURE (sample → CFD → DNN → GA). Their specific control
   point parameterization (4 characteristic curves, 29 parameters) is
   a different geometry model from our SAC-based model and cannot be
   directly merged — the parameters mean different things.
   What CAN be combined: we record the SAC control point coordinates
   (X_0E, X_0R, B_max, etc.) alongside R_T and V, so the surrogate
   can optionally learn geometry → performance in the CP-space style
   the paper describes.

SAMPLING: LEAPED HALTON SEQUENCE  (Kim et al. 2024, Section 3.1)
──────────────────────────────────────────────────────────────────
A standard Halton sequence in d=10 dimensions can develop partial linear
correlations between variables when many dimensions are used. Kim et al.
fix this by using only every 139th value of the sequence (the "leaped"
Halton). This breaks the inter-variable correlations while preserving the
uniform space-filling property.

For d=10 (our case), the standard Halton is already less correlated than
their d=29 case, but we apply the same leap factor as a best practice.

HOW MANY SAMPLES DO WE NEED?
─────────────────────────────
Kim et al. used 896 samples for d=29. A common rule of thumb for
surrogate modeling is 10×d to 30×d samples. For d=10:
  Minimum:  100 samples  (10×d, risky — surrogate may underfit)
  Adequate: 200 samples  (20×d, good for RBF or GPR surrogates)
  Good:     500 samples  (50×d, good for DNN surrogates)
  Best:    1000 samples  (100×d, allows high-accuracy DNN)

Default in this script: 500 samples (a good balance of quality vs time).
Each true model evaluation takes ~0.05-0.2s, so 500 samples ≈ 1-2 min.

CSV OUTPUT FORMAT
─────────────────
Columns:
  h0 .. h9        — normalised design variables ∈ [0,1]  (10 columns)
  CP, XLCB,       — physical design variables (for interpretability)
  X0E, X0R,
  a0E, a0R,
  a1E, a1R,
  Bmax, xBmax
  R_T             — total resistance [N]          ← primary target
  R_F             — frictional resistance [N]     ← diagnostic
  R_W             — wave resistance [N]            ← diagnostic
  V               — displacement volume [m³]      ← constraint target
  S               — wetted surface area [m²]      ← diagnostic
  CP_achieved     — achieved prismatic coeff      ← verify vs input
  LCB_achieved    — achieved LCB/L                ← verify vs input
  valid           — 1 if evaluation succeeded, 0 if it raised an error

Usage
─────
  # Default: 500 samples, save to hull_dataset.csv
  python hull_generate_data.py

  # Custom: 1000 samples, custom output file
  python hull_generate_data.py --n 1000 --out my_data.csv --seed 42

  # Quick test: 20 samples to verify everything runs
  python hull_generate_data.py --n 20 --out test_data.csv
"""

import argparse
import csv
import time
import numpy as np
import sys
import os

# ── Import the true model ──────────────────────────────────────────────────
# hull_true_model.py must be in the same directory as this script.
try:
    from Hull_truth_test import (
        true_model, to_physical, N_VARS, BOUNDS, FIXED, default_h
    )
except ImportError:
    print("ERROR: hull_true_model.py not found in the current directory.")
    print("       Make sure both files are in the same folder.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LEAPED HALTON SEQUENCE  (Kim et al. 2024, Section 3.1)
# ─────────────────────────────────────────────────────────────────────────────

# First 10 prime numbers used as bases for the 10-dimensional Halton sequence
_HALTON_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

# Leap factor: use every 139th value (same as Kim et al.).
# This breaks inter-dimensional correlations in high-d Halton sequences.
_LEAP = 139


def _halton_scalar(index: int, base: int) -> float:
    """
    Compute a single value of the Halton sequence.

    The Halton sequence at index `i` in base `b` is computed by reversing
    the base-b representation of i and placing it after the decimal point.
    Example (base 2):  i=1 → 0.1₂ = 0.5,  i=2 → 0.01₂ = 0.25, ...

    Parameters
    ----------
    index : int   sequence index (1-based)
    base  : int   prime base for this dimension

    Returns
    -------
    float ∈ (0, 1)
    """
    result = 0.0
    denominator = 1.0
    n = index
    while n > 0:
        denominator *= base
        n, remainder = divmod(n, base)
        result += remainder / denominator
    return result


def halton_leaped(n_samples: int, d: int = None, leap: int = _LEAP,
                  seed_offset: int = 0) -> np.ndarray:
    """
    Generate n_samples points in [0,1]^d using the leaped Halton sequence.

    Following Kim et al. (2024): we use only every `leap`-th value of the
    Halton sequence to avoid partial linear correlations between dimensions.

    Parameters
    ----------
    n_samples    : int   number of samples to generate
    d            : int   dimensionality (default: N_VARS from true model)
    leap         : int   leap factor (default: 139, same as Kim et al.)
    seed_offset  : int   starting offset in the sequence (for reproducibility
                         and for extending an existing dataset)

    Returns
    -------
    X : (n_samples, d)  array of samples, each row ∈ [0,1]^d
    """
    if d is None:
        d = N_VARS

    if d > len(_HALTON_PRIMES):
        raise ValueError(
            f"Leaped Halton implemented for d ≤ {len(_HALTON_PRIMES)}, "
            f"got d={d}. Add more primes to _HALTON_PRIMES."
        )

    X = np.zeros((n_samples, d))
    for i in range(n_samples):
        # The leaped index: we skip `leap` values between samples
        seq_idx = seed_offset + (i + 1) * leap
        for j in range(d):
            X[i, j] = _halton_scalar(seq_idx, _HALTON_PRIMES[j])

    return X


def halton_standard(n_samples: int, d: int = None,
                    seed: int = 0) -> np.ndarray:
    """
    Standard (non-leaped) Halton sequence using scipy.
    Used as fallback comparison or for small d.
    """
    if d is None:
        d = N_VARS
    try:
        from scipy.stats import qmc
        sampler = qmc.Halton(d=d, scramble=True, seed=seed)
        return sampler.random(n=n_samples)
    except ImportError:
        print("  scipy not available, using leaped Halton instead.")
        return halton_leaped(n_samples, d=d)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

# Column definitions for the CSV
# h0..h9: normalised inputs
# physical parameters: readable interpretations of h
# R_T, R_F, R_W, V, S: outputs from true model
# CP_achieved, LCB_achieved: verify the SAC constraints were satisfied
# valid: flag (1=success, 0=the true model raised an exception)

_H_COLS     = [f"h{i}" for i in range(N_VARS)]
_V_NAMES    = ["CP", "XLCB", "X0E", "X0R", "a0E", "a0R",
               "a1E", "a1R", "Bmax", "xBmax"]
_OUT_COLS   = ["R_T", "R_F", "R_W", "V", "S", "CP_achieved",
               "LCB_achieved", "valid"]
ALL_COLUMNS = _H_COLS + _V_NAMES + _OUT_COLS


def evaluate_design(h: np.ndarray) -> dict:
    """
    Evaluate a single design vector h through the true model.

    Wraps true_model() in a try/except so that a single failed evaluation
    does not crash the entire data generation run.

    Parameters
    ----------
    h : (N_VARS,) normalised design vector ∈ [0,1]^N_VARS

    Returns
    -------
    dict with all CSV column values (see ALL_COLUMNS)
    """
    row = {}

    # Store normalised inputs
    for i, val in enumerate(h):
        row[f"h{i}"] = float(val)

    # Store physical parameter values (human-readable)
    v = to_physical(h)
    for name, val in zip(_V_NAMES, v):
        row[name] = float(val)

    # Evaluate the true model
    try:
        result = true_model(h)
        row["R_T"]          = float(result["R_T"])
        row["R_F"]          = float(result["resist"]["R_F"])
        row["R_W"]          = float(result["resist"]["R_W"])
        row["V"]            = float(result["V"])
        row["S"]            = float(result["S"])
        row["CP_achieved"]  = float(result["CP"])
        row["LCB_achieved"] = float(result["LCB"])
        row["valid"]        = 1
    except Exception as e:
        # Record failure — don't crash the whole run
        row["R_T"]          = float("nan")
        row["R_F"]          = float("nan")
        row["R_W"]          = float("nan")
        row["V"]            = float("nan")
        row["S"]            = float("nan")
        row["CP_achieved"]  = float("nan")
        row["LCB_achieved"] = float("nan")
        row["valid"]        = 0
        row["_error"]       = str(e)   # not written to CSV, just for logging

    return row


def generate_dataset(
    n_samples: int = 500,
    leap: int = _LEAP,
    seed_offset: int = 0,
    verbose: bool = True,
    print_every: int = 50,
) -> tuple[np.ndarray, list[dict]]:
    """
    Sample n_samples hull designs using the leaped Halton sequence and
    evaluate each through the true model.

    Parameters
    ----------
    n_samples   : int   number of designs to evaluate
    leap        : int   Halton leap factor (default 139)
    seed_offset : int   starting offset (use non-zero to extend existing data)
    verbose     : bool  print progress
    print_every : int   print a progress line every this many evaluations

    Returns
    -------
    H    : (n_samples, N_VARS)  normalised design matrix
    rows : list of dicts         one dict per sample (matches ALL_COLUMNS)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Hull dataset generation")
        print(f"  n_samples   = {n_samples}")
        print(f"  N_VARS      = {N_VARS}  (design space dimension)")
        print(f"  leap factor = {leap}  (Kim et al. 2024 recommendation)")
        print(f"  seed_offset = {seed_offset}")
        print(f"{'='*60}")
        print(f"\nSampling {n_samples} points with leaped Halton sequence...")

    # Generate the sample points
    H = halton_leaped(n_samples, d=N_VARS, leap=leap,
                      seed_offset=seed_offset)

    if verbose:
        print(f"  Generated {H.shape} sample matrix.")
        print(f"\nEvaluating true model...")
        print(f"  (Each dot = {print_every} evaluations)")

    rows = []
    n_failed = 0
    t_start = time.time()

    for i, h in enumerate(H):
        row = evaluate_design(h)
        rows.append(row)

        if row["valid"] == 0:
            n_failed += 1
            if verbose:
                err = row.get("_error", "unknown error")
                print(f"\n  [WARN] Sample {i}: failed — {err}")

        if verbose and (i + 1) % print_every == 0:
            elapsed = time.time() - t_start
            rate    = (i + 1) / elapsed
            eta     = (n_samples - i - 1) / rate
            print(f"  {i+1:5d}/{n_samples}  "
                  f"({100*(i+1)/n_samples:.0f}%)  "
                  f"{elapsed:.1f}s elapsed  "
                  f"ETA {eta:.1f}s")

    elapsed_total = time.time() - t_start

    # Filter valid rows for statistics
    valid_rows = [r for r in rows if r["valid"] == 1]
    R_vals = np.array([r["R_T"] for r in valid_rows])
    V_vals = np.array([r["V"]   for r in valid_rows])

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Done in {elapsed_total:.1f}s  ({elapsed_total/n_samples*1000:.1f} ms/sample)")
        print(f"  Valid: {len(valid_rows)}/{n_samples}   "
              f"Failed: {n_failed}")
        print(f"\n  R_T  [{R_vals.min():.3f}, {R_vals.max():.3f}] N   "
              f"mean={R_vals.mean():.3f} N   std={R_vals.std():.3f} N")
        print(f"  V    [{V_vals.min():.4f}, {V_vals.max():.4f}] m³  "
              f"mean={V_vals.mean():.4f} m³  std={V_vals.std():.4f} m³")
        print(f"{'─'*55}")

    return H, rows


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CSV I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(path: str, rows: list[dict], verbose: bool = True) -> None:
    """
    Write the dataset to a CSV file.

    Only the columns in ALL_COLUMNS are written.  The '_error' key
    (used internally for logging) is deliberately excluded.

    Parameters
    ----------
    path    : str path to output CSV
    rows    : list of dicts from generate_dataset()
    verbose : bool
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        valid = sum(1 for r in rows if r["valid"] == 1)
        print(f"\n  Saved {len(rows)} rows ({valid} valid) → {path}")


def load_csv(path: str,
             drop_invalid: bool = True,
             verbose: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a dataset CSV written by save_csv().

    Parameters
    ----------
    path         : str  path to CSV
    drop_invalid : bool if True, drop rows where valid==0 (default True)
    verbose      : bool

    Returns
    -------
    H   : (n, N_VARS)  normalised design matrix
    R_T : (n,)         total resistance [N]
    V   : (n,)         displacement volume [m³]

    Note: returns only H, R_T, V — the three columns needed for surrogate
    training.  For other columns (R_F, R_W, S, etc.) load manually with
    numpy.loadtxt or pandas.
    """
    data = np.genfromtxt(path, delimiter=",", names=True,
                         dtype=float, encoding="utf-8")

    if drop_invalid:
        mask = data["valid"] == 1
        n_dropped = int(np.sum(~mask))
        if verbose and n_dropped > 0:
            print(f"  [INFO] Dropped {n_dropped} invalid rows (valid==0).")
        data = data[mask]

    H   = np.column_stack([data[f"h{i}"] for i in range(N_VARS)])
    R_T = data["R_T"].astype(float)
    V   = data["V"].astype(float)

    if verbose:
        print(f"\n  Loaded {len(R_T)} valid rows from '{path}'")
        print(f"  R_T ∈ [{R_T.min():.3f}, {R_T.max():.3f}] N   "
              f"mean={R_T.mean():.3f} N")
        print(f"  V   ∈ [{V.min():.4f}, {V.max():.4f}] m³  "
              f"mean={V.mean():.4f} m³")

    return H, R_T, V


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — TRAIN / TEST SPLIT  (Kim et al. 2024, Section 3.3)
# ─────────────────────────────────────────────────────────────────────────────
#
# Kim et al. use an 80/20 train/test split — exactly as in our old pipeline.
# The split is done AFTER loading so that the same CSV can be used for
# multiple experiments with different split seeds.
#
# IMPORTANT: we split by RANDOM SHUFFLE, not chronologically.
# Why? The leaped Halton sequence fills space uniformly by construction,
# so chronological and random splits give equivalent spatial coverage.
# But random shuffle is standard practice and avoids any accidental
# structure in the sequence order.


def train_test_split(
    H: np.ndarray,
    R_T: np.ndarray,
    V: np.ndarray,
    test_fraction: float = 0.20,
    seed: int = 0,
) -> tuple:
    """
    Randomly split the dataset into training and test sets.

    Following Kim et al. (2024): 80% train, 20% test.

    Parameters
    ----------
    H, R_T, V      : arrays from load_csv()
    test_fraction  : float  fraction for test set (default 0.20)
    seed           : int    random seed for reproducibility

    Returns
    -------
    H_train, R_train, V_train, H_test, R_test, V_test
    """
    rng = np.random.default_rng(seed)
    n = len(R_T)
    idx = rng.permutation(n)

    n_test  = max(1, int(n * test_fraction))
    n_train = n - n_test

    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]

    return (H[train_idx],   R_T[train_idx], V[train_idx],
            H[test_idx],    R_T[test_idx],  V[test_idx])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — DIAGNOSTIC PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset_diagnostics(rows: list[dict],
                              save: str = None) -> None:
    """
    Four diagnostic plots to verify the generated dataset is well-behaved:

    (A) Design space coverage: scatter of first two h values.
        A well-sampled Halton dataset should show uniform coverage with
        no obvious clustering.

    (B) R_T distribution: histogram.
        Should be roughly bell-shaped.  A bimodal distribution suggests
        a discontinuity in the true model (worth investigating).

    (C) V distribution: histogram.
        Should be roughly uniform — V is a smooth function of C_P and B_max.

    (D) R_T vs V scatter.
        Shows the fundamental tradeoff: ships with larger displacement
        generally have more resistance.  If this looks random, there may
        be a bug in the true model.

    Parameters
    ----------
    rows : list of dicts from generate_dataset()
    save : str  optional filepath to save the figure
    """
    import matplotlib.pyplot as plt

    valid = [r for r in rows if r["valid"] == 1]
    if len(valid) == 0:
        print("No valid rows — cannot plot diagnostics.")
        return

    h0  = np.array([r["h0"]  for r in valid])
    h1  = np.array([r["h1"]  for r in valid])
    R   = np.array([r["R_T"] for r in valid])
    V   = np.array([r["V"]   for r in valid])
    CP  = np.array([r["CP"]  for r in valid])
    a1E = np.array([r["a1E"] for r in valid])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"Dataset diagnostics  ({len(valid)} valid samples)\n"
        f"R_T ∈ [{R.min():.2f}, {R.max():.2f}] N   "
        f"V ∈ [{V.min():.4f}, {V.max():.4f}] m³",
        fontsize=11, fontweight="bold",
    )

    # (A) Halton coverage: h0 (C_P) vs h1 (X_LCB)
    axes[0,0].scatter(h0, h1, s=8, alpha=0.45, color="steelblue",
                      edgecolors="none")
    axes[0,0].set_xlabel("h0  (normalised C_P)", fontsize=9)
    axes[0,0].set_ylabel("h1  (normalised X_LCB)", fontsize=9)
    axes[0,0].set_title("(A)  Design space coverage — h0 vs h1\n"
                         "(uniform fill = good Halton sampling)", fontsize=9)
    axes[0,0].grid(True, alpha=0.2)

    # (B) R_T histogram
    axes[0,1].hist(R, bins=30, color="steelblue", edgecolor="white",
                   linewidth=0.5, alpha=0.85)
    axes[0,1].axvline(R.mean(), color="darkorange", lw=1.5,
                       label=f"Mean = {R.mean():.2f} N")
    axes[0,1].set_xlabel("R_T  [N]", fontsize=9)
    axes[0,1].set_ylabel("Count", fontsize=9)
    axes[0,1].set_title("(B)  R_T distribution", fontsize=9)
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.2)

    # (C) V histogram
    axes[1,0].hist(V, bins=30, color="seagreen", edgecolor="white",
                   linewidth=0.5, alpha=0.85)
    axes[1,0].axvline(V.mean(), color="darkorange", lw=1.5,
                       label=f"Mean = {V.mean():.4f} m³")
    axes[1,0].set_xlabel("V  [m³]", fontsize=9)
    axes[1,0].set_ylabel("Count", fontsize=9)
    axes[1,0].set_title("(C)  Displacement volume distribution", fontsize=9)
    axes[1,0].legend(fontsize=8)
    axes[1,0].grid(True, alpha=0.2)

    # (D) R_T vs V — the fundamental resistance-displacement tradeoff
    sc = axes[1,1].scatter(V, R, s=12, alpha=0.45,
                            c=a1E, cmap="RdYlGn_r",
                            edgecolors="none")
    plt.colorbar(sc, ax=axes[1,1], label="α₁E entrance angle (°)")
    axes[1,1].set_xlabel("V  [m³]  — displacement volume", fontsize=9)
    axes[1,1].set_ylabel("R_T  [N]  — total resistance", fontsize=9)
    axes[1,1].set_title("(D)  R_T vs V  (color = entrance angle α₁E)\n"
                          "Red = blunt bow entry, Green = sharp entry",
                          fontsize=9)
    axes[1,1].grid(True, alpha=0.2)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved diagnostics → {save}")
    plt.show()


def check_cp_convergence(rows: list[dict]) -> None:
    """
    Verify that the true model's SAC solver correctly matched the target C_P.

    The true model internally iterates to match C_P. If the iteration
    converged, CP_achieved should be very close to the input CP.
    A large discrepancy means the SAC solver failed for that sample.

    Prints a summary of the C_P convergence error across all samples.
    """
    valid = [r for r in rows if r["valid"] == 1]
    if not valid:
        print("No valid rows.")
        return

    errors = [abs(r["CP_achieved"] - r["CP"]) for r in valid]
    errors = np.array(errors)

    print(f"\n  C_P convergence check  ({len(valid)} valid samples):")
    print(f"    Mean |CP_achieved - CP_target| = {errors.mean():.4f}")
    print(f"    Max  |CP_achieved - CP_target| = {errors.max():.4f}")
    print(f"    Fraction within 0.01: "
          f"{100*np.mean(errors < 0.01):.0f}%")
    print(f"    Fraction within 0.05: "
          f"{100*np.mean(errors < 0.05):.0f}%")

    if errors.mean() > 0.03:
        print("\n  WARNING: Large C_P errors detected.")
        print("  This means the SAC solver is not converging reliably.")
        print("  Consider tightening the tolerance in build_sac().")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate hull MDO surrogate training dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n", type=int, default=500,
        help="Number of samples to generate. "
             "Recommended: 200 (quick), 500 (good), 1000 (best).",
    )
    parser.add_argument(
        "--out", type=str, default="hull_dataset.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--leap", type=int, default=_LEAP,
        help=f"Halton leap factor (Kim et al. use {_LEAP}).",
    )
    parser.add_argument(
        "--seed_offset", type=int, default=0,
        help="Starting offset in the Halton sequence. "
             "Use non-zero to generate a disjoint extension of an "
             "existing dataset (e.g. --seed_offset 500 to add 500 more "
             "samples without overlap).",
    )
    parser.add_argument(
        "--no_plot", action="store_true",
        help="Skip diagnostic plots.",
    )
    parser.add_argument(
        "--plot_out", type=str, default="dataset_diagnostics.png",
        help="File path for the diagnostic plot.",
    )
    args = parser.parse_args()

    # ── Generate ──────────────────────────────────────────────────────────────
    H, rows = generate_dataset(
        n_samples=args.n,
        leap=args.leap,
        seed_offset=args.seed_offset,
        verbose=True,
        print_every=max(1, args.n // 10),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    save_csv(args.out, rows, verbose=True)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    check_cp_convergence(rows)

    if not args.no_plot:
        print(f"\nGenerating diagnostic plots → {args.plot_out}")
        plot_dataset_diagnostics(rows, save=args.plot_out)

    # ── Summary ───────────────────────────────────────────────────────────────
    valid = sum(1 for r in rows if r["valid"] == 1)
    print(f"\n{'='*60}")
    print(f"Dataset generation complete.")
    print(f"  Total samples:  {len(rows)}")
    print(f"  Valid samples:  {valid}")
    print(f"  Output file:    {args.out}")
    print(f"\nNext step:  python hull_surrogate.py --csv {args.out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()