"""
hull_surrogate.py
=================
Hybrid Kriging surrogate for hull MDO.

ARCHITECTURE
────────────
  Stage 1 — Geometry Kriging (GPR):   h → [V, S, CP, LCB]
  Stage 2 — Exact resistance formula: [V, S, CP, a1E, a1R, B_max] → R_T

WHY KRIGING INSTEAD OF A DNN?
──────────────────────────────
The geometry outputs (V, S, CP, LCB) are smooth functions of h — they are
integrals of the SAC spline. Kriging (Gaussian Process Regression) is the
classical optimal interpolator for smooth, low-noise functions. It:

  1. Provably minimizes mean squared prediction error under the assumed
     kernel (Best Linear Unbiased Predictor, BLUP property).
  2. Provides a prediction UNCERTAINTY (std) at each point — the DNN does
     not. This tells you how much to trust the surrogate at a given h.
  3. Requires no architecture choices, no learning rate tuning, no epochs.
     Hyperparameters (kernel length scales, noise level) are found by
     maximizing the marginal likelihood — fully automatic.
  4. With 400 training points and d=10, Kriging typically matches or
     beats a DNN because the DNN needs more data to overcome random
     initialisation and local minima.

KERNEL CHOICE: Matérn ν=2.5
─────────────────────────────
The Matérn kernel with ν=2.5 assumes the function is twice differentiable.
This matches the geometry outputs, which come from cubic spline integrals.
  - RBF (ν=∞): assumes infinitely smooth — too strong, can overfit.
  - Matérn ν=0.5 (Ornstein-Uhlenbeck): assumes only continuity — too weak.
  - Matérn ν=2.5: the standard choice for physical simulation surrogates.

One GPR is trained per output (V, S, CP, LCB). Each gets its own
length scales and noise level, optimized independently. This is better
than one shared model because each output has different smoothness.

UNCERTAINTY ESTIMATE
─────────────────────
predict_geo(h) returns both a mean prediction and a standard deviation.
The std tells you how far you are from the training data in input space:
  - std ≈ 0:    interpolating near a training point — prediction reliable
  - std >> 0:   extrapolating far from data — prediction less reliable

The optimizer can use this to avoid regions of high uncertainty.

SCALING: O(n³) TRAINING, O(n) PREDICTION
──────────────────────────────────────────
GPR training solves a linear system of size n×n.
At n=400: fast (~10-30s). At n=4000: slow (~hours).
Prediction is O(n) per query point — fast even for the optimizer.

Usage
─────
  python hull_surrogate.py --csv hull_dataset.csv
"""

import argparse
import sys
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, WhiteKernel, ConstantKernel as C
)
from sklearn.preprocessing import StandardScaler

try:
    from GenHullData import N_VARS, halton_leaped
    from Hull_truth_test import true_model, to_physical, FIXED
except ImportError as e:
    print(f"ERROR: {e}")
    print("  Need GenHullData.py and Hull_truth_test.py in the same folder.")
    sys.exit(1)

SEED = 42
np.random.seed(SEED)

GEO_NAMES  = ["V", "S", "CP", "LCB"]
GEO_LABELS = ["V  [m3]", "S  [m2]", "C_P  [-]", "LCB/L  [-]"]


# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

def load_full_csv(path: str) -> dict:
    """Load dataset CSV. Returns H (inputs) and geometry targets."""
    data = np.genfromtxt(path, delimiter=",", names=True,
                         dtype=float, encoding="utf-8")
    mask   = data["valid"] == 1
    n_drop = int(np.sum(~mask))
    if n_drop > 0:
        print(f"  Dropped {n_drop} invalid rows.")
    data = data[mask]

    H   = np.column_stack([data[f"h{i}"] for i in range(N_VARS)])
    R_T = data["R_T"].astype(float)
    geo = np.column_stack([
        data["V"].astype(float),
        data["S"].astype(float),
        data["CP_achieved"].astype(float),
        data["LCB_achieved"].astype(float),
    ])

    print(f"  Loaded {len(R_T)} valid samples from '{path}'")
    print(f"  R_T  in [{R_T.min():.2f}, {R_T.max():.2f}] N")
    for i, lbl in enumerate(GEO_LABELS):
        print(f"  {lbl:<12}  in [{geo[:,i].min():.4f}, {geo[:,i].max():.4f}]")
    return dict(H=H, R_T=R_T, geo=geo)


def split_data(H, R_T, geo, test_fraction=0.20, seed=SEED):
    """80/20 random split. Same permutation applied to all arrays."""
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(R_T))
    n_te = max(1, int(len(R_T) * test_fraction))
    tr, te = idx[n_te:], idx[:n_te]
    return (H[tr], R_T[tr], geo[tr],
            H[te], R_T[te], geo[te],
            len(tr), len(te))


# =============================================================================
# SECTION 2 — KRIGING (GPR) GEOMETRY SURROGATE
# =============================================================================

def build_kernel(n_features: int) -> object:
    """
    Build the Matérn ν=2.5 kernel with ARD (Automatic Relevance Determination).

    ARD means each input dimension gets its own length scale.
    This lets the kernel learn that, e.g., h[0] (CP) has a stronger
    influence on V than h[7] (run angle).

    The kernel is:
      k(x, x') = C · Matérn(x, x'; length_scales) + WhiteKernel(noise)

    where:
      C             — amplitude (signal variance)
      length_scales — one per input dimension (ARD)
      WhiteKernel   — captures observation noise / model error
    """
    return (
        C(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(
            length_scale=np.ones(n_features),
            length_scale_bounds=(0.01, 10.0),   # bounded to avoid warnings
            nu=2.5
        )
        + WhiteKernel(
            noise_level=1e-3,
            noise_level_bounds=(1e-6, 1e-1)
        )
    )


def fit_geometry_kriging(
    H_train: np.ndarray,
    geo_train: np.ndarray,
    n_restarts: int = 5,
    verbose: bool = True,
) -> tuple:
    """
    Fit one GPR per geometry output.

    Returns
    -------
    models  : list of 4 fitted GaussianProcessRegressor objects
    scalers : list of 4 StandardScaler objects (one per output)

    Why StandardScaler on outputs?
    GPR with normalize_y=True internally centers and scales the output,
    which improves the marginal likelihood optimization. We do this
    explicitly so we can control it and apply the same scaling at
    prediction time.

    Why StandardScaler on inputs?
    The inputs H are already in [0,1] so input scaling is not strictly
    needed, but it makes the length_scale_bounds more interpretable
    (length scale of 1.0 = the full input range).
    """
    models, scalers = [], []

    for i, name in enumerate(GEO_NAMES):
        y = geo_train[:, i]

        # Scale output to zero mean, unit variance
        scaler = StandardScaler()
        y_s    = scaler.fit_transform(y.reshape(-1, 1)).flatten()

        kernel = build_kernel(H_train.shape[1])
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            normalize_y=False,    # we already scaled manually
            random_state=SEED,
            alpha=1e-8,           # numerical stability (jitter)
        )

        if verbose:
            print(f"  Fitting GPR for {name}...", end=" ", flush=True)
        t0 = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress convergence warnings
            gpr.fit(H_train, y_s)

        t1 = time.time()
        if verbose:
            print(f"done ({t1-t0:.1f}s)  "
                  f"log-likelihood={gpr.log_marginal_likelihood_value_:.2f}")

        models.append(gpr)
        scalers.append(scaler)

    return models, scalers


def predict_geo(
    H: np.ndarray,
    models: list,
    scalers: list,
    return_std: bool = False,
) -> tuple:
    """
    Predict geometry outputs [V, S, CP, LCB] for a batch of designs H.

    Parameters
    ----------
    H          : (n, 10)  normalised design matrix
    models     : list of 4 GPR models (from fit_geometry_kriging)
    scalers    : list of 4 StandardScalers
    return_std : bool  if True, also return prediction uncertainty

    Returns
    -------
    geo_pred : (n, 4)  predicted geometry [V, S, CP, LCB]
    geo_std  : (n, 4)  prediction std [m3, m2, -, -]  (only if return_std)

    WHAT DOES THE UNCERTAINTY MEAN?
    The std is the GPR's estimate of its own prediction error at each point.
    It is ZERO at training points (GPR interpolates exactly in the noise-free
    limit) and grows as you move away from training data.
    For the optimizer: high std regions are less trustworthy — the optimizer
    should prefer designs where the surrogate is confident.
    """
    single = (H.ndim == 1)
    if single:
        H = H.reshape(1, -1)

    preds, stds = [], []
    for gpr, scaler in zip(models, scalers):
        if return_std:
            p_s, s_s = gpr.predict(H, return_std=True)
            # Inverse-transform: std scales by scaler.scale_
            p = scaler.inverse_transform(p_s.reshape(-1, 1)).flatten()
            s = s_s * scaler.scale_[0]
        else:
            p_s = gpr.predict(H, return_std=False)
            p   = scaler.inverse_transform(p_s.reshape(-1, 1)).flatten()
            s   = np.zeros(len(p))
        preds.append(p)
        stds.append(s)

    geo_pred = np.column_stack(preds)
    geo_std  = np.column_stack(stds)

    if single:
        if return_std:
            return geo_pred[0], geo_std[0]
        return geo_pred[0]

    if return_std:
        return geo_pred, geo_std
    return geo_pred


# =============================================================================
# SECTION 3 — HYBRID RESISTANCE FORMULA  (exact, vectorized)
# =============================================================================

def hybrid_resistance(
    S: np.ndarray,
    CP: np.ndarray,
    V: np.ndarray,
    a1E: np.ndarray,
    a1R: np.ndarray,
    B_max: np.ndarray,
) -> tuple:
    """
    Exact ITTC-57 + wave resistance formula, vectorized over arrays.
    Identical physics to Hull_truth_test.py compute_resistance().

    The ONLY approximation in the full hybrid pipeline comes from the
    geometry Kriging above. This formula itself is exact.
    """
    L, U, rho, nu, g = (FIXED["L"], FIXED["U"], FIXED["rho"],
                         FIXED["nu"], FIXED["g"])

    Re  = max(U * L / nu, 1e4)
    F_n = U / np.sqrt(g * L)
    C_F = 0.075 / (np.log10(Re) - 2.0) ** 2

    # Mean draft approximation from displacement volume
    # T_mean ≈ V / (L · B_max · 0.6)  — avoids needing the full T(x) array
    T_approx = V / np.maximum(L * B_max * 0.6, 1e-9)
    CB = np.clip(V / np.maximum(L * B_max * T_approx, 1e-9), 0.4, 0.9)
    LB = L / np.maximum(B_max, 1e-9)
    k  = np.clip(0.93 + 0.4871 * (CB / LB**0.1228) * (B_max / L)**0.6906,
                 0.05, 0.50)

    R_F = 0.5 * rho * U**2 * S * C_F * (1.0 + k)

    alpha_ref  = 20.0
    f_angle    = (1.0 + (a1E/alpha_ref)**1.5 +
                  0.5 * (a1R/alpha_ref)**1.5) / 2.0 + 0.5
    C_W        = 0.55 * F_n**4 * CP**2 * f_angle
    R_W        = 0.5 * rho * U**2 * S * C_W

    return R_F + R_W, R_F, R_W


def predict_hybrid(
    H: np.ndarray,
    models: list,
    scalers: list,
    return_std: bool = False,
) -> dict:
    """
    Full hybrid pipeline: h → Kriging geometry → exact R_T.

    This is the function called by the optimizer.

    Parameters
    ----------
    H          : (n,10) or (10,)  normalised design vectors
    models     : GPR models from fit_geometry_kriging
    scalers    : StandardScalers from fit_geometry_kriging
    return_std : bool  if True, include geometry uncertainty estimates

    Returns
    -------
    dict with R_T, R_F, R_W, V, S, CP, LCB
    (and V_std, S_std, CP_std, LCB_std if return_std=True)
    """
    single = (H.ndim == 1)
    if single:
        H = H.reshape(1, -1)

    # Stage 1: Kriging geometry prediction
    if return_std:
        geo, geo_std = predict_geo(H, models, scalers, return_std=True)
    else:
        geo     = predict_geo(H, models, scalers, return_std=False)
        geo_std = None

    V, S, CP, LCB = geo[:,0], geo[:,1], geo[:,2], geo[:,3]

    # Stage 2: extract a1E, a1R, B_max directly from h (no approximation)
    v     = to_physical(H)
    a1E   = v[:, 6]
    a1R   = v[:, 7]
    B_max = v[:, 8]

    # Stage 3: exact resistance formula
    R_T, R_F, R_W = hybrid_resistance(S, CP, V, a1E, a1R, B_max)

    out = dict(R_T=R_T, R_F=R_F, R_W=R_W,
               V=V, S=S, CP=CP, LCB=LCB)

    if return_std and geo_std is not None:
        out["V_std"]   = geo_std[:, 0]
        out["S_std"]   = geo_std[:, 1]
        out["CP_std"]  = geo_std[:, 2]
        out["LCB_std"] = geo_std[:, 3]

    if single:
        return {k: float(v[0]) if isinstance(v, np.ndarray) else v
                for k, v in out.items()}
    return out


# =============================================================================
# SECTION 4 — METRICS
# =============================================================================

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                 label: str = "") -> dict:
    """MAE, MAE%, MSE, R² for one output."""
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    mae     = float(np.mean(np.abs(y_true - y_pred)))
    mae_pct = float(np.mean(np.abs(y_true - y_pred) /
                             (np.abs(y_true) + 1e-12)) * 100)
    mse     = float(np.mean((y_true - y_pred) ** 2))
    ss_res  = float(np.sum((y_true - y_pred) ** 2))
    ss_tot  = float(np.sum((y_true - y_true.mean()) ** 2))
    r2      = float(1.0 - ss_res / (ss_tot + 1e-12))
    return dict(label=label, MAE=mae, MAE_pct=mae_pct, MSE=mse, R2=r2)


def print_table(rows: list, title: str) -> None:
    sep = "-" * 66
    print(f"\n{sep}\n  {title}\n{sep}")
    print(f"  {'Output':<18} {'MAE':>10} {'MAE%':>8} {'R2':>8}  {'Pass?':>6}")
    print(sep)
    for m in rows:
        flag = "OK" if m["R2"] > 0.95 else ("~" if m["R2"] > 0.85 else "FAIL")
        print(f"  {m['label']:<18} {m['MAE']:>10.5f} "
              f"{m['MAE_pct']:>7.2f}%  {m['R2']:>7.4f}  {flag}")
    print(sep)


# =============================================================================
# SECTION 5 — FRESH TRUE-MODEL VALIDATION
# =============================================================================

def fresh_validation(models: list, scalers: list,
                     n_samples: int = 50) -> dict:
    """
    Evaluate n_samples completely new designs (different Halton seed)
    through both the true model and the hybrid surrogate, then compare.

    This is stronger than test-set validation because these designs
    were never seen in any form during training.
    """
    print(f"\n  Generating {n_samples} fresh designs (seed_offset=999000)...")
    H_new = halton_leaped(n_samples, d=N_VARS, seed_offset=999_000)

    R_t, V_t, S_t, CP_t, LCB_t = [], [], [], [], []
    for h in H_new:
        try:
            r = true_model(h)
            R_t.append(r["R_T"]);   V_t.append(r["V"])
            S_t.append(r["S"]);     CP_t.append(r["CP"])
            LCB_t.append(r["LCB"])
        except Exception:
            [lst.append(np.nan) for lst in [R_t, V_t, S_t, CP_t, LCB_t]]

    ok    = ~np.isnan(R_t)
    H_v   = H_new[ok]
    R_t   = np.array(R_t)[ok]
    V_t   = np.array(V_t)[ok]
    S_t   = np.array(S_t)[ok]
    CP_t  = np.array(CP_t)[ok]
    LCB_t = np.array(LCB_t)[ok]
    print(f"  {ok.sum()}/{n_samples} evaluations succeeded.")

    pred = predict_hybrid(H_v, models, scalers, return_std=True)

    return dict(
        R_true=R_t,    R_pred=pred["R_T"],
        V_true=V_t,    V_pred=pred["V"],    V_std=pred.get("V_std"),
        S_true=S_t,    S_pred=pred["S"],    S_std=pred.get("S_std"),
        CP_true=CP_t,  CP_pred=pred["CP"],  CP_std=pred.get("CP_std"),
        LCB_true=LCB_t,LCB_pred=pred["LCB"],LCB_std=pred.get("LCB_std"),
    )


# =============================================================================
# SECTION 6 — PLOTTING  (all figures built before plt.show())
# =============================================================================

def fig_parity(val: dict) -> plt.Figure:
    """
    Five parity plots. For R_T, residual size = geometry Kriging error
    propagated through the exact formula — no resistance model error.
    """
    pairs = [
        (val["R_true"], val["R_pred"],  "R_T [N]",    "steelblue",
         "Kriging + exact formula"),
        (val["V_true"],  val["V_pred"], "V [m3]",     "seagreen",   "Kriging"),
        (val["S_true"],  val["S_pred"], "S [m2]",     "darkorange", "Kriging"),
        (val["CP_true"], val["CP_pred"],"C_P [-]",    "purple",     "Kriging"),
        (val["LCB_true"],val["LCB_pred"],"LCB/L [-]", "firebrick",  "Kriging"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle(
        "Parity plots — Kriging geometry surrogate + exact resistance formula\n"
        "x = true model,  y = hybrid surrogate.  Perfect = on the diagonal.",
        fontsize=11, fontweight="bold",
    )
    for ax, (y_t, y_p, label, color, source) in zip(axes, pairs):
        m   = calc_metrics(y_t, y_p, label)
        ax.scatter(y_t, y_p, s=22, alpha=0.7, color=color, edgecolors="none")
        lo  = min(y_t.min(), y_p.min())
        hi  = max(y_t.max(), y_p.max())
        pad = (hi - lo) * 0.05
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "k--", lw=1.2,
                label="Perfect")
        ax.set_xlabel(f"True  {label}", fontsize=8)
        ax.set_ylabel(f"Surrogate  {label}", fontsize=8)
        ax.set_title(f"{label}\nR2={m['R2']:.4f}  MAE={m['MAE_pct']:.2f}%"
                     f"\n[{source}]", fontsize=8)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def fig_residuals(val: dict) -> plt.Figure:
    """Residual plots: (surrogate − true) vs true value."""
    pairs = [
        (val["R_true"], val["R_pred"],  "R_T [N]",    "steelblue"),
        (val["V_true"],  val["V_pred"], "V [m3]",     "seagreen"),
        (val["S_true"],  val["S_pred"], "S [m2]",     "darkorange"),
        (val["CP_true"], val["CP_pred"],"C_P [-]",    "purple"),
        (val["LCB_true"],val["LCB_pred"],"LCB/L [-]", "firebrick"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle(
        "Residual plots — (surrogate − true model)\n"
        "Random scatter around zero = good.  Systematic slope = bias.",
        fontsize=11, fontweight="bold",
    )
    for ax, (y_t, y_p, label, color) in zip(axes, pairs):
        res = y_p - y_t
        ax.scatter(y_t, res, s=22, alpha=0.7, color=color, edgecolors="none")
        ax.axhline(0, color="black", lw=1.0, ls="--")
        ax.set_xlabel(f"True  {label}", fontsize=8)
        ax.set_ylabel("Residual (surr − true)", fontsize=8)
        ax.set_title(f"{label}\nbias={res.mean():.5f}  std={res.std():.5f}",
                     fontsize=9)
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def fig_uncertainty(val: dict) -> plt.Figure:
    """
    Kriging uncertainty plots: prediction std vs absolute error.

    A well-calibrated GPR should show that points with larger std
    also have larger actual errors — the std is a reliable indicator
    of where the model is uncertain.  This is unique to Kriging and
    is not available from a DNN.
    """
    pairs = [
        (val.get("V_std"),   val["V_true"],   val["V_pred"],
         "V [m3]",   "seagreen"),
        (val.get("S_std"),   val["S_true"],   val["S_pred"],
         "S [m2]",   "darkorange"),
        (val.get("CP_std"),  val["CP_true"],  val["CP_pred"],
         "C_P [-]",  "purple"),
        (val.get("LCB_std"), val["LCB_true"], val["LCB_pred"],
         "LCB [-]",  "firebrick"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle(
        "Kriging uncertainty calibration\n"
        "x = predicted std (Kriging confidence),  "
        "y = actual |error|.  Good calibration: positive correlation.",
        fontsize=11, fontweight="bold",
    )
    for ax, (std, y_t, y_p, label, color) in zip(axes, pairs):
        if std is None:
            ax.text(0.5, 0.5, "No std available",
                    ha="center", transform=ax.transAxes)
            continue
        err = np.abs(y_p - y_t)
        ax.scatter(std, err, s=22, alpha=0.7, color=color, edgecolors="none")
        ax.set_xlabel("Kriging std (prediction uncertainty)", fontsize=8)
        ax.set_ylabel(f"|error|  {label}", fontsize=8)
        ax.set_title(f"{label}\ncorr={np.corrcoef(std,err)[0,1]:.3f}",
                     fontsize=9)
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def fig_accuracy_summary(val: dict) -> plt.Figure:
    """Bar chart of MAE% for all outputs."""
    names  = ["R_T", "V", "S", "C_P", "LCB/L"]
    trues  = [val["R_true"], val["V_true"], val["S_true"],
              val["CP_true"], val["LCB_true"]]
    preds  = [val["R_pred"], val["V_pred"], val["S_pred"],
              val["CP_pred"], val["LCB_pred"]]
    colors = ["steelblue","seagreen","darkorange","purple","firebrick"]
    maes   = [float(np.mean(np.abs(t-p)/(np.abs(t)+1e-12))*100)
              for t, p in zip(trues, preds)]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Surrogate accuracy summary — MAE% per output",
                 fontsize=11, fontweight="bold")
    bars = ax.bar(names, maes, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(5.0, color="red",   lw=1.0, ls="--", label="5%  (good)")
    ax.axhline(2.0, color="green", lw=1.0, ls="--", label="2%  (excellent)")
    for bar, pct in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{pct:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.text(0, maes[0] + 0.4, "Kriging\n+ exact\nformula",
            ha="center", fontsize=7, color="steelblue", style="italic")
    ax.set_ylabel("MAE [%]"); ax.set_ylim(0, max(maes)*1.6)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 7 — SAVE / LOAD
# =============================================================================

def save_surrogate(models: list, scalers: list,
                   path: str = "surrogate_kriging.pkl") -> None:
    """
    Save GPR models and scalers using joblib.
    joblib is the standard way to serialise sklearn objects.
    """
    joblib.dump({"models": models, "scalers": scalers}, path)
    print(f"  Saved Kriging surrogate -> {path}")
    print(f"  (Resistance uses exact formula — no file needed for it)")


def load_surrogate(path: str = "surrogate_kriging.pkl") -> tuple:
    """Load surrogate. Called by the optimizer."""
    data    = joblib.load(path)
    return data["models"], data["scalers"]


# =============================================================================
# SECTION 8 — MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Kriging hull surrogate (geometry GPR + exact formula)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",        default="hull_dataset.csv")
    parser.add_argument("--n_restarts", type=int, default=5,
                        help="GPR kernel hyperparameter restarts "
                             "(more = better but slower)")
    parser.add_argument("--val_n",      type=int, default=50,
                        help="Fresh true-model designs for final validation")
    parser.add_argument("--no_save",    action="store_true")
    args = parser.parse_args()

    SEP = "=" * 66

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}\nSTEP 1 — Load dataset\n{SEP}")
    ds = load_full_csv(args.csv)
    H, R_T, geo = ds["H"], ds["R_T"], ds["geo"]

    # ── 2. Split ──────────────────────────────────────────────────────────────
    print(f"\n{SEP}\nSTEP 2 — Train/test split  (80% / 20%)\n{SEP}")
    H_tr, R_tr, geo_tr, H_te, R_te, geo_te, n_tr, n_te = split_data(
        H, R_T, geo)
    print(f"  Train: {n_tr}   Test: {n_te}")

    # ── 3. Fit Kriging ────────────────────────────────────────────────────────
    print(f"\n{SEP}\nSTEP 3 — Fit Kriging geometry surrogate\n{SEP}")
    print(f"  Kernel: ConstantKernel * Matern(nu=2.5, ARD) + WhiteKernel")
    print(f"  n_restarts_optimizer = {args.n_restarts}")
    print(f"  Fitting 4 separate GPRs (one per output)...")
    t0 = time.time()
    models, scalers = fit_geometry_kriging(
        H_tr, geo_tr, n_restarts=args.n_restarts, verbose=True)
    print(f"  Total fit time: {time.time()-t0:.1f}s")

    # Print optimized kernel parameters for each output
    print(f"\n  Optimized kernel parameters:")
    for name, gpr in zip(GEO_NAMES, models):
        noise = gpr.kernel_.k2.noise_level
        amp   = gpr.kernel_.k1.k1.constant_value
        print(f"    {name:<5}  amplitude={amp:.3f}  noise={noise:.2e}")

    # ── 4. Test-set validation — geometry ────────────────────────────────────
    print(f"\n{SEP}\nSTEP 4 — Test-set validation  (geometry Kriging)\n{SEP}")
    geo_pred_te = predict_geo(H_te, models, scalers, return_std=False)
    geo_rows    = [calc_metrics(geo_te[:,i], geo_pred_te[:,i], GEO_LABELS[i])
                   for i in range(4)]
    print_table(geo_rows, "Geometry Kriging — test set")

    # ── 5. Test-set validation — hybrid R_T ──────────────────────────────────
    print(f"\n{SEP}\nSTEP 5 — Test-set validation  (hybrid R_T)\n{SEP}")
    pred_te = predict_hybrid(H_te, models, scalers)
    print_table([calc_metrics(R_te, pred_te["R_T"], "R_T [N] (hybrid)")],
                "Hybrid resistance — test set")
    print("  R_T accuracy reflects geometry Kriging accuracy — formula is exact.")

    # ── 6. Fresh true-model validation ───────────────────────────────────────
    print(f"\n{SEP}\nSTEP 6 — Fresh true-model validation  "
          f"({args.val_n} new designs)\n{SEP}")
    val = fresh_validation(models, scalers, n_samples=args.val_n)

    print_table([calc_metrics(val["V_true"],   val["V_pred"],   "V [m3]"),
                 calc_metrics(val["S_true"],   val["S_pred"],   "S [m2]"),
                 calc_metrics(val["CP_true"],  val["CP_pred"],  "C_P [-]"),
                 calc_metrics(val["LCB_true"], val["LCB_pred"], "LCB/L [-]")],
                "Geometry Kriging — fresh designs")

    print_table([calc_metrics(val["R_true"], val["R_pred"], "R_T [N] (hybrid)")],
                "Hybrid resistance — fresh designs")

    # ── 7. Build all figures ──────────────────────────────────────────────────
    print(f"\n{SEP}\nSTEP 7 — Generating plots\n{SEP}")
    figs = {
        "surrogate_parity.png":      fig_parity(val),
        "surrogate_residuals.png":   fig_residuals(val),
        "surrogate_uncertainty.png": fig_uncertainty(val),
        "surrogate_accuracy.png":    fig_accuracy_summary(val),
    }
    for fname, fig in figs.items():
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {fname}")

    # ── 8. Save ───────────────────────────────────────────────────────────────
    if not args.no_save:
        print(f"\n{SEP}\nSTEP 8 — Save\n{SEP}")
        save_surrogate(models, scalers)

    # ── Summary ───────────────────────────────────────────────────────────────
    r2_rt = calc_metrics(val["R_true"], val["R_pred"])["R2"]
    r2_v  = calc_metrics(val["V_true"], val["V_pred"])["R2"]
    print(f"\n{SEP}\nSUMMARY\n{SEP}")
    print(f"  Hybrid R_T:  R2 = {r2_rt:.4f}")
    print(f"  Geometry V:  R2 = {r2_v:.4f}")
    if r2_rt > 0.90:
        print("  PASS — surrogate accurate.  Safe to run optimizer.")
    elif r2_rt > 0.80:
        print("  ADEQUATE — optimizer will work. More data would improve it.")
    else:
        print("  FAIL — check geometry Kriging accuracy above.")
    print(f"\n  Next step: python hull_optimizer.py")
    print(SEP)
    print("\n  All figures open. Close them to exit.")
    plt.show()


if __name__ == "__main__":
    main()