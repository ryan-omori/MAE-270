"""
truth_resistance_model.py
=========================
Canonical hull physics model.

  h (29,) normalized [0,1]  →  R_total [N],  V [m³]

This is the ONLY place resistance and volume are computed.
Both generate_dataset.py and hull_mdo_surrogate.py import from here.

Hull geometry
-------------
PROFILE VIEW (side, y=0 plane):
  Deck line : P1(0,0,F) ──────────────────── P5(L,0,F)   flat, FIXED
  Keel line : P1 → P2 → P3 → P4 → P5                    Bézier, optimised

  P1 = (0,   0, F)    bow  top          FIXED
  P2 = (0,   0, h1)   bow  entry depth  h1 controls z
  P3 = (h2,  0, -T)   keel forward      h2 controls x
  P4 = (h3,  0, -T)   keel aft          h3 controls x
  P5 = (L,   0, F)    stern top         FIXED

DECK EDGE (top, plan view):
  D1 → D2 → D3 → D4   Bézier beam planform  (h4..h9)

KEEL PLAN (bottom, z=-T):
  B1 → B2 → B3 → B4   Bézier keel width planform  (h10..h15)

FRONT VIEW (bow cross-section, y-z plane at x=0):
  FK = (0, 0,     -h16·T)   keel depth at bow
  FW = (0, h17·B,  0   )   hull half-width at waterline
"""

import numpy as np
from scipy.special import comb

# ── Fixed hull dimensions ─────────────────────────────────────────────────────
CONST = dict(L=1.00, B=0.375, T=0.25, F=0.30)

# ── Parameter bounds: maps h[i] in [0,1] to physical range ───────────────────
H_BOUNDS = np.array([
    [0.00, 0.20],   # h1 : P2 z   bow entry height
    [0.05, 0.30],   # h2 : P3 x   keel fwd start
    [0.70, 0.95],   # h3 : P4 x   keel aft end
    [0.10, 0.35],   # h4 : D2 x   deck edge ctrl pt 2
    [0.60, 1.00],   # h5 : D2 y
    [0.40, 0.60],   # h6 : D3 x   deck edge ctrl pt 3
    [0.60, 1.00],   # h7 : D3 y
    [0.65, 0.90],   # h8 : D4 x   deck edge ctrl pt 4
    [0.60, 1.00],   # h9 : D4 y
    [0.05, 0.25],   # h10: B1 x   keel plan ctrl pt 1
    [0.15, 0.40],   # h11: B2 x   keel plan ctrl pt 2
    [0.00, 0.30],   # h12: B2 y
    [0.60, 0.85],   # h13: B3 x   keel plan ctrl pt 3
    [0.00, 0.30],   # h14: B3 y
    [0.75, 0.95],   # h15: B4 x   keel plan ctrl pt 4
    [0.30, 1.00],   # h16: FK z   bow keel depth  (fraction of T)
    [0.40, 1.00],   # h17: FW y   bow half-width  (fraction of B)
    *([[0.0, 1.0]] * 12),  # h18..h29 reserved
])


def unnorm(h: np.ndarray) -> np.ndarray:
    """Map h ∈ [0,1]^29 to physical parameter values."""
    lo, hi = H_BOUNDS[:, 0], H_BOUNDS[:, 1]
    return lo + h * (hi - lo)


# ── Geometry ──────────────────────────────────────────────────────────────────

def build_control_points(h, L=1.0, B=0.375, T=0.25, F=0.30) -> dict:
    """Build all Bézier control points from normalized parameters h."""
    h  = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    hp = unnorm(h)

    # Profile / keel curve
    P1 = np.array([0.0,    0.0,        F    ])   # bow  top  (FIXED)
    P2 = np.array([0.0,    0.0,        hp[0]])   # bow  entry depth
    P3 = np.array([hp[1],  0.0,       -T    ])   # keel fwd
    P4 = np.array([hp[2],  0.0,       -T    ])   # keel aft
    P5 = np.array([L,      0.0,        F    ])   # stern top (FIXED)

    # Deck edge planform
    D1 = np.array([0.0,    B,          F])
    D2 = np.array([hp[3],  hp[4] * B,  F])
    D3 = np.array([hp[5],  hp[6] * B,  F])
    D4 = np.array([hp[7],  hp[8] * B,  F])

    # Keel width planform
    B1 = np.array([hp[9],   0.0,        -T])
    B2 = np.array([hp[10],  hp[11] * B, -T])
    B3 = np.array([hp[12],  hp[13] * B, -T])
    B4 = np.array([hp[14],  0.0,        -T])

    # Bow cross-section control points
    FK = np.array([0.0,  0.0,         -hp[15] * T])  # keel depth
    FW = np.array([0.0,  hp[16] * B,   0.0       ])  # hull half-width

    return dict(P1=P1, P2=P2, P3=P3, P4=P4, P5=P5,
                D1=D1, D2=D2, D3=D3, D4=D4,
                B1=B1, B2=B2, B3=B3, B4=B4,
                FK=FK, FW=FW)


def bezier(pts: list, n: int = 200) -> np.ndarray:
    """Evaluate a Bézier curve through pts at n equispaced parameter values."""
    pts = np.asarray(pts, dtype=float)
    deg = len(pts) - 1
    t   = np.linspace(0.0, 1.0, n)
    C   = np.zeros((n, 3))
    for k, p in enumerate(pts):
        b = comb(deg, k, exact=True) * t**k * (1.0 - t)**(deg - k)
        C += np.outer(b, p)
    return C


def hull_curves(cp: dict, n: int = 250) -> dict:
    """Return all hull curves as (n, 3) arrays of 3-D points."""
    L = cp["P5"][0]
    F = cp["P1"][2]
    return dict(
        deck_line  = np.array([[0.0, 0.0, F], [L, 0.0, F]]),
        keel_line  = bezier([cp["P1"], cp["P2"], cp["P3"],
                              cp["P4"], cp["P5"]], n),
        deck_edge  = bezier([cp["D1"], cp["D2"], cp["D3"], cp["D4"]], n),
        keel_plan  = bezier([cp["B1"], cp["B2"], cp["B3"], cp["B4"]], n),
        front_xsec = bezier([
            np.array([0.0, -cp["FW"][1],  0.0]),
            np.array([0.0, -cp["FW"][1],  cp["FK"][2]]),
            cp["FK"],
            np.array([0.0,  cp["FW"][1],  cp["FK"][2]]),
            np.array([0.0,  cp["FW"][1],  0.0]),
        ], n),
    )


# ── Physics ───────────────────────────────────────────────────────────────────

def _sort_interp(curve: np.ndarray, xs: np.ndarray, col: int) -> np.ndarray:
    """Interpolate column `col` of a curve sorted by column 0."""
    idx = np.argsort(curve[:, 0])
    return np.interp(xs, curve[idx, 0], curve[idx, col])


def compute_R_and_V(
    h,
    L: float = 1.0,
    B: float = 0.375,
    T: float = 0.25,
    F: float = 0.30,
    U: float = 2.0,
    rho: float = 1000.0,
    nu: float = 1e-6,
    g: float = 9.81,
) -> tuple[float, float]:
    """
    Truth model: build hull geometry from h, integrate to get R and V.

    Parameters
    ----------
    h   : array-like (29,)  normalized design parameters in [0,1]
    L,B,T,F : hull fixed dimensions (length, beam, draft, freeboard)
    U   : ship speed [m/s]
    rho : water density [kg/m³]
    nu  : kinematic viscosity [m²/s]
    g   : gravitational acceleration [m/s²]

    Returns
    -------
    R_total : float  total resistance [N]  (frictional + wave)
    V       : float  displaced volume  [m³]
    """
    cp = build_control_points(h, L, B, T, F)
    cv = hull_curves(cp, n=300)

    # Sample cross-sections along the hull length
    xs    = np.linspace(0.02 * L, 0.98 * L, 80)
    y_hw  = _sort_interp(cv["deck_edge"], xs, col=1)   # half-beam at each x
    z_k   = _sort_interp(cv["keel_line"], xs, col=2)   # keel z at each x
    draft = np.maximum(-z_k, 0.0)                      # local draft (positive)

    # Displaced volume: integrate elliptic cross-sectional areas
    sec_area = 0.5 * np.pi * y_hw * draft
    V = float(np.trapz(2.0 * sec_area, xs))

    # Wetted surface: integrate elliptic perimeters
    perim = np.pi * np.sqrt(0.5 * (y_hw**2 + draft**2))
    S     = float(np.trapz(2.0 * perim, xs))

    # ── Frictional resistance (ITTC-57 line) ─────────────────────────────────
    Re  = max(U * L / nu, 1e4)
    Cf  = 0.075 / (np.log10(Re) - 2.0)**2
    k   = 0.10 + 0.20 * float(np.mean(y_hw)) / B      # form factor
    R_f = 0.5 * rho * U**2 * S * Cf * (1.0 + k)

    # ── Wave resistance proxy ─────────────────────────────────────────────────
    Fn  = U / np.sqrt(g * L)                           # Froude number
    phi = (float(np.mean(y_hw)) / B) * (float(np.mean(draft)) / T)
    R_w = 0.5 * rho * U**2 * S * 0.55 * Fn**4 * phi

    return float(R_f + R_w), float(V)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    h0 = np.full(29, 0.5)
    R, V = compute_R_and_V(h0, **CONST)
    print(f"Smoke test  h=0.5 everywhere:")
    print(f"  R_total = {R:.4f} N")
    print(f"  V       = {V:.5f} m³")
    cp = build_control_points(h0, **CONST)
    print(f"  Beam B  = {CONST['B']:.3f} m   Draft T = {CONST['T']:.3f} m")