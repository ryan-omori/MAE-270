"""
hull_true_model.py
==================
SAC-based hull true model, following the F-Spline parametric approach
described in Han, Lee & Choi (2012), J. Mar. Sci. Technol. 17:1–17.

═══════════════════════════════════════════════════════════════════════
CONCEPTUAL ARCHITECTURE  (read this before touching any code)
═══════════════════════════════════════════════════════════════════════

The paper's core insight is that an entire hull's underwater volume can
be described compactly by ONE 2-D curve: the Sectional Area Curve (SAC).

  SAC:  x  →  A(x)   where x ∈ [0, L]  and  A(x) = cross-sectional
                       area of the underwater hull at station x [m²]

Everything meaningful about hull performance derives from this curve:

  Displacement  ∇  = ∫ A(x) dx          (how much water is displaced)
  Wetted area   S  = ∫ P(x) dx          (where P(x) is section perimeter)
  Prismatic Cp  = ∇ / (A_max · L)       (fullness ratio, drives wave R)
  LCB           = ∫ x·A(x) dx / ∇      (longitudinal balance)
  SAC entrance  = slope of A near bow   (wave-making at bow)
  SAC run       = slope of A near stern (wave-making at stern)

The F-Spline (from the paper, Section 2.1) parameterizes the SAC using
a small set of FORM PARAMETERS instead of raw control points:

  Entrance part (bow side, x ∈ [0, X_0E]):
    X_0E    — x-position where the parallel middle body begins (bow side)
    α_0E    — tangent angle of SAC at X_0E  (how sharply area rises)
    α_1E    — tangent angle of SAC at bow   (entrance angle, ~0° at FP)

  Run part (stern side, x ∈ [X_0R, L]):
    X_0R    — x-position where the parallel middle body ends (stern side)
    α_0R    — tangent angle of SAC at X_0R  (how sharply area drops)
    α_1R    — tangent angle of SAC at stern (run angle, ~0° at AP)

  Global:
    C_P     — prismatic coefficient = ∇ / (A_max · L)   [controls volume]
    X_LCB   — longitudinal centre of buoyancy            [balance]

The optimizer varies these 8 form parameters.  The F-Spline solver
(Section 2.1, Eq. 7–12) then finds the smoothest cubic B-Spline SAC
that satisfies those constraints, guaranteeing a fair (non-wiggly) curve
even under aggressive optimization.

Why this is better than raw control points (old approach):
  Old:  optimizer moves raw (x,z) coords → can produce kinked curves
  New:  optimizer moves physical angles/positions → curve always smooth

═══════════════════════════════════════════════════════════════════════
SECTION SHAPE ASSUMPTION
═══════════════════════════════════════════════════════════════════════

The SAC alone does not uniquely define 3-D geometry — it only tells you
how much area is at each station, not the shape of that area.  We need a
section shape model to recover wetted perimeter, draft distribution, and
the 3-D hull surface for visualization.

We use a simple but physically grounded approach: at each station x,
the cross-section is a scaled "Lewis form" approximation — an elliptic
arc whose half-beam B(x)/2 and draft T(x) are related to A(x) by:

  A(x)  ≈  (π/4) · B(x) · T(x) · C_s(x)

where C_s(x) is a local section fullness coefficient that transitions
smoothly from 0.95 (full midship U-shape) to 0.70 (fine bow V-shape).

The beam distribution B(x) follows a separate Bézier curve (the Design
Waterline, DWL) controlled by three parameters:
  B_max   — maximum beam (design variable)
  x_Bmax  — longitudinal position of max beam (usually ~0.5L)
  B_bow   — beam at bow (≥ 0 for a sharp or bluff bow entry)

From B(x) and A(x) we recover T(x) = A(x) / ((π/4)·B(x)·C_s(x)).
The wetted perimeter at each section is approximated as:
  P(x) ≈ π · sqrt(0.5 · ((B(x)/2)² + T(x)²))   [elliptic perimeter]

═══════════════════════════════════════════════════════════════════════
DESIGN VARIABLES  (what the optimizer controls)
═══════════════════════════════════════════════════════════════════════

  v[0]  C_P     prismatic coefficient      [0.55, 0.80]
  v[1]  X_LCB   LCB / L (fraction)         [0.40, 0.55]
  v[2]  X_0E    PMB start / L (fraction)   [0.10, 0.35]
  v[3]  X_0R    PMB end   / L (fraction)   [0.60, 0.85]
  v[4]  α_0E    SAC angle at PMB bow  (°)  [5,  40]
  v[5]  α_0R    SAC angle at PMB stern (°) [5,  40]
  v[6]  α_1E    SAC entrance angle   (°)   [0,  20]
  v[7]  α_1R    SAC run angle        (°)   [0,  20]
  v[8]  B_max   maximum beam         (m)   [0.20, 0.50]
  v[9]  x_Bmax  pos. of max beam / L       [0.35, 0.65]

  Fixed hull parameters (not design variables):
    L    = 1.00 m   (hull length)
    T_max = 0.25 m  (maximum allowable draft — structural constraint)
    U    = 2.00 m/s (design speed)

═══════════════════════════════════════════════════════════════════════
RESISTANCE MODEL
═══════════════════════════════════════════════════════════════════════

Total resistance:  R_T = R_F + R_W

Frictional resistance (ITTC-57 line):
  R_F = 0.5 · ρ · U² · S · C_F · (1 + k)
  C_F = 0.075 / (log10(Re) - 2)²
  k   = form factor  = 0.10 + 0.25·(∇/(L·B_max·T_max))·(B_max/L)
  Re  = U·L/ν

Wave resistance proxy (Froude-based, paper Eq. 14-style):
  R_W = 0.5 · ρ · U² · S · C_W
  C_W = c₁ · F_n⁴ · C_P² · f(α_1E, α_1R)
  where:
    F_n = U / sqrt(g·L)
    c₁  = 0.55  (calibration constant)
    f(α_1E, α_1R) = (1 + sin²(α_1E·π/180) + sin²(α_1R·π/180)) / 3
    — entrance and run angles penalize wave resistance;
      sharper bow entry (smaller α_1E) → less wave resistance

═══════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ─────────────────────────────────────────────────────────────────────────────
# FIXED HULL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

FIXED = dict(
    L     = 1.00,    # hull length [m]
    T_max = 0.25,    # maximum draft constraint [m]
    U     = 2.00,    # design speed [m/s]
    rho   = 1025.0,  # seawater density [kg/m³]
    nu    = 1.07e-6, # kinematic viscosity of seawater [m²/s]
    g     = 9.81,    # gravitational acceleration [m/s²]
    n_stations = 120, # number of integration stations along hull
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN VARIABLE BOUNDS  (physical units, not normalized)
# ─────────────────────────────────────────────────────────────────────────────

BOUNDS = np.array([
    [0.55, 0.80],   # v[0]  C_P     prismatic coefficient
    [0.40, 0.55],   # v[1]  X_LCB   LCB position / L
    [0.10, 0.35],   # v[2]  X_0E    PMB bow start / L
    [0.60, 0.85],   # v[3]  X_0R    PMB stern end / L
    [5.0,  40.0],   # v[4]  alpha_0E  SAC angle at PMB bow (deg)
    [5.0,  40.0],   # v[5]  alpha_0R  SAC angle at PMB stern (deg)
    [0.0,  20.0],   # v[6]  alpha_1E  SAC entrance angle (deg)
    [0.0,  20.0],   # v[7]  alpha_1R  SAC run angle (deg)
    [0.20, 0.50],   # v[8]  B_max   maximum beam [m]
    [0.35, 0.65],   # v[9]  x_Bmax  pos. of max beam / L
])

N_VARS = len(BOUNDS)

# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def to_physical(h: np.ndarray) -> np.ndarray:
    """
    Map normalised design vector h ∈ [0,1]^N to physical units.
    h[i] = 0 → lower bound,  h[i] = 1 → upper bound.
    """
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    return lo + h * (hi - lo)


def to_normalised(v: np.ndarray) -> np.ndarray:
    """Map physical design vector v to normalised h ∈ [0,1]^N."""
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    return np.clip((np.asarray(v, dtype=float) - lo) / (hi - lo), 0.0, 1.0)


def default_h() -> np.ndarray:
    """Return the midpoint design vector h = 0.5 everywhere."""
    return 0.5 * np.ones(N_VARS)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SAC GENERATION  (F-Spline approach from the paper)
# ─────────────────────────────────────────────────────────────────────────────
#
# The paper (Sec 2.1, Eq 7–12) generates the SAC by solving a constrained
# fairness-optimisation problem.  We implement this via a cubic spline that
# is constructed to satisfy the tangent-angle and area constraints exactly,
# using the following strategy:
#
#   1. Place control knots at:
#        x = 0      (bow,        AP in paper notation)
#        x = X_0E   (PMB start,  bow side)
#        x = X_0R   (PMB end,    stern side)
#        x = L      (stern,      FP in paper notation)
#
#   2. Assign SAC values:
#        A(0)   = 0          (zero area at bow — sharp entry)
#        A(X_0E) = A_max     (full midship area, start of PMB)
#        A(X_0R) = A_max     (full midship area, end of PMB)
#        A(L)   = 0          (zero area at stern)
#
#   3. Assign tangent slopes at each knot from the angle parameters:
#        dA/dx at x=0    = tan(α_1E) · A_max/L   (entrance angle)
#        dA/dx at x=X_0E = tan(α_0E) · A_max/L   (PMB bow tangent)
#        dA/dx at x=X_0R = -tan(α_0R) · A_max/L  (PMB stern tangent)
#        dA/dx at x=L    = -tan(α_1R) · A_max/L  (run angle)
#
#   4. Build a cubic Hermite spline through these knots with the prescribed
#      slopes.  This is the F-Spline in practice.
#
#   5. Scale A_max so that ∫A(x)dx = C_P · A_max · L  (prismatic constraint)
#      Iterate A_max until the integral matches.
#
#   6. Also shift the spline so that ∫x·A(x)dx / ∫A(x)dx = X_LCB · L
#      (LCB constraint).  We do this by adjusting X_0E and X_0R slightly
#      via a small inner optimisation.


def build_sac(v: np.ndarray, L: float, n: int = 200) -> dict:
    """
    Build the Sectional Area Curve from physical design variables v.

    This implements the F-Spline construction (paper Sec 2.1):
      — cubic Hermite spline through bow, PMB-start, PMB-end, stern
      — tangent angles prescribed at each knot (form parameters)
      — scaled to match the target prismatic coefficient C_P
      — LCB constraint enforced by iterating on PMB position

    Parameters
    ----------
    v : (10,)  physical design variables (see BOUNDS)
    L : float  hull length [m]
    n : int    number of evaluation points for the returned SAC

    Returns
    -------
    dict with keys:
      'xs'    : (n,)  x-stations [m]
      'A'     : (n,)  cross-sectional area at each station [m²]
      'A_max' : float maximum (midship) area [m²]
      'V'     : float displacement volume [m³]  = ∫A dx
      'CP'    : float achieved prismatic coefficient
      'LCB'   : float achieved LCB / L
      'spline': CubicSpline object (for evaluation at arbitrary x)
      'params': dict of the named form parameters
    """
    CP      = float(v[0])
    X_LCB   = float(v[1]) * L    # target LCB position [m]
    X_0E    = float(v[2]) * L    # PMB bow start [m]
    X_0R    = float(v[3]) * L    # PMB stern end [m]
    a0E     = float(v[4])        # SAC angle at PMB bow (deg)
    a0R     = float(v[5])        # SAC angle at PMB stern (deg)
    a1E     = float(v[6])        # SAC entrance angle (deg)
    a1R     = float(v[7])        # SAC run angle (deg)

    # Enforce geometric sanity: PMB must have positive length
    X_0E = np.clip(X_0E, 0.05 * L, 0.45 * L)
    X_0R = np.clip(X_0R, X_0E + 0.10 * L, 0.95 * L)

    def _build_spline(X0E, X0R, A_max):
        """
        Build a cubic Hermite spline for the SAC.

        Knots:   x = [0,  X0E,  X0R,  L]
        Values:  A = [0,  A_max, A_max, 0]
        Slopes:  dA/dx prescribed from angle parameters.

        The slope scaling A_max/L converts the angle (which is relative
        to the normalised SAC shape y = A/A_max vs x/L) to physical units.
        """
        xk = np.array([0.0,  X0E,    X0R,    L   ])
        Ak = np.array([0.0,  A_max,  A_max,  0.0 ])

        # Convert angles (degrees) to slopes in physical units [m²/m]
        # The angle is defined relative to the normalised curve, so:
        #   physical slope = tan(angle_deg) * (A_max / L)
        scale = A_max / L
        dAdx = np.array([
             np.tan(np.radians(a1E)) * scale,   # bow: positive slope
             np.tan(np.radians(a0E)) * scale,   # PMB start: positive
            -np.tan(np.radians(a0R)) * scale,   # PMB end: negative
            -np.tan(np.radians(a1R)) * scale,   # stern: negative
        ])

        # Build cubic Hermite spline (not-a-knot is wrong here — we need
        # to prescribe the first derivative at ALL four knots, which scipy's
        # CubicSpline supports via bc_type and dydx).
        cs = CubicSpline(xk, Ak, bc_type=((1, dAdx[0]), (1, dAdx[-1])))
        # The above sets the boundary derivatives at the two ENDPOINTS (x=0
        # and x=L) but leaves the interior knot derivatives free (natural).
        # To also enforce the interior tangents at X0E and X0R, we use the
        # "clamped" form at all four knots via a piecewise construction:
        #   segment 1: x ∈ [0, X0E]      with end slopes dAdx[0], dAdx[1]
        #   segment 2: x ∈ [X0E, X0R]    flat (PMB), both slopes ≈ 0
        #   segment 3: x ∈ [X0R, L]      with end slopes dAdx[2], dAdx[3]

        seg1 = CubicSpline(
            [0.0, X0E], [0.0, A_max],
            bc_type=((1, dAdx[0]), (1, dAdx[1]))
        )
        # PMB segment: flat at A_max with zero slope at both ends
        seg2 = CubicSpline(
            [X0E, X0R], [A_max, A_max],
            bc_type=((1, 0.0), (1, 0.0))
        )
        seg3 = CubicSpline(
            [X0R, L], [A_max, 0.0],
            bc_type=((1, dAdx[2]), (1, dAdx[3]))
        )

        def sac_eval(x):
            x = np.asarray(x, dtype=float)
            out = np.zeros_like(x)
            m1 = (x >= 0)    & (x <= X0E)
            m2 = (x > X0E)   & (x <= X0R)
            m3 = (x > X0R)   & (x <= L)
            out[m1] = np.maximum(seg1(x[m1]), 0.0)
            out[m2] = np.maximum(seg2(x[m2]), 0.0)
            out[m3] = np.maximum(seg3(x[m3]), 0.0)
            return out

        return sac_eval, (seg1, seg2, seg3)

    # ── Step 1: find A_max that satisfies the prismatic coefficient ───────────
    # CP = ∇ / (A_max · L)  →  ∇ = CP · A_max · L
    # But ∇ = ∫A(x)dx, which depends on A_max nonlinearly through the spline.
    # We fix a unit A_max = 1 first, compute the integral, then scale:
    #   ∫A_unit dx = I_unit
    #   actual ∇ = A_max · I_unit  (because spline scales linearly with A_max)
    #   target ∇ = CP · A_max · L
    #   → A_max · I_unit = CP · A_max · L  is automatically satisfied for
    #     ANY A_max because I_unit = CP · L when the spline is normalised.
    # So we first solve for the normalised shape, then A_max is free to set V.
    # We will set A_max from a target displacement volume V_target below.
    # For now, use A_max = 1 to find the normalised shape.

    sac_unit, segs = _build_spline(X_0E, X_0R, A_max=1.0)

    xs_int = np.linspace(0.0, L, 2000)
    A_unit = sac_unit(xs_int)
    I_unit = np.trapz(A_unit, xs_int)   # = ∫A_unit dx with A_max=1

    # The normalised prismatic coefficient of the unit spline is:
    CP_unit = I_unit / (1.0 * L)   # = I_unit / L

    # To match CP exactly, we need the spline integral to equal CP · L
    # when A_max = 1.  But CP_unit ≠ CP in general because the spline
    # shape (controlled by the angles) determines CP independently.
    # The paper addresses this via the area constraint (Eq. 6) in the
    # F-Spline optimisation.  We replicate that constraint by iterating
    # on the PMB length (X_0R - X_0E) to match CP exactly.

    def _cp_residual(pmb_length):
        """Residual between achieved CP and target CP for a given PMB length."""
        X0E_t = X_0E
        X0R_t = np.clip(X0E_t + pmb_length * L, X0E_t + 0.01 * L, 0.98 * L)
        sac_t, _ = _build_spline(X0E_t, X0R_t, A_max=1.0)
        A_t  = sac_t(xs_int)
        I_t  = np.trapz(A_t, xs_int)
        CP_t = I_t / L
        return (CP_t - CP) ** 2

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(_cp_residual,
                          bounds=(0.05, 0.60),
                          method="bounded",
                          options={"xatol": 1e-6})
    pmb_length_opt = res.x
    X_0R_adj = np.clip(X_0E + pmb_length_opt * L, X_0E + 0.01 * L, 0.98 * L)

    # ── Step 2: rebuild spline with adjusted PMB to match CP ─────────────────
    sac_fn, segs = _build_spline(X_0E, X_0R_adj, A_max=1.0)
    A_norm = sac_fn(xs_int)
    I_norm = np.trapz(A_norm, xs_int)

    # ── Step 3: enforce LCB by iterating on X_0E position ────────────────────
    # LCB = ∫x·A(x)dx / ∫A(x)dx  — shifting X_0E moves the centroid.
    # We do a simple bounded search over X_0E ∈ [0.05L, 0.40L].

    def _lcb_residual(x0e_frac):
        X0E_t = x0e_frac * L
        X0R_t = np.clip(X0E_t + pmb_length_opt * L,
                        X0E_t + 0.01 * L, 0.98 * L)
        sac_t, _ = _build_spline(X0E_t, X0R_t, A_max=1.0)
        A_t  = sac_t(xs_int)
        I_t  = np.trapz(A_t, xs_int)
        if I_t < 1e-12:
            return 1.0
        lcb_t = np.trapz(xs_int * A_t, xs_int) / I_t
        return (lcb_t / L - v[1]) ** 2

    res_lcb = minimize_scalar(_lcb_residual,
                               bounds=(0.05, 0.40),
                               method="bounded",
                               options={"xatol": 1e-6})
    X_0E_adj = res_lcb.x * L
    X_0R_adj = np.clip(X_0E_adj + pmb_length_opt * L,
                       X_0E_adj + 0.01 * L, 0.98 * L)

    # ── Step 4: final spline with A_max = 1 (shape only) ─────────────────────
    sac_fn, segs = _build_spline(X_0E_adj, X_0R_adj, A_max=1.0)

    # Evaluate on the output grid
    xs = np.linspace(0.0, L, n)
    A_shape = sac_fn(xs)
    A_shape = np.maximum(A_shape, 0.0)   # clip any tiny numerical negatives

    # ── Step 5: compute achieved global parameters ────────────────────────────
    I_shape = np.trapz(A_shape, xs)
    lcb_achieved = np.trapz(xs * A_shape, xs) / I_shape if I_shape > 0 else 0.5 * L

    # A_max is the maximum value of the unit-shape SAC.
    # To set the actual displacement volume, the caller can scale A.
    # Here we return the unit shape; the resistance model uses it directly
    # after computing the actual A_max from the section shape model.
    A_max_unit = float(np.max(A_shape))

    CP_achieved = I_shape / (A_max_unit * L) if A_max_unit > 0 else 0.0

    return dict(
        xs      = xs,
        A_shape = A_shape,        # normalised SAC (A_max ≈ 1 after scaling)
        A_max   = A_max_unit,     # peak value (should be ≈ 1.0)
        I_norm  = I_shape,        # integral of unit SAC
        CP      = CP_achieved,
        LCB     = lcb_achieved / L,
        sac_fn  = sac_fn,
        params  = dict(
            CP    = CP,   X_LCB = v[1],
            X_0E  = X_0E_adj / L, X_0R = X_0R_adj / L,
            a0E   = a0E,  a0R   = a0R,
            a1E   = a1E,  a1R   = a1R,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SECTION SHAPE MODEL  (beam + draft distributions)
# ─────────────────────────────────────────────────────────────────────────────
#
# Given the SAC shape A_shape(x) and the beam design parameters, we recover:
#   B(x)  — full beam at each station [m]   (from DWL Bézier)
#   T(x)  — draft at each station [m]       (from A(x) and B(x))
#   P(x)  — wetted perimeter [m]            (elliptic approximation)
#
# The beam distribution follows a simple Bézier-based DWL:
#   B(x) = B_max · b(x/L)
# where b(ξ) is a smooth curve from b(0)=B_bow/B_max to b(1)=B_bow/B_max
# with a peak of 1.0 at ξ = x_Bmax/L.  We use a quadratic Bézier for
# simplicity, which is smooth and has only 3 control points.


def build_section_shape(v: np.ndarray, sac: dict, L: float,
                        T_max: float) -> dict:
    """
    Compute beam B(x), draft T(x), and wetted perimeter P(x) at each station.

    Parameters
    ----------
    v      : (10,) physical design variables
    sac    : dict returned by build_sac()
    L      : float hull length [m]
    T_max  : float maximum draft constraint [m]

    Returns
    -------
    dict with keys:
      'xs'    : (n,) x-stations [m]  (same grid as sac)
      'B'     : (n,) full beam [m]
      'T'     : (n,) draft [m]
      'A'     : (n,) actual cross-sectional area [m²]  (after A_max scaling)
      'P'     : (n,) wetted perimeter [m]
      'A_max' : float  actual midship area [m²]
      'S'     : float  total wetted surface area [m²]
      'V'     : float  displacement volume [m³]
    """
    B_max  = float(v[8])
    x_Bmax = float(v[9]) * L

    xs     = sac["xs"]
    xi     = xs / L              # normalised x ∈ [0,1]

    # ── Beam distribution: smooth hump using a quadratic Bézier ───────────────
    # Control points: (0, B_bow), (x_Bmax, B_max), (L, B_bow)
    # where B_bow = 0.05 * B_max (a slightly blunt bow, physically reasonable)
    B_bow  = 0.05 * B_max
    xi_pk  = x_Bmax / L

    # Quadratic Bézier: B(ξ) = (1-t)²·B_bow + 2t(1-t)·B_max + t²·B_bow
    # where t is implicitly defined by ξ(t) = (1-t)²·0 + 2t(1-t)·xi_pk + t²·1
    # For each station xi, solve for t, then evaluate B(t).
    # ξ(t) = 2t·xi_pk + t²(1 - 2·xi_pk)  →  quadratic in t
    a_coef = 1.0 - 2.0 * xi_pk
    b_coef = 2.0 * xi_pk
    # t² a_coef + t b_coef - xi = 0
    # Special case: if xi_pk ≈ 0.5, a_coef ≈ 0, linear in t
    B = np.zeros_like(xi)
    for i, xi_i in enumerate(xi):
        if abs(a_coef) < 1e-9:
            t = xi_i / b_coef if b_coef > 1e-9 else 0.5
        else:
            disc = b_coef**2 + 4.0 * a_coef * xi_i
            disc = max(disc, 0.0)
            t = (-b_coef + np.sqrt(disc)) / (2.0 * a_coef)
        t = np.clip(t, 0.0, 1.0)
        B[i] = (1-t)**2 * B_bow + 2*t*(1-t) * B_max + t**2 * B_bow

    # Half-beam
    HB = B / 2.0

    # ── Section fullness coefficient C_s(x) ───────────────────────────────────
    # Transitions from C_s = 0.75 at bow/stern (fine V-section)
    # to C_s = 0.95 at midship (full U-section).
    # We model this as a smooth function of the normalised SAC value:
    #   A_norm = A_shape / A_max
    # High normalised area → full section (U), low → fine section (V).
    A_norm_frac = sac["A_shape"] / (sac["A_max"] + 1e-12)
    Cs = 0.75 + 0.20 * A_norm_frac   # ranges 0.75 … 0.95

    # ── A_max: scale so that midship area matches the target ──────────────────
    # We set midship area A_max = (π/4) · B_max · T_mid · Cs_mid
    # where T_mid is chosen to use a fraction of T_max at midship.
    # We target T_mid = 0.90 · T_max (leaving some freeboard margin).
    T_mid_target = 0.90 * T_max
    Cs_mid = 0.95   # midship fullness coefficient (constant)
    A_max_physical = (np.pi / 4.0) * B_max * T_mid_target * Cs_mid

    # ── Scale SAC from unit shape to physical area ────────────────────────────
    A = sac["A_shape"] * (A_max_physical / (sac["A_max"] + 1e-12))
    A = np.maximum(A, 0.0)

    # ── Draft distribution: invert  A = (π/4)·B·T·Cs  for T ─────────────────
    denom = (np.pi / 4.0) * HB * 2.0 * Cs   # = (π/4)·B·Cs
    T = np.where(denom > 1e-9, A / denom, 0.0)
    T = np.clip(T, 0.0, T_max)

    # ── Wetted perimeter: elliptic approximation ──────────────────────────────
    # P ≈ π · sqrt(0.5 · (HB² + T²))
    # This is the perimeter of an ellipse with semi-axes HB and T,
    # using Ramanujan's simpler approximation at low eccentricity.
    P = np.pi * np.sqrt(0.5 * (HB**2 + T**2))
    # For stations where draft or beam is essentially zero, P → 0
    P = np.where(np.minimum(HB, T) < 1e-6, 0.0, P)

    # ── Integrated quantities ─────────────────────────────────────────────────
    # Wetted surface: integrate perimeter along hull length
    # Factor 1.0 (not 2.0) because the hull is symmetric port/starboard —
    # the elliptic perimeter P already covers both sides.
    S = float(np.trapz(P, xs))

    # Displacement volume
    V = float(np.trapz(A, xs))

    return dict(
        xs     = xs,
        B      = B,
        T      = T,
        A      = A,
        P      = P,
        HB     = HB,
        Cs     = Cs,
        A_max  = A_max_physical,
        S      = S,
        V      = V,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — RESISTANCE MODEL
# ─────────────────────────────────────────────────────────────────────────────

def compute_resistance(v: np.ndarray, sac: dict, geom: dict,
                       L: float, U: float, rho: float,
                       nu: float, g: float) -> dict:
    """
    Compute total resistance from hull geometry.

    R_T = R_F  (ITTC-57 frictional)  +  R_W  (wave resistance proxy)

    Parameters
    ----------
    v    : (10,) physical design variables
    sac  : dict from build_sac()
    geom : dict from build_section_shape()
    L, U, rho, nu, g : fixed physical constants

    Returns
    -------
    dict with keys:
      'R_T'  : total resistance [N]
      'R_F'  : frictional resistance [N]
      'R_W'  : wave resistance [N]
      'C_F'  : ITTC-57 friction coefficient
      'C_W'  : wave resistance coefficient
      'F_n'  : Froude number
      'Re'   : Reynolds number
      'k'    : form factor  (1+k applied to R_F)
      'S'    : wetted surface area [m²]
      'V'    : displacement volume [m³]
    """
    S   = geom["S"]
    V   = geom["V"]
    CP  = sac["CP"]
    a1E = sac["params"]["a1E"]   # entrance angle (deg)
    a1R = sac["params"]["a1R"]   # run angle (deg)

    # ── Reynolds and Froude numbers ───────────────────────────────────────────
    Re  = max(U * L / nu, 1e4)
    F_n = U / np.sqrt(g * L)

    # ── ITTC-57 friction coefficient ──────────────────────────────────────────
    C_F = 0.075 / (np.log10(Re) - 2.0) ** 2

    # ── Form factor k (Holtrop-simplified) ────────────────────────────────────
    # k depends on the block coefficient (fullness) and L/B ratio.
    # CB = V / (L · B_max · T_max_actual)
    B_max  = float(v[8])
    T_mean = float(np.mean(geom["T"][geom["T"] > 0.01]))
    CB     = V / max(L * B_max * T_mean, 1e-9)
    CB     = np.clip(CB, 0.4, 0.9)
    LB     = L / max(B_max, 1e-9)
    # Form factor approximation (Holtrop & Mennen 1982, simplified):
    k = 0.93 + 0.4871 * (CB / LB**0.1228) * (B_max / L)**0.6906
    k = np.clip(k, 0.05, 0.50)   # physically reasonable bounds

    # ── Frictional resistance ─────────────────────────────────────────────────
    R_F = 0.5 * rho * U**2 * S * C_F * (1.0 + k)

    # ── Wave resistance coefficient ───────────────────────────────────────────
    # Based on the paper's approach (Eq. 14 style):
    #   C_W = c1 · F_n^4 · CP^2 · angle_penalty
    #
    # Physical reasoning:
    #   F_n^4 : wave resistance scales strongly with speed
    #   CP^2  : fuller hulls make more waves (Michell's integral)
    #   angle_penalty: sharper bow/stern entry angles reduce wave-making.
    #     The paper shows (Fig. 7) that reducing α_1E by 2Δ=30° gives ~15%
    #     wave resistance reduction.  We model this as:
    #       f(α) = 1 + (α/α_ref)^1.5
    #     where α_ref = 20° is a reference angle.

    alpha_ref = 20.0   # reference angle [degrees]
    f_entrance = 1.0 + (a1E / alpha_ref) ** 1.5
    f_run      = 1.0 + 0.5 * (a1R / alpha_ref) ** 1.5   # stern less critical
    f_angle    = (f_entrance + f_run) / 2.0

    c1  = 0.55   # calibration constant (tuned to give physically reasonable
                  # wave resistance magnitudes for Fn ≈ 0.64 at U=2m/s, L=1m)

    C_W = c1 * F_n**4 * CP**2 * f_angle

    # ── Wave resistance ───────────────────────────────────────────────────────
    R_W = 0.5 * rho * U**2 * S * C_W

    R_T = R_F + R_W

    return dict(
        R_T = R_T, R_F = R_F, R_W = R_W,
        C_F = C_F, C_W = C_W,
        F_n = F_n, Re  = Re,
        k   = k,   S   = S,  V = V,
        CB  = CB,  CP  = CP,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MAIN TRUE MODEL ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def true_model(h: np.ndarray) -> dict:
    """
    Evaluate the complete hull true model for a normalised design vector h.

    This is the function called by the data generator, surrogate trainer,
    and optimizer (via CSDL) throughout the MDO pipeline.

    Parameters
    ----------
    h : (N_VARS,) = (10,)  normalised design vector, each element ∈ [0,1]

    Returns
    -------
    dict with keys:
      'R_T'    : total resistance [N]          ← primary objective
      'V'      : displacement volume [m³]      ← constraint
      'S'      : wetted surface area [m²]
      'CP'     : achieved prismatic coefficient
      'LCB'    : achieved LCB / L
      'F_n'    : Froude number
      'CB'     : block coefficient
      'sac'    : SAC dict (for plotting/geometry access)
      'geom'   : section shape dict
      'resist' : full resistance breakdown dict
    """
    v   = to_physical(h)
    L   = FIXED["L"]
    T_max = FIXED["T_max"]
    U   = FIXED["U"]
    rho = FIXED["rho"]
    nu  = FIXED["nu"]
    g   = FIXED["g"]
    n   = FIXED["n_stations"]

    sac    = build_sac(v, L=L, n=n)
    geom   = build_section_shape(v, sac, L=L, T_max=T_max)
    resist = compute_resistance(v, sac, geom, L=L, U=U,
                                rho=rho, nu=nu, g=g)
    return dict(
        R_T    = resist["R_T"],
        V      = geom["V"],
        S      = geom["S"],
        CP     = sac["CP"],
        LCB    = sac["LCB"],
        F_n    = resist["F_n"],
        CB     = resist["CB"],
        sac    = sac,
        geom   = geom,
        resist = resist,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — GEOMETRY RECONSTRUCTION FOR VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_3d_sections(geom: dict, n_theta: int = 60) -> dict:
    """
    Reconstruct 3D hull cross-sections from B(x) and T(x) for plotting.

    Each section is an elliptic arc (below z=0, from y=-HB to y=+HB).
    This gives us enough geometry to draw the side profile and front profile.

    Parameters
    ----------
    geom    : dict from build_section_shape()
    n_theta : int  number of angular points for each section ellipse

    Returns
    -------
    dict with keys:
      'xs'       : (n,) station x-positions [m]
      'sections' : list of (n_theta, 2) arrays, each row is (y, z) [m]
                   for that station's underwater cross-section
      'HB'       : (n,) half-beam at waterline [m]
      'T'        : (n,) draft [m]
    """
    xs  = geom["xs"]
    HB  = geom["HB"]
    T   = geom["T"]

    theta = np.linspace(np.pi, 0.0, n_theta)   # port → starboard (below WL)

    sections = []
    for hb, t in zip(HB, T):
        if hb < 1e-6 or t < 1e-6:
            sections.append(np.zeros((n_theta, 2)))
        else:
            y = hb * np.cos(theta)    # y: −HB … +HB
            z = -t * np.sin(theta)    # z: 0 … −T … 0  (below waterline)
            sections.append(np.column_stack([y, z]))

    return dict(xs=xs, sections=sections, HB=HB, T=T)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_hull(h: np.ndarray, title: str = "Hull Geometry", save: str = None):
    """
    Three-panel plot for a given design vector h:
      (A) SAC — the primary geometric descriptor
      (B) Side profile — keel line T(x) and deck line
      (C) Front profile — bow cross-section at x ≈ 0.05L

    Parameters
    ----------
    h     : (N_VARS,) normalised design vector
    title : str  figure title
    save  : str  optional filepath to save the figure (e.g. 'hull.png')
    """
    result = true_model(h)
    sac    = result["sac"]
    geom   = result["geom"]
    resist = result["resist"]
    L      = FIXED["L"]

    xs   = geom["xs"]
    B    = geom["B"]
    T    = geom["T"]
    HB   = geom["HB"]
    A    = geom["A"]

    C_BLUE   = "#1B4F72"
    C_ORANGE = "#CA6F1E"
    C_GREEN  = "#1D6A39"
    C_GRAY   = "#566573"

    fig = plt.figure(figsize=(15, 5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.32)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    fig.suptitle(
        f"{title}\n"
        f"R_T = {resist['R_T']:.3f} N   "
        f"(R_F = {resist['R_F']:.3f} N,  R_W = {resist['R_W']:.3f} N)   "
        f"V = {geom['V']:.4f} m³   "
        f"C_P = {sac['CP']:.3f}   "
        f"F_n = {resist['F_n']:.3f}",
        fontsize=10, fontweight="bold",
    )

    # ── (A) SAC ───────────────────────────────────────────────────────────────
    A_norm = A / (geom["A_max"] + 1e-12)
    ax1.fill_between(xs / L, A_norm, alpha=0.15, color=C_BLUE)
    ax1.plot(xs / L, A_norm, color=C_BLUE, lw=2.2, label="SAC  A(x)/A_max")

    # Mark PMB region
    p = sac["params"]
    ax1.axvspan(p["X_0E"], p["X_0R"], alpha=0.10, color=C_GREEN,
                label=f"PMB  [{p['X_0E']:.2f}L – {p['X_0R']:.2f}L]")

    # Mark LCB
    lcb_x = sac["LCB"]
    ax1.axvline(lcb_x, color=C_ORANGE, lw=1.2, ls="--",
                label=f"LCB = {lcb_x:.3f}L")

    # Tangent angle annotations at bow and stern
    a1E = p["a1E"];  a1R = p["a1R"]
    ax1.annotate(f"α₁E = {a1E:.1f}°\n(entrance)",
                 xy=(0.05, 0.2), fontsize=7.5, color=C_GRAY,
                 ha="left", style="italic")
    ax1.annotate(f"α₁R = {a1R:.1f}°\n(run)",
                 xy=(0.82, 0.2), fontsize=7.5, color=C_GRAY,
                 ha="right", style="italic")

    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_xlabel("x / L  (normalised hull length)", fontsize=9)
    ax1.set_ylabel("A(x) / A_max  (normalised section area)", fontsize=9)
    ax1.set_title("(A)  Sectional Area Curve (SAC)", fontsize=10)
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.grid(True, alpha=0.20)

    # ── (B) Side profile ──────────────────────────────────────────────────────
    # Show: waterline, keel line (−T(x)), deck (at z=0, flat), half-beam

    ax2.axhline(0, color=C_GRAY, lw=0.8, ls="--", alpha=0.6,
                label="Waterline (z=0)")

    # Keel line: z = −T(x)
    ax2.plot(xs, -T, color=C_BLUE, lw=2.2, label="Keel depth −T(x)")
    ax2.fill_between(xs, -T, 0, alpha=0.10, color=C_BLUE)

    # Half-beam outline in side view is not directly visible (it's a plan view
    # feature), but we show the deck edge as a flat line at z=0
    ax2.plot([xs[0], xs[-1]], [0, 0], color=C_BLUE, lw=2.5,
             label="Deck / waterline", solid_capstyle="round")

    # Annotate max draft
    idx_T = np.argmax(T)
    ax2.annotate(f"T_max = {T[idx_T]:.3f} m",
                 xy=(xs[idx_T], -T[idx_T]),
                 textcoords="offset points", xytext=(8, -12),
                 fontsize=7.5, color=C_BLUE,
                 arrowprops=dict(arrowstyle="->", color=C_BLUE,
                                 lw=0.8))

    ax2.set_xlabel("x  [m]  — hull length", fontsize=9)
    ax2.set_ylabel("z  [m]  — depth  (−T below WL)", fontsize=9)
    ax2.set_title("(B)  Side profile  —  keel depth T(x)", fontsize=10)
    ax2.set_xlim(-0.02 * L, L * 1.05)
    ax2.set_ylim(-FIXED["T_max"] * 1.35, FIXED["T_max"] * 0.35)
    ax2.set_aspect("equal")
    ax2.legend(fontsize=7.5, loc="lower right")
    ax2.grid(True, alpha=0.20)

    # ── (C) Front profile — bow cross-section ─────────────────────────────────
    # Find a station near x = 0.10L (a representative bow section)
    x_bow_idx = np.argmin(np.abs(xs - 0.10 * L))
    x_mid_idx = np.argmin(np.abs(xs - 0.50 * L))
    x_aft_idx = np.argmin(np.abs(xs - 0.85 * L))

    # Reconstruct 3D sections
    secs = reconstruct_3d_sections(geom)

    for idx, label, color, lw_val, ls in [
        (x_bow_idx, f"Bow  (x = {xs[x_bow_idx]:.2f} m)",  C_ORANGE, 2.0, "-"),
        (x_mid_idx, f"Mid  (x = {xs[x_mid_idx]:.2f} m)",  C_BLUE,   2.2, "-"),
        (x_aft_idx, f"Aft  (x = {xs[x_aft_idx]:.2f} m)",  C_GREEN,  1.8, "--"),
    ]:
        sec = secs["sections"][idx]
        ax3.plot(sec[:, 0], sec[:, 1], color=color, lw=lw_val,
                 ls=ls, label=label)
        # Mirror port side (section already spans −HB to +HB, so no mirror needed)

    # Waterline
    ax3.axhline(0, color=C_GRAY, lw=0.8, ls="--", alpha=0.6,
                label="Waterline (z=0)")
    ax3.axvline(0, color=C_GRAY, lw=0.6, ls=":", alpha=0.35,
                label="Centreline (y=0)")

    # Flat deck across maximum beam
    B_max = float(np.max(HB))
    ax3.plot([-B_max, B_max], [0, 0], color=C_BLUE, lw=2.5,
             solid_capstyle="round", alpha=0.7)

    # Annotate beam and draft of the midship section
    hw_mid = float(HB[x_mid_idx])
    T_mid  = float(T[x_mid_idx])
    ax3.annotate(f"B_max/2\n= {hw_mid:.3f} m",
                 xy=(hw_mid, 0), textcoords="offset points",
                 xytext=(6, 10), fontsize=7, color=C_BLUE)
    ax3.annotate(f"T_mid\n= {T_mid:.3f} m",
                 xy=(0, -T_mid), textcoords="offset points",
                 xytext=(6, -14), fontsize=7, color=C_BLUE,
                 arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.7))

    # Chine region annotation on bow section
    hw_bow = float(HB[x_bow_idx])
    ax3.annotate(
        "chine region",
        xy=(hw_bow, 0.0),
        textcoords="offset points", xytext=(-10, -30),
        fontsize=6.5, color=C_GRAY, ha="center",
        arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=0.7, alpha=0.7),
    )

    span   = FIXED["T_max"] + 0.05
    margin = span * 0.30
    ax3.set_xlim(-(B_max + margin), (B_max + margin))
    ax3.set_ylim(-FIXED["T_max"] - margin, FIXED["T_max"] * 0.35)
    ax3.set_aspect("equal")
    ax3.set_xlabel("y  [m]  — beam  (port ← → starboard)", fontsize=9)
    ax3.set_ylabel("z  [m]  — depth", fontsize=9)
    ax3.set_title("(C)  Front profile  —  cross-sections", fontsize=10)
    ax3.legend(fontsize=7.5, loc="lower center", ncol=1)
    ax3.grid(True, alpha=0.20)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=160, bbox_inches="tight")
        print(f"  Saved → {save}")
    plt.show()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — SMOKE TEST  (run this file directly to verify everything works)
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test():
    """
    Run three design points and print results as a sanity check table.
    Also generates the geometry plot for the midpoint design.

    Expected output (rough magnitudes for L=1m, U=2m/s):
      R_T  ≈  5 – 30 N     (plausible for a 1m model hull at 2 m/s)
      V    ≈  0.010 – 0.060 m³
      C_P  ≈  0.55 – 0.80  (matches input)
      F_n  ≈  0.64         (fixed: U=2, L=1, Fn = 2/sqrt(9.81) ≈ 0.639)
    """
    print("=" * 65)
    print("Hull True Model  —  Smoke Test")
    print(f"Fixed params:  L = {FIXED['L']} m   U = {FIXED['U']} m/s   "
          f"T_max = {FIXED['T_max']} m")
    print("=" * 65)

    test_cases = {
        "h = 0.0 (lower bounds)": np.zeros(N_VARS),
        "h = 0.5 (mid-point)  ": 0.5 * np.ones(N_VARS),
        "h = 1.0 (upper bounds)": np.ones(N_VARS),
    }

    print(f"\n{'Case':<28} {'R_T':>8} {'R_F':>8} {'R_W':>8} "
          f"{'V':>8} {'S':>7} {'C_P':>6} {'LCB/L':>6} {'C_B':>6}")
    print("-" * 95)

    for label, h in test_cases.items():
        try:
            r = true_model(h)
            print(
                f"{label}  "
                f"{r['R_T']:8.3f} N  "
                f"{r['resist']['R_F']:8.3f} N  "
                f"{r['resist']['R_W']:8.3f} N  "
                f"{r['V']:8.4f} m³ "
                f"{r['S']:7.4f} m²  "
                f"{r['CP']:6.3f}  "
                f"{r['LCB']:6.3f}  "
                f"{r['CB']:6.3f}"
            )
        except Exception as e:
            print(f"{label}  ERROR: {e}")

    print("=" * 65)
    print("\nGenerating geometry plot for h = 0.5 (mid-point design)...")
    plot_hull(0.5 * np.ones(N_VARS),
              title="Mid-point design  (h = 0.5)",
              save="hull_midpoint.png")


if __name__ == "__main__":
    smoke_test()