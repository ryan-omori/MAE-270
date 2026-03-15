import numpy as np
from dataclasses import dataclass


# ============================================================
# GLOBAL CONSTANTS
# ============================================================
RHO_WATER = 1025.0       # kg/m^3
NU_WATER = 1.19e-6       # m^2/s
G = 9.81                 # m/s^2


@dataclass
class HullSettings:
    n_beam_cp: int = 4          # control points for half-beam distribution
    n_draft_cp: int = 4         # control points for draft distribution
    L: float = 30.0             # hull length [m]
    V_ship: float = 6.0         # ship speed [m/s]
    target_volume: float = 280.0  # displaced volume target [m^3]
    section_shape_coeff: float = 0.82  # relates B(x),T(x) to area
    min_half_beam: float = 0.4
    max_half_beam: float = 3.8
    min_draft: float = 0.6
    max_draft: float = 3.0
    nx: int = 200


# ============================================================
# BASIC BERNSTEIN / BEZIER UTILITIES
# ============================================================
def comb(n, k):
    from math import comb as _comb
    return _comb(n, k)


def bernstein_basis(n, x):
    """
    Returns matrix B of shape (len(x), n+1)
    """
    x = np.asarray(x)
    B = np.zeros((len(x), n + 1))
    for k in range(n + 1):
        B[:, k] = comb(n, k) * (x ** k) * ((1 - x) ** (n - k))
    return B


# ============================================================
# HULL PARAMETERIZATION
# design vector z = [beam cps..., draft cps...]
# ============================================================
def unpack_design(z, settings: HullSettings):
    nb = settings.n_beam_cp
    nd = settings.n_draft_cp
    z = np.asarray(z).ravel()

    assert len(z) == nb + nd, (
        f"Expected {nb + nd} design vars, got {len(z)}"
    )

    beam_cp = z[:nb]
    draft_cp = z[nb:nb + nd]
    return beam_cp, draft_cp


def hull_sections(z, settings: HullSettings):
    """
    Creates smooth half-beam B/2(x) and draft T(x) distributions.
    """
    beam_cp, draft_cp = unpack_design(z, settings)

    x_hat = np.linspace(0.0, 1.0, settings.nx)
    Bm = bernstein_basis(settings.n_beam_cp - 1, x_hat)
    Dm = bernstein_basis(settings.n_draft_cp - 1, x_hat)

    half_beam = Bm @ beam_cp
    draft = Dm @ draft_cp

    # Enforce bow/stern taper and smoothness with multiplicative envelope
    # so the end sections do not stay unrealistically full.
    taper = np.sin(np.pi * x_hat) ** 0.85
    taper = np.maximum(taper, 0.08)

    half_beam = half_beam * taper
    draft = draft * (0.70 + 0.30 * taper)

    return x_hat, half_beam, draft


def sectional_area_curve(z, settings: HullSettings):
    """
    Approximate immersed sectional area:
        A(x) = C_sec * 2*half_beam(x) * draft(x)
    """
    x_hat, half_beam, draft = hull_sections(z, settings)
    area = settings.section_shape_coeff * 2.0 * half_beam * draft
    x = x_hat * settings.L
    return x, area, half_beam, draft


def hull_volume(z, settings: HullSettings):
    x, area, _, _ = sectional_area_curve(z, settings)
    volume = np.trapz(area, x)
    return volume


def max_beam_and_draft(z, settings: HullSettings):
    _, _, half_beam, draft = sectional_area_curve(z, settings)
    B = 2.0 * np.max(half_beam)
    T = np.max(draft)
    return B, T


def wetted_surface_area(z, settings: HullSettings):
    """
    Simple 3D-ish approximation from longitudinal strip integration.
    """
    x, area, half_beam, draft = sectional_area_curve(z, settings)

    dBdx = np.gradient(half_beam, x)
    dTdx = np.gradient(draft, x)

    # crude but smooth approximation for wetted area of both sides + bottom
    side_strip = 2.0 * draft * np.sqrt(1.0 + dBdx**2)
    bottom_strip = 2.0 * half_beam * np.sqrt(1.0 + dTdx**2)

    Sw = 2.0 * np.trapz(side_strip + bottom_strip, x)
    return Sw


def principal_coefficients(z, settings: HullSettings):
    volume = hull_volume(z, settings)
    B, T = max_beam_and_draft(z, settings)
    L = settings.L
    Cb = volume / (L * B * T + 1e-12)

    x, area, _, _ = sectional_area_curve(z, settings)
    Am = np.max(area)
    Cm = Am / (B * T + 1e-12)
    Cp = volume / (Am * L + 1e-12)

    return {
        "L": L,
        "B": B,
        "T": T,
        "volume": volume,
        "Cb": Cb,
        "Cm": Cm,
        "Cp": Cp,
    }


# ============================================================
# TRUTH RESISTANCE MODEL
# Replace this block with the exact equations from your pictures.
# ============================================================
def truth_total_resistance(z, settings: HullSettings):
    """
    Total resistance = frictional + residual(wave) + mild shape penalties.
    This is smooth and works well for training/optimization.

    Returns:
        Rt, aux_dict
    """
    coeffs = principal_coefficients(z, settings)
    L = coeffs["L"]
    B = coeffs["B"]
    T = coeffs["T"]
    volume = coeffs["volume"]
    Cb = coeffs["Cb"]
    Cp = coeffs["Cp"]

    V = settings.V_ship
    Sw = wetted_surface_area(z, settings)

    # Reynolds number / friction
    Re = V * L / NU_WATER
    Cf = 0.075 / ((np.log10(Re) - 2.0) ** 2 + 1e-12)
    Rf = 0.5 * RHO_WATER * V**2 * Sw * Cf

    # Froude number
    Fn = V / np.sqrt(G * L + 1e-12)

    # Residual / wave-making style term
    slenderness = L / (B + 1e-12)
    fullness_penalty = (Cb - 0.58) ** 2 + 0.8 * (Cp - 0.62) ** 2

    Cr = (
        0.004
        + 0.08 * Fn**3
        + 0.02 / (slenderness + 1e-12)
        + 0.10 * fullness_penalty
    )

    Sref = volume ** (2.0 / 3.0)
    Rr = 0.5 * RHO_WATER * V**2 * Sref * Cr

    # Soft geometric regularizers to discourage weird hulls
    dV = max(0.0, abs(volume - settings.target_volume) / settings.target_volume - 0.10)
    vol_pen = 5e4 * dV**2

    # unrealistic beam/draft ratio penalty
    ratio = B / (T + 1e-12)
    ratio_pen = 2e4 * max(0.0, 2.2 - ratio)**2 + 2e4 * max(0.0, ratio - 5.5)**2

    Rt = Rf + Rr + vol_pen + ratio_pen

    aux = {
        "Rf": Rf,
        "Rr": Rr,
        "Sw": Sw,
        "Fn": Fn,
        "Re": Re,
        "volume": volume,
        "Cb": Cb,
        "Cp": Cp,
        "B": B,
        "T": T,
        "Rt": Rt,
    }
    return Rt, aux


# ============================================================
# DESIGN SPACE
# ============================================================
def design_bounds(settings: HullSettings):
    lb_beam = np.full(settings.n_beam_cp, settings.min_half_beam)
    ub_beam = np.full(settings.n_beam_cp, settings.max_half_beam)

    lb_draft = np.full(settings.n_draft_cp, settings.min_draft)
    ub_draft = np.full(settings.n_draft_cp, settings.max_draft)

    lb = np.concatenate([lb_beam, lb_draft])
    ub = np.concatenate([ub_beam, ub_draft])
    return lb, ub


def baseline_design(settings: HullSettings):
    """
    Safe mid-range initial hull.
    """
    beam_cp = np.linspace(2.2, 3.3, settings.n_beam_cp)
    draft_cp = np.linspace(1.6, 2.3, settings.n_draft_cp)
    return np.concatenate([beam_cp, draft_cp])