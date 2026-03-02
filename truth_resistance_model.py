import numpy as np

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def truth_resistance_and_volume(
    h,
    U=2.0,
    L=1.0,
    B_max=0.75,
    T_max=0.5,
    rho=1000.0,
    nu=1.0e-6,
    g=9.81,
    form_factor_base=0.15,
    wave_a=0.55,
    wave_m=4.0,
):
    """
    Fast 'truth' model for dataset generation.

    Inputs
    ------
    h : array-like, shape (29,)
        Design variables normalized to [0, 1].
    U : float
        Speed (m/s). Default 2.0.
    L : float
        Length (m). Fixed at 1.0.
    B_max : float
        Max beam constraint (m). Fixed at 0.75.
    T_max : float
        Max draft/height constraint (m). Fixed at 0.5.

    Outputs
    -------
    R_total : float
        Total resistance (N).
    V : float
        Volume proxy (m^3).
    meta : dict
        Useful intermediate values for debugging/plots.
    """
    h = np.asarray(h, dtype=float).reshape(-1)
    if h.size != 29:
        raise ValueError(f"Expected h to have 29 elements, got {h.size}")

    # Clamp inputs for safety (your sampling should already be [0,1])
    h = np.clip(h, 0.0, 1.0)

    # --- Shape descriptors from the 29 parameters ---
    # These groupings are "paper-inspired" (bow/mid/stern / chine & bottom regions),
    # but simplified so you don't need full geometry to get a useful mapping.

    p_bow   = np.mean([h[1], h[2], h[22]])               # h2, h3, h23 (0-indexed)
    p_mid   = np.mean(h[11:22])                          # h12..h22
    p_stern = np.mean(h[23:29])                          # h24..h29
    p_deck  = np.mean(h[5:11])                           # h6..h11
    p_prof  = np.mean(h[0:5])                            # h1..h5

    # Fullness factor: higher => fuller hull (more volume, more wave drag)
    fullness_raw = (
        0.40 * p_mid +
        0.20 * p_stern +
        0.15 * p_deck -
        0.10 * p_bow +
        0.10 * p_prof
    )
    # Map to a nice range ~[0.75, 1.25]
    fullness = 0.75 + 0.50 * np.clip(fullness_raw, 0.0, 1.0)

    # Slenderness-like factor based on "bow sharpness" and mid fullness
    # Sharper bow (lower p_bow) should reduce wave drag a bit.
    sharpness = 1.0 - p_bow  # 0..1
    slender_wave_factor = 0.85 + 0.30 * (p_mid) - 0.20 * (sharpness)

    # --- Geometric scales respecting your constraints ---
    # We keep L fixed. Beam and draft are treated as limited by B_max/T_max and shape.
    # They never exceed constraints.
    B = B_max * (0.70 + 0.30 * p_deck)     # in [0.525, 0.75]
    T = T_max * (0.65 + 0.35 * p_mid)      # in [0.325, 0.5]

    # --- Volume proxy (m^3) ---
    # Base box volume scaled by fullness and a smooth nonlinearity
    V_box = L * B * T
    # Add smooth variation without creating crazy extremes.
    V = V_box * fullness * (0.92 + 0.10 * _sigmoid(4.0 * (p_mid - 0.5)))

    # --- Wetted surface proxy (m^2) ---
    # Rectangular-prism-like estimate, nudged by fullness.
    # (You could also use a prolate/ellipsoid approximation, but this is stable.)
    S = L * (2.0 * T + B) * (0.90 + 0.25 * (fullness - 0.75))  # mild scaling

    # --- Frictional resistance (ITTC-57 line) ---
    Re = max(U * L / nu, 1e4)
    Cf = 0.075 / (np.log10(Re) - 2.0) ** 2

    # Form factor: increases with fullness (more viscous pressure drag)
    k = form_factor_base + 0.25 * (fullness - 0.75)  # mild effect
    R_fric = 0.5 * rho * U**2 * S * Cf * (1.0 + k)

    # --- Wave resistance proxy ---
    Fn = U / np.sqrt(g * L)

    # Make a smooth positive multiplier from shape; keep it stable
    phi = np.clip(slender_wave_factor, 0.6, 1.6) * np.clip(fullness, 0.6, 1.6)

    # Wave proxy scaled by dynamic pressure * area, grows with Fn^m
    R_wave = 0.5 * rho * U**2 * S * wave_a * (Fn ** wave_m) * phi

    R_total = R_fric + R_wave

    meta = {
        "B": B,
        "T": T,
        "S": S,
        "V_box": V_box,
        "fullness": fullness,
        "Fn": Fn,
        "Re": Re,
        "Cf": Cf,
        "k": k,
        "R_fric": R_fric,
        "R_wave": R_wave,
    }

    return float(R_total), float(V), meta
h0 = np.full(29, 0.5)
R, V, meta = truth_resistance_and_volume(h0)
print("R_total [N] =", R)
print("Volume [m^3] =", V)
print("Beam B [m] =", meta["B"], "Draft T [m] =", meta["T"])
print("Constraint: Length L[m] =",float(1))
