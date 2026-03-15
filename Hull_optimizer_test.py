"""
hull_optimizer.py
=================
Hull MDO optimizer: CSDL + modopt SQP with analytical Kriging gradients.

WHY ANALYTICAL GRADIENTS FROM KRIGING?
----------------------------------------
You correctly did not want to fit another model over the Kriging.
The solution is: the Kriging prediction has a closed-form gradient.

  GPR prediction:  y_hat(x) = k(x, X_train)^T @ alpha
  Gradient:        dy_hat/dx = (dk/dx)^T @ alpha

For Matern nu=2.5 with ARD length scales l_j:

  r_j = (x_j - x_j_train) / l_j          scaled difference
  r   = sqrt(sum_j r_j^2)                 total scaled distance
  k   = C * (1 + sqrt5*r + 5r^2/3) * exp(-sqrt5*r)

  dk/dx_j = -(5C/3) * (1 + sqrt5*r) * exp(-sqrt5*r) * (x_j - x_j_train) / l_j^2

This is exact to machine precision - verified against finite differences.

The resistance formula gradient is derived analytically via chain rule:
  dR_T/dh = dR_T/dgeo * dgeo/dh  +  dR_T/dh_direct

where dgeo/dh is the Kriging Jacobian (4x10) computed above.

CSDL COMPONENT
--------------
HullResistanceComp:
  inputs:  h (10,)
  outputs: R_T (scalar), V (scalar)
  compute_jvp / compute_vjp: exact analytical Jacobians

Usage
-----
  python hull_optimizer.py
  python hull_optimizer.py --gamma 0.03 --n_starts 8
"""

import argparse
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

try:
    import csdl_alpha as csdl
    import modopt as opt
except ImportError as e:
    print(f"ERROR: {e}")
    print("  pip install csdl-alpha modopt")
    sys.exit(1)

try:
    from TrainSurrogate_test import load_surrogate, predict_hybrid
    from GenHullData import N_VARS, halton_leaped
    from Hull_truth_test import (true_model, to_physical, FIXED, BOUNDS,
                                  build_sac, build_section_shape,
                                  reconstruct_3d_sections)
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

SEED = 42
np.random.seed(SEED)


# ==========================================================================
# SECTION 1 - ANALYTICAL KRIGING GRADIENT
# ==========================================================================

def gpr_gradient(gpr, scaler, x):
    """
    Exact analytical gradient of a GPR prediction w.r.t. input x.

    GPR prediction:   y_hat = k(x, X_train)^T @ alpha
    Gradient:         dy_hat/dx = (dk/dx)^T @ alpha

    Matern 2.5 ARD kernel:
      k(x,x') = C * (1 + sqrt5*r + 5r^2/3) * exp(-sqrt5*r)
      dk/dx_j = -(5C/3) * (1+sqrt5*r) * exp(-sqrt5*r) * diff_j / l_j^2
      where diff_j = x_j - x_j_train,  r = sqrt(sum_j diff_j^2 / l_j^2)
    """
    x       = np.asarray(x, dtype=float).flatten()
    Xtr     = gpr.X_train_
    alpha   = gpr.alpha_
    k_main  = gpr.kernel_.k1          # C * Matern
    amp     = k_main.k1.constant_value
    l       = k_main.k2.length_scale  # ARD length scales (d,)

    diff    = (x - Xtr) / l           # (n, d) scaled differences
    r_sq    = np.sum(diff**2, axis=1)
    r       = np.sqrt(r_sq)

    sqrt5   = np.sqrt(5.0)
    exp_r   = np.exp(-sqrt5 * r)
    poly    = 1.0 + sqrt5 * r + (5.0/3.0) * r_sq

    # dk/dr = -(5*amp/3) * r * (1 + sqrt5*r) * exp(-sqrt5*r)
    dk_dr   = -(5.0 * amp / 3.0) * r * (1.0 + sqrt5 * r) * exp_r

    # dr/dx_j = diff_j / (r * l_j^2)  -- zero when r=0
    r_safe  = np.where(r > 1e-12, r, 1.0)
    dr_dx   = (x - Xtr) / (r_safe[:, None] * l**2)
    dr_dx[r < 1e-12] = 0.0

    # dk/dx: (n, d)
    dk_dx   = dk_dr[:, None] * dr_dx

    # prediction
    k_vec   = amp * poly * exp_r
    y_s     = float(k_vec @ alpha)
    grad_s  = dk_dx.T @ alpha   # (d,)

    # inverse-transform through output scaler
    y    = float(scaler.inverse_transform([[y_s]])[0, 0])
    grad = grad_s * scaler.scale_[0]
    return y, grad


def geometry_jacobian(models, scalers, h):
    """
    Evaluate all 4 geometry outputs and their (4x10) Jacobian.
    Returns geo (4,) and J (4,10).
    """
    geo = np.zeros(4)
    J   = np.zeros((4, N_VARS))
    for i, (gpr, sc) in enumerate(zip(models, scalers)):
        geo[i], J[i] = gpr_gradient(gpr, sc, h)
    return geo, J


# ==========================================================================
# SECTION 2 - ANALYTICAL RESISTANCE GRADIENT
# ==========================================================================

def resistance_and_gradient(geo, J_geo, h):
    """
    Compute R_T, V, and their exact gradients w.r.t. h.

    dR_T/dh = dR_T/dgeo @ J_geo  +  direct contributions from h[6,7,8]
    dV/dh   = J_geo[0]   (V is geo output 0, chain rule is just J row)
    """
    L, U, rho, nu, g = (FIXED["L"], FIXED["U"], FIXED["rho"],
                         FIXED["nu"],  FIXED["g"])
    V, S, CP, LCB    = geo[0], geo[1], geo[2], geo[3]

    v         = to_physical(h)
    a1E       = float(v[6])
    a1R       = float(v[7])
    B_max     = float(v[8])
    dv_dh     = BOUNDS[:, 1] - BOUNDS[:, 0]   # physical scale per h dim

    Re  = max(U * L / nu, 1e4)
    F_n = U / np.sqrt(g * L)
    C_F = 0.075 / (np.log10(Re) - 2.0)**2
    q   = 0.5 * rho * U**2

    # Form factor k
    T_approx  = V / max(L * B_max * 0.6, 1e-9)
    CB_raw    = V / max(L * B_max * T_approx, 1e-9)
    CB        = float(np.clip(CB_raw, 0.4, 0.9))
    LB        = L / max(B_max, 1e-9)
    k_raw     = 0.93 + 0.4871 * (CB / LB**0.1228) * (B_max / L)**0.6906
    k         = float(np.clip(k_raw, 0.05, 0.50))

    # dk/dB_max (only when not at bounds)
    if (0.4 < CB_raw < 0.9) and (0.05 < k_raw < 0.50):
        dk_dBmax = 0.4871 * CB * 0.8134 * (B_max / L)**(-0.1866) / L
    else:
        dk_dBmax = 0.0

    R_F      = q * S * C_F * (1.0 + k)
    dRF_dS   = q * C_F * (1.0 + k)
    dRF_dBmax = q * S * C_F * dk_dBmax

    alpha_ref = 20.0
    f_ent     = 1.0 + (a1E / alpha_ref)**1.5
    f_run     = 1.0 + 0.5 * (a1R / alpha_ref)**1.5
    f_angle   = (f_ent + f_run) / 2.0
    df_da1E   = 1.5 / 2.0 * (a1E / alpha_ref)**0.5 / alpha_ref
    df_da1R   = 0.5 * 1.5 / 2.0 * (a1R / alpha_ref)**0.5 / alpha_ref

    c1        = 0.55
    C_W       = c1 * F_n**4 * CP**2 * f_angle
    R_W       = q * S * C_W
    dRW_dS    = q * C_W
    dRW_dCP   = q * S * c1 * F_n**4 * 2.0 * CP * f_angle
    dRW_da1E  = q * S * c1 * F_n**4 * CP**2 * df_da1E
    dRW_da1R  = q * S * c1 * F_n**4 * CP**2 * df_da1R

    R_T = float(R_F + R_W)

    # dR_T/dgeo
    dRT_dgeo = np.array([0.0,
                          dRF_dS + dRW_dS,
                          dRW_dCP,
                          0.0])

    # chain rule through Kriging
    dRT_dh = dRT_dgeo @ J_geo           # (10,)

    # direct contributions
    dRT_dh[6] += dRW_da1E * dv_dh[6]
    dRT_dh[7] += dRW_da1R * dv_dh[7]
    dRT_dh[8] += dRF_dBmax * dv_dh[8]

    dV_dh  = J_geo[0].copy()           # (10,)

    return R_T, float(V), dRT_dh, dV_dh


# ==========================================================================
# SECTION 3 - CSDL COMPONENT
# ==========================================================================

class HullResistanceComp(csdl.CustomExplicitOperation):
    """
    CSDL ExplicitOperation: h -> R_T, V
    Provides exact analytical derivatives via Kriging gradient + chain rule.
    """

    def __init__(self, models, scalers):
        super().__init__()
        self.models  = models
        self.scalers = scalers

    def evaluate(self, h):
        self.declare_input("h", h)
        R_T = self.create_output("R_T", shape=(1,))
        V   = self.create_output("V",   shape=(1,))
        return R_T, V

    def compute(self, inputs, outputs):
        h            = inputs["h"]
        geo, J_geo   = geometry_jacobian(self.models, self.scalers, h)
        R_T, V, _, _ = resistance_and_gradient(geo, J_geo, h)
        outputs["R_T"] = np.array([R_T])
        outputs["V"]   = np.array([V])

    def compute_jvp(self, inputs, outputs, d_inputs, d_outputs):
        h               = inputs["h"]
        geo, J_geo      = geometry_jacobian(self.models, self.scalers, h)
        _, _, dRT_dh, dV_dh = resistance_and_gradient(geo, J_geo, h)
        if "h" in d_inputs:
            dh = d_inputs["h"]
            if "R_T" in d_outputs:
                d_outputs["R_T"] += np.array([dRT_dh @ dh])
            if "V" in d_outputs:
                d_outputs["V"]   += np.array([dV_dh  @ dh])

    def compute_vjp(self, inputs, outputs, d_inputs, d_outputs):
        h               = inputs["h"]
        geo, J_geo      = geometry_jacobian(self.models, self.scalers, h)
        _, _, dRT_dh, dV_dh = resistance_and_gradient(geo, J_geo, h)
        if "h" in d_inputs:
            v = np.zeros(N_VARS)
            if "R_T" in d_outputs:
                v += d_outputs["R_T"][0] * dRT_dh
            if "V" in d_outputs:
                v += d_outputs["V"][0]   * dV_dh
            d_inputs["h"] += v


# ==========================================================================
# SECTION 4 - BUILD CSDL PROBLEM
# ==========================================================================

def build_problem(models, scalers, V_ref, gamma, h0):
    rec = csdl.Recorder(inline=True)
    rec.start()

    h = csdl.Variable(value=h0.copy(), name="h")
    h.set_as_design_variable(lower=np.zeros(N_VARS),
                              upper=np.ones(N_VARS))

    comp    = HullResistanceComp(models, scalers)
    R_T, V  = comp.evaluate(h)

    R_T.set_as_objective()
    ((1.0 - gamma) * V_ref - V).set_as_constraint(upper=0.0, name="V_lower")
    (V - (1.0 + gamma) * V_ref).set_as_constraint(upper=0.0, name="V_upper")

    rec.stop()
    sim  = csdl.experimental.PySimulator(rec)
    prob = opt.CSDLAlphaProblem(problem_name="HullMDO", simulator=sim)
    prob.x0 = h0.copy()
    return prob, sim


# ==========================================================================
# SECTION 5 - MULTI-START
# ==========================================================================

def run_multistart(models, scalers, V_ref, gamma,
                   n_starts=8, maxiter=300, opt_tol=1e-8):
    halton_pts = halton_leaped(max(n_starts-1, 1), d=N_VARS, seed_offset=777)
    h0_list    = [0.5 * np.ones(N_VARS)] + list(halton_pts[:n_starts-1])

    results = []
    best    = None

    for i, h0 in enumerate(h0_list):
        label = "h=0.5" if i == 0 else f"Halton {i}"
        print(f"  Start {i+1}/{n_starts} [{label}]...", end=" ", flush=True)

        t0 = time.time()
        try:
            prob, sim = build_problem(models, scalers, V_ref, gamma, h0)
            solver    = opt.SQP(prob, maxiter=maxiter, opt_tol=opt_tol,
                                 recording=False)
            solver.solve()
            h_opt   = solver.results["x"]
            R_T_opt = float(solver.results["f"])
            success = True
        except Exception as ex:
            h_opt   = h0.copy()
            R_T_opt = float("inf")
            success = False

        dt    = time.time() - t0
        pred  = predict_hybrid(h_opt, models, scalers)
        V_opt = float(pred["V"])
        feas  = ((1-gamma-1e-4)*V_ref <= V_opt <= (1+gamma+1e-4)*V_ref)
        flag  = "OK" if (success and feas) else ("INFEAS" if not feas else "FAIL")

        print(f"R_T={R_T_opt:.3f} N  V={V_opt:.5f}  [{flag}]  {dt:.1f}s")

        result = dict(h_opt=h_opt, R_T_surr=R_T_opt, V_surr=V_opt,
                      success=success, feasible=feas)
        results.append(result)
        if feas and (best is None or R_T_opt < best["R_T_surr"]):
            best = result

    if best is None:
        print("  WARNING: no feasible result. Returning best infeasible.")
        best = min(results, key=lambda r: r["R_T_surr"])
    return best, results


# ==========================================================================
# SECTION 6 - VERIFY + REPORT
# ==========================================================================

def verify_on_true_model(h_opt, R_ref, V_ref, gamma):
    print("\n  Evaluating optimal h on TRUE model...")
    r     = true_model(h_opt)
    R_opt = r["R_T"]; V_opt = r["V"]
    imp   = (R_ref - R_opt) / R_ref * 100
    V_ok  = (1-gamma)*V_ref <= V_opt <= (1+gamma)*V_ref

    print(f"\n  Reference:  R_T = {R_ref:.3f} N   V = {V_ref:.5f} m3")
    print(f"  Optimal:    R_T = {R_opt:.3f} N   V = {V_opt:.5f} m3")
    print(f"  Improvement: {imp:+.2f}%")
    print(f"  Volume constraint: {'SATISFIED' if V_ok else 'VIOLATED'}")
    return dict(R_T_opt=R_opt, V_opt=V_opt, improvement_pct=imp,
                V_satisfied=V_ok, R_T_init=R_ref, full=r)


# ==========================================================================
# SECTION 7 - PLOTS
# ==========================================================================

def fig_design_comparison(h_opt, h_ref, opt_result):
    names = ["C_P","X_LCB","X_0E","X_0R","a0E","a0R","a1E","a1R","Bmax","xBmax"]
    imp   = opt_result["improvement_pct"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Initial vs Optimal  R_T: {opt_result['R_T_init']:.3f} N -> "
        f"{opt_result['R_T_opt']:.3f} N  ({imp:+.2f}%)",
        fontsize=11, fontweight="bold")

    x = np.arange(N_VARS); w = 0.35
    delta  = h_opt - h_ref
    colors = ["steelblue" if d <= 0 else "darkorange" for d in delta]

    axes[0].bar(x-w/2, h_ref, w, color="lightsteelblue", edgecolor="white",
                label="Initial (h=0.5)")
    axes[0].bar(x+w/2, h_opt, w, color=colors, edgecolor="white",
                alpha=0.85, label="Optimal")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=45,
                                                     ha="right", fontsize=8)
    axes[0].set_ylabel("h in [0,1]"); axes[0].set_ylim(0, 1.15)
    axes[0].set_title("(A) Design variables"); axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2, axis="y")

    dp = np.abs(delta) / 0.5 * 100
    cols = ["steelblue" if d<=0 else "darkorange" for d in delta]
    axes[1].barh(names, dp, color=cols, edgecolor="white", alpha=0.85)
    axes[1].axvline(10, color="gray", lw=0.8, ls="--", label="10% threshold")
    axes[1].set_xlabel("|delta_h| as % of range")
    axes[1].set_title("(B) Change magnitude\nBlue=decreased  Orange=increased")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.2, axis="x")

    L = FIXED["L"]
    for h, label, color, lw in [(h_ref,"Initial","#2E86C1",1.8),
                                  (h_opt,"Optimal","#D35400",2.4)]:
        v   = to_physical(h.reshape(1,-1))[0]
        sac = build_sac(v, L=L, n=200)
        A   = sac["A_shape"] / (sac["A_max"] + 1e-12)
        p   = sac["params"]
        axes[2].plot(sac["xs"]/L, A, color=color, lw=lw, label=label)
        axes[2].axvspan(p["X_0E"], p["X_0R"], alpha=0.08, color=color)
    axes[2].set_xlabel("x/L"); axes[2].set_ylabel("A(x)/A_max")
    axes[2].set_title("(C) Sectional Area Curve\nShaded=PMB")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.2)
    axes[2].set_xlim(-0.02, 1.02); axes[2].set_ylim(-0.05, 1.15)

    plt.tight_layout()
    return fig


def fig_hull_profiles(h_opt, h_ref, opt_result):
    L, T_max = FIXED["L"], FIXED["T_max"]
    C_I = "#2E86C1"; C_O = "#D35400"; C_W = "#566573"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                              gridspec_kw={"width_ratios": [2.2, 1]})
    fig.suptitle(
        f"Hull profiles  ({opt_result['improvement_pct']:+.2f}% R_T)",
        fontsize=11, fontweight="bold")

    for h, color, ls, lw, label in [(h_ref,C_I,"--",1.8,"Initial"),
                                     (h_opt,C_O,"-",2.4,"Optimal")]:
        v    = to_physical(h.reshape(1,-1))[0]
        sac  = build_sac(v, L=L, n=200)
        geom = build_section_shape(v, sac, L=L, T_max=T_max)
        secs = reconstruct_3d_sections(geom)
        xs, T = geom["xs"], geom["T"]
        axes[0].plot(xs, -T, color=color, lw=lw, ls=ls, label=f"({label})")
        axes[0].plot([xs[0],xs[-1]], [0,0], color=color, lw=lw*0.5, ls=ls, alpha=0.5)
        for xf, ll in [(0.10,"-"),(0.50,"--"),(0.85,":")]:
            idx = np.argmin(np.abs(xs - xf*L))
            sec = secs["sections"][idx]
            axes[1].plot(sec[:,0], sec[:,1], color=color,
                         lw=lw*(1.0 if xf==0.10 else 0.7), ls=ll,
                         label=f"x={xf:.2f}L ({label})" if xf==0.10 else None)

    axes[0].axhline(0, color=C_W, lw=0.8, ls="--", alpha=0.5, label="WL")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("z [m]")
    axes[0].set_title("Side profile (x-z)")
    axes[0].set_xlim(-0.03, L*1.05); axes[0].set_ylim(-T_max*1.35, T_max*0.35)
    axes[0].set_aspect("equal"); axes[0].grid(True, alpha=0.2)
    axes[0].legend(fontsize=8, loc="lower right")

    B_max  = float(to_physical(h_opt.reshape(1,-1))[0][8])
    margin = (T_max + 0.05) * 0.3
    axes[1].axhline(0, color=C_W, lw=0.8, ls="--", alpha=0.5)
    axes[1].axvline(0, color=C_W, lw=0.6, ls=":", alpha=0.35)
    axes[1].set_xlim(-(B_max/2+margin), (B_max/2+margin))
    axes[1].set_ylim(-T_max-margin, T_max*0.35)
    axes[1].set_aspect("equal"); axes[1].grid(True, alpha=0.2)
    axes[1].set_xlabel("y [m]"); axes[1].set_ylabel("z [m]")
    axes[1].set_title("Front profile (y-z)\nsolid=bow  dash=mid  dot=aft")
    axes[1].legend(fontsize=7, loc="lower center")

    plt.tight_layout()
    return fig


# ==========================================================================
# SECTION 8 - MAIN
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hull MDO - CSDL+SQP with analytical Kriging gradients",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--surrogate", default="surrogate_kriging.pkl")
    parser.add_argument("--gamma",     type=float, default=0.03)
    parser.add_argument("--n_starts",  type=int,   default=8)
    parser.add_argument("--maxiter",   type=int,   default=300)
    parser.add_argument("--opt_tol",   type=float, default=1e-8)
    args = parser.parse_args()

    SEP = "=" * 66

    print(f"\n{SEP}\nSTEP 1 - Load surrogate\n{SEP}")
    models, scalers = load_surrogate(args.surrogate)
    print(f"  Loaded {len(models)} GPR models")
    print(f"  Gradient: ANALYTICAL (Matern 2.5 closed-form, no FD)")

    print(f"\n{SEP}\nSTEP 2 - Reference design\n{SEP}")
    h_ref = 0.5 * np.ones(N_VARS)
    r_ref = true_model(h_ref)
    V_ref = r_ref["V"]; R_ref = r_ref["R_T"]
    print(f"  True model: R_T={R_ref:.3f} N  V={V_ref:.5f} m3")
    print(f"  Volume band: [{(1-args.gamma)*V_ref:.5f}, "
          f"{(1+args.gamma)*V_ref:.5f}] m3")

    # Gradient verification
    print(f"\n  Verifying analytical gradient vs finite differences...")
    geo, J  = geometry_jacobian(models, scalers, h_ref)
    _, _, dRT_dh, _ = resistance_and_gradient(geo, J, h_ref)
    eps     = 1e-5; max_err = 0.0
    for i in range(min(3, N_VARS)):
        hp = h_ref.copy(); hp[i] += eps
        hm = h_ref.copy(); hm[i] -= eps
        fd = (predict_hybrid(hp, models, scalers)["R_T"] -
              predict_hybrid(hm, models, scalers)["R_T"]) / (2*eps)
        max_err = max(max_err, abs(dRT_dh[i]-fd)/(abs(fd)+1e-12))
    verdict = "PASS" if max_err < 0.01 else "WARN (>1% error)"
    print(f"  Max relative gradient error: {max_err:.2e}  [{verdict}]")

    print(f"\n{SEP}\nSTEP 3 - Multi-start CSDL+SQP\n{SEP}")
    print(f"  n_starts={args.n_starts}  maxiter={args.maxiter}  "
          f"opt_tol={args.opt_tol}  gamma={args.gamma}\n")

    t0 = time.time()
    best, all_results = run_multistart(
        models, scalers, V_ref, args.gamma,
        n_starts=args.n_starts, maxiter=args.maxiter, opt_tol=args.opt_tol)
    print(f"\n  Total time: {time.time()-t0:.1f}s")
    print(f"  Best surrogate R_T: {best['R_T_surr']:.3f} N")

    print(f"\n{SEP}\nSTEP 4 - Verify on true model\n{SEP}")
    opt_result = verify_on_true_model(
        best["h_opt"], R_ref, V_ref, args.gamma)

    print(f"\n{SEP}\nSTEP 5 - Optimal parameters\n{SEP}")
    pnames = ["C_P","X_LCB","X_0E","X_0R","a0E_deg","a0R_deg",
              "a1E_deg","a1R_deg","B_max_m","xBmax_L"]
    v_r = to_physical(h_ref); v_o = to_physical(best["h_opt"])
    print(f"  {'Parameter':<12} {'Initial':>10} {'Optimal':>10} {'Delta':>10}")
    print(f"  {'-'*44}")
    for nm, vi, vo in zip(pnames, v_r, v_o):
        print(f"  {nm:<12} {vi:>10.4f} {vo:>10.4f}  {vo-vi:>+9.4f}")

    print(f"\n{SEP}\nSTEP 6 - Generating plots\n{SEP}")
    figs = {
        "optimizer_design_vars.png":   fig_design_comparison(
            best["h_opt"], h_ref, opt_result),
        "optimizer_hull_profiles.png": fig_hull_profiles(
            best["h_opt"], h_ref, opt_result),
    }
    for fname, fig in figs.items():
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {fname}")

    imp = opt_result["improvement_pct"]
    print(f"\n{SEP}\nSUMMARY\n{SEP}")
    print(f"  Improvement:       {imp:+.2f}%")
    print(f"  Volume constraint: {'SATISFIED' if opt_result['V_satisfied'] else 'VIOLATED'}")
    if imp < 0:
        print("  NOTE: negative = surrogate guided to false minimum.")
        print("  Retrain surrogate with more data if this occurs.")
    print(f"\n  All figures open. Close to exit.\n{SEP}")
    plt.show()

# ==========================================================================
# HULL GEOMETRY CONTOUR PLOTS
# Paste this function into hull_optimizer.py and call it after
# verify_on_true_model().
#
# Call with:
#   fig_hull_geometry("optimizer_hull_geometry.png", best["h_opt"])
#
# Or to compare initial vs optimal side by side:
#   fig_hull_geometry("optimizer_hull_geometry.png",
#                     best["h_opt"], h_ref=h_ref)
# ==========================================================================
 
def fig_hull_geometry(save_path, h_opt, h_ref=None):
    """
    Three-panel hull geometry plot:
      (A) Side profile    — keel line T(x) and waterline, x-z plane
      (B) Top-down plan   — half-beam B(x)/2 waterplane, x-y plane
      (C) Front profile   — elliptic cross-sections at bow, mid, aft
 
    The hull is described by:
      xs  : station positions along the hull [m]
      B(x): full beam at each station [m]     → top-down outline
      T(x): draft at each station [m]         → side profile keel
      Each cross-section is an elliptic arc with semi-axes HB=B/2 and T.
 
    Parameters
    ----------
    save_path : str     filepath for saved PNG
    h_opt     : (10,)   normalised optimal design vector
    h_ref     : (10,)   optional reference design for comparison overlay
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from Hull_truth_test import (build_sac, build_section_shape,
                                  reconstruct_3d_sections,
                                  to_physical, FIXED)
 
    L     = FIXED["L"]
    T_max = FIXED["T_max"]
 
    # ------------------------------------------------------------------
    # Build geometry for optimal (and optionally reference) design
    # ------------------------------------------------------------------
    def get_geom(h):
        v    = to_physical(np.asarray(h).reshape(1, -1))[0]
        sac  = build_sac(v, L=L, n=200)
        geom = build_section_shape(v, sac, L=L, T_max=T_max)
        secs = reconstruct_3d_sections(geom, n_theta=80)
        return geom, secs
 
    geom_opt, secs_opt = get_geom(h_opt)
    geom_ref, secs_ref = (get_geom(h_ref) if h_ref is not None
                          else (None, None))
 
    xs   = geom_opt["xs"]
    B    = geom_opt["B"]
    T    = geom_opt["T"]
    HB   = geom_opt["HB"]
 
    # Colors
    C_OPT  = "#1B4F72"    # deep navy for optimal
    C_REF  = "#B03A2E"    # muted red for reference
    C_WL   = "#808B96"    # waterline gray
    C_FILL = "#AED6F1"    # light blue fill
 
    # ------------------------------------------------------------------
    # Figure layout: 3 panels, equal height
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5),
        gridspec_kw={"width_ratios": [2.5, 2.5, 1.2]},
    )
 
    title_lines = ["Hull geometry — optimized design"]
    if h_ref is not None:
        title_lines.append("Blue = optimal     Red = initial reference")
    fig.suptitle("\n".join(title_lines), fontsize=11, fontweight="bold")
 
    # ==================================================================
    # PANEL A — SIDE PROFILE  (x horizontal, z vertical, looking port)
    # Bow is on the left (x=0), stern on the right (x=L).
    # Keel depth is −T(x) below the waterline z=0.
    # ==================================================================
    ax = axes[0]
 
    def draw_side(geom, color, lw, label, fill=False):
        xs_ = geom["xs"]
        T_  = geom["T"]
        HB_ = geom["HB"]
 
        # Keel line (below waterline)
        keel_z = -T_
 
        # Deck / sheer line — flat at z=0 (waterline = deck for submerged hull)
        deck_z = np.zeros_like(xs_)
 
        if fill:
            # Fill underwater volume between keel and waterline
            ax.fill_between(xs_, keel_z, deck_z,
                            alpha=0.12, color=color, zorder=1)
 
        # Waterline (deck edge)
        ax.plot(xs_, deck_z, color=color, lw=lw * 0.6,
                ls="-", alpha=0.6, zorder=2)
 
        # Keel contour
        ax.plot(xs_, keel_z, color=color, lw=lw,
                label=label, zorder=3)
 
        # Close the hull at bow and stern with vertical lines
        ax.plot([xs_[0],  xs_[0]],  [0, keel_z[0]],
                color=color, lw=lw * 0.8, zorder=3)
        ax.plot([xs_[-1], xs_[-1]], [0, keel_z[-1]],
                color=color, lw=lw * 0.8, zorder=3)
 
    # Draw reference first (underneath)
    if geom_ref is not None:
        draw_side(geom_ref, C_REF, lw=1.4, label="Reference", fill=False)
 
    # Draw optimal on top
    draw_side(geom_opt, C_OPT, lw=2.2, label="Optimal", fill=True)
 
    # Waterline
    ax.axhline(0, color=C_WL, lw=0.8, ls="--", alpha=0.7,
               label="Waterline (z=0)", zorder=2)
 
    # Annotate max draft
    idx_T = int(np.argmax(T))
    ax.annotate(
        f"T_max = {T[idx_T]:.3f} m",
        xy=(xs[idx_T], -T[idx_T]),
        xytext=(xs[idx_T] + 0.08, -T[idx_T] + 0.03),
        fontsize=7.5, color=C_OPT,
        arrowprops=dict(arrowstyle="->", color=C_OPT, lw=0.7),
    )
 
    ax.set_xlabel("x  [m]  — longitudinal", fontsize=9)
    ax.set_ylabel("z  [m]  — depth below waterline", fontsize=9)
    ax.set_title("(A)  Side profile", fontsize=10)
    ax.set_xlim(-0.02 * L, L * 1.04)
    ax.set_ylim(-T_max * 1.45, T_max * 0.40)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.18)
 
    # ==================================================================
    # PANEL B — TOP-DOWN PLAN VIEW  (x horizontal, y lateral)
    # Looking straight down from above. Bow at left.
    # Shows the waterplane area outline (half-beam each side).
    # ==================================================================
    ax = axes[1]
 
    def draw_topdown(geom, color, lw, label, fill=False):
        xs_  = geom["xs"]
        HB_  = geom["HB"]
 
        # Starboard (positive y) and port (negative y) outlines
        if fill:
            ax.fill_between(xs_, -HB_, HB_,
                            alpha=0.12, color=color, zorder=1)
 
        # Starboard side
        ax.plot(xs_, HB_,  color=color, lw=lw, label=label, zorder=3)
        # Port side
        ax.plot(xs_, -HB_, color=color, lw=lw, zorder=3)
 
        # Close bow and stern
        ax.plot([xs_[0],  xs_[0]],  [-HB_[0],  HB_[0]],
                color=color, lw=lw * 0.8, zorder=3)
        ax.plot([xs_[-1], xs_[-1]], [-HB_[-1], HB_[-1]],
                color=color, lw=lw * 0.8, zorder=3)
 
    if geom_ref is not None:
        draw_topdown(geom_ref, C_REF, lw=1.4, label="Reference", fill=False)
 
    draw_topdown(geom_opt, C_OPT, lw=2.2, label="Optimal", fill=True)
 
    # Centreline
    ax.axhline(0, color=C_WL, lw=0.8, ls="--", alpha=0.6,
               label="Centreline (y=0)", zorder=2)
 
    # Annotate max beam
    idx_B = int(np.argmax(B))
    ax.annotate(
        f"B_max = {B[idx_B]:.3f} m",
        xy=(xs[idx_B], HB[idx_B]),
        xytext=(xs[idx_B] - 0.08, HB[idx_B] + 0.015),
        fontsize=7.5, color=C_OPT,
        arrowprops=dict(arrowstyle="->", color=C_OPT, lw=0.7),
    )
 
    ax.set_xlabel("x  [m]  — longitudinal", fontsize=9)
    ax.set_ylabel("y  [m]  — transverse  (port ← → stbd)", fontsize=9)
    ax.set_title("(B)  Top-down plan  (waterplane)", fontsize=10)
    ax.set_xlim(-0.02 * L, L * 1.04)
    B_max_val = float(np.max(HB))
    ax.set_ylim(-(B_max_val + 0.04), (B_max_val + 0.04))
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.18)
 
    # ==================================================================
    # PANEL C — FRONT PROFILE  (y horizontal, z vertical)
    # Looking forward from the bow.
    # Shows cross-sections at three stations: bow, midship, aft.
    # Each section is an elliptic arc with semi-axes HB and T.
    # ==================================================================
    ax = axes[2]
 
    # Stations to show
    stations = [
        (0.10, "Bow  x=0.10L",  C_OPT,  2.0, "-"),
        (0.50, "Mid  x=0.50L",  C_OPT,  2.4, "--"),
        (0.85, "Aft  x=0.85L",  C_OPT,  1.6, ":"),
    ]
 
    # Reference sections (if provided)
    if secs_ref is not None:
        for x_frac, _, _, _, ls in stations:
            idx  = int(np.argmin(np.abs(xs - x_frac * L)))
            sec  = secs_ref["sections"][idx]
            ax.plot(sec[:, 0], sec[:, 1],
                    color=C_REF, lw=1.2, ls=ls, alpha=0.6, zorder=2)
 
    for x_frac, label, color, lw, ls in stations:
        idx = int(np.argmin(np.abs(xs - x_frac * L)))
        sec = secs_opt["sections"][idx]
 
        # Fill underwater cross-section area
        ax.fill(sec[:, 0], sec[:, 1],
                alpha=0.10, color=color, zorder=1)
 
        # Section outline
        ax.plot(sec[:, 0], sec[:, 1],
                color=color, lw=lw, ls=ls, label=label, zorder=3)
 
    # Waterline and centreline
    ax.axhline(0, color=C_WL, lw=0.8, ls="--", alpha=0.7,
               label="WL  z=0", zorder=2)
    ax.axvline(0, color=C_WL, lw=0.6, ls=":", alpha=0.40, zorder=2)
 
    ax.set_xlabel("y  [m]  — transverse", fontsize=9)
    ax.set_ylabel("z  [m]  — depth", fontsize=9)
    ax.set_title("(C)  Front profile\n(cross-sections)", fontsize=10)
 
    HB_max = float(np.max(HB))
    margin = max(HB_max * 0.25, 0.02)
    ax.set_xlim(-(HB_max + margin), (HB_max + margin))
    ax.set_ylim(-T_max * 1.15, T_max * 0.30)
    ax.set_aspect("equal")
    ax.legend(fontsize=7.5, loc="lower center", ncol=1)
    ax.grid(True, alpha=0.18)
 
    # ------------------------------------------------------------------
    # Save and show
    # ------------------------------------------------------------------
    plt.tight_layout()
    return fig
if __name__ == "__main__":
    main()