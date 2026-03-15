import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.stats.qmc import Halton

# ======================================================
# CONSTANTS
# ======================================================

rho = 1025
nu = 1.19e-6
g = 9.81

L = 100.0
V_ship = 8.0


# ======================================================
# B-SPLINE CURVE
# ======================================================

def bspline_curve(x_ctrl, y_ctrl, degree=3, n=200):

    k = degree
    m = len(x_ctrl)

    t = np.concatenate((

        np.zeros(k),
        np.linspace(0,1,m-k+1),
        np.ones(k)

    ))

    spline = BSpline(t, y_ctrl, k)

    xs = np.linspace(0,1,n)
    ys = spline(xs)

    return xs, ys


# ======================================================
# DESIGN VARIABLES → PHYSICAL PARAMETERS
# ======================================================

def to_physical(h):

    sac = 0.2 + 0.8*h[:6]

    Bmax = 10 + 6*h[6]

    xBmax = 0.4 + 0.2*h[7]

    bow_full = 0.5 + 0.5*h[8]
    stern_full = 0.5 + 0.5*h[9]

    entrance_angle = 10 + 30*h[10]
    run_angle = 5 + 25*h[11]

    return sac, Bmax, xBmax, bow_full, stern_full, entrance_angle, run_angle


# ======================================================
# SECTIONAL AREA CURVE
# ======================================================

def build_sac(sac_ctrl):

    x_ctrl = np.linspace(0,1,len(sac_ctrl)+2)

    y_ctrl = np.concatenate(([0], sac_ctrl, [0]))

    xs, A = bspline_curve(x_ctrl, y_ctrl)

    xs = xs * L

    A = np.maximum(A,0)

    return xs, A


# ======================================================
# BEAM DISTRIBUTION
# ======================================================

def build_beam(xs, Bmax, xBmax, bow_full, stern_full):

    x_ctrl = np.array([
        0,
        0.25*xBmax,
        xBmax,
        (1+xBmax)/2,
        1
    ])

    y_ctrl = np.array([
        Bmax*bow_full*0.4,
        Bmax*bow_full,
        Bmax,
        Bmax*stern_full,
        Bmax*stern_full*0.4
    ])

    x_spline, B = bspline_curve(x_ctrl, y_ctrl)

    B = np.interp(xs/L, x_spline, B)

    return B


# ======================================================
# SECTION GEOMETRY
# ======================================================

def section_geometry(A, B):

    Cs = 0.85

    T = A / ((np.pi/4)*B*Cs + 1e-8)

    P = np.pi*np.sqrt(0.5*((B/2)**2 + T**2))

    return T, P


# ======================================================
# GEOMETRY METRICS
# ======================================================

def compute_geometry(xs, A, B, T, P):

    Volume = np.trapezoid(A, xs)

    WettedSurface = np.trapezoid(P, xs)

    Bmax = np.max(B)
    Tmax = np.max(T)

    Cp = Volume/(L*Bmax*Tmax)

    xcb = np.trapezoid(xs*A, xs)/Volume

    return Volume, WettedSurface, Cp, xcb


# ======================================================
# RESISTANCE MODEL
# ======================================================

def resistance_model(S, Volume, Bmax, Cp, entrance_angle, run_angle):

    Re = V_ship*L/nu

    Cf = 0.075 / (np.log10(Re)-2)**2

    RF = 0.5*rho*V_ship**2*S*Cf

    Fn = V_ship / np.sqrt(g*L)

    CB = Volume/(L*Bmax*Bmax)

    Cw = (

        0.004
        * Fn**4
        * (Bmax/L)
        * (1 + 3*(CB-0.6)**2)
        * (1 + 0.01*entrance_angle)
        * (1 + 0.01*run_angle)

    )

    RW = 0.5*rho*V_ship**2*S*Cw

    return RF + RW


# ======================================================
# TRUE MODEL
# ======================================================

def true_model(h):

    sac, Bmax, xBmax, bow_full, stern_full, entrance_angle, run_angle = to_physical(h)

    xs, A = build_sac(sac)

    B = build_beam(xs, Bmax, xBmax, bow_full, stern_full)

    T, P = section_geometry(A, B)

    Volume, S, Cp, xcb = compute_geometry(xs, A, B, T, P)

    R = resistance_model(S, Volume, Bmax, Cp, entrance_angle, run_angle)

    return R, Volume, S, Cp, xcb


# ======================================================
# DATASET GENERATION
# ======================================================

def generate_dataset(n_samples=500):

    sampler = Halton(d=12)

    H = sampler.random(n_samples)

    Y = []

    for h in H:

        R, V, S, Cp, xcb = true_model(h)

        Y.append([V,S,Cp,xcb,R])

    return H, np.array(Y)
import numpy as np
import joblib

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel




# ======================================================
# GENERATE TRAINING DATA
# ======================================================

H, Y = generate_dataset(500)

X = H
y = Y


# ======================================================
# TRAIN GPR MODELS
# ======================================================

models = []

kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel()

for i in range(y.shape[1]):

    gpr = GaussianProcessRegressor(

        kernel=kernel,
        n_restarts_optimizer=5

    )

    gpr.fit(X, y[:,i])

    models.append(gpr)


# ======================================================
# SAVE MODEL
# ======================================================

joblib.dump(models,"hull_surrogate.pkl")


print("Surrogate trained successfully")