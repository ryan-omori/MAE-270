import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from smt.surrogate_models import RBF

from truth_model_model import (
    HullSettings,
    design_bounds,
    truth_total_resistance,
    hull_volume,
)


def lhs_samples(n_samples, lb, ub, seed=7):
    dim = len(lb)
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    Xn = sampler.random(n=n_samples)
    return qmc.scale(Xn, lb, ub)


def build_dataset(settings, n_samples=300, seed=7):
    lb, ub = design_bounds(settings)
    X = lhs_samples(n_samples, lb, ub, seed=seed)

    y_R = np.zeros((n_samples, 1))
    y_V = np.zeros((n_samples, 1))

    for i in range(n_samples):
        Rt, _ = truth_total_resistance(X[i], settings)
        V = hull_volume(X[i], settings)
        y_R[i, 0] = Rt
        y_V[i, 0] = V

    return X, y_R, y_V


def remove_near_duplicates(X, yR, yV, tol=1e-6):
    keep = []
    for i in range(len(X)):
        duplicate = False
        for j in keep:
            if np.linalg.norm(X[i] - X[j]) < tol:
                duplicate = True
                break
        if not duplicate:
            keep.append(i)

    keep = np.array(keep, dtype=int)
    return X[keep], yR[keep], yV[keep]


def standardize(y):
    mu = np.mean(y, axis=0)
    sig = np.std(y, axis=0) + 1e-12
    ys = (y - mu) / sig
    return ys, mu, sig


def train_rbf(X_train, y_train):
    sm = RBF(
        d0=1.0,
        poly_degree=1,
        reg=1e-10,
        print_training=False,
        print_prediction=False,
    )
    sm.set_training_values(X_train, y_train)
    sm.train()
    return sm


def metrics(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label}:")
    print(f"  RMSE = {rmse:.6e}")
    print(f"  MAE  = {mae:.6e}")
    print(f"  R2   = {r2:.6f}")


def main():
    settings = HullSettings(
        n_beam_cp=4,
        n_draft_cp=4,
        L=30.0,
        V_ship=6.0,
        target_volume=280.0,
    )

    X, y_R, y_V = build_dataset(settings, n_samples=300, seed=11)
    X, y_R, y_V = remove_near_duplicates(X, y_R, y_V, tol=1e-6)

    X_train, X_val, yR_train, yR_val, yV_train, yV_val = train_test_split(
        X, y_R, y_V, test_size=0.2, random_state=21
    )

    yR_train_s, yR_mu, yR_sig = standardize(yR_train)
    yV_train_s, yV_mu, yV_sig = standardize(yV_train)

    sm_R = train_rbf(X_train, yR_train_s)
    sm_V = train_rbf(X_train, yV_train_s)

    yR_pred = sm_R.predict_values(X_val) * yR_sig + yR_mu
    yV_pred = sm_V.predict_values(X_val) * yV_sig + yV_mu

    metrics(yR_val, yR_pred, "Resistance surrogate")
    metrics(yV_val, yV_pred, "Volume surrogate")

    plt.figure(figsize=(6, 6))
    plt.scatter(yR_val, yR_pred, alpha=0.7)
    mn = min(yR_val.min(), yR_pred.min())
    mx = max(yR_val.max(), yR_pred.max())
    plt.plot([mn, mx], [mn, mx], "--")
    plt.xlabel("Truth resistance")
    plt.ylabel("RBF predicted resistance")
    plt.title("RBF resistance parity plot")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    import pickle
    with open("rbf_resistance.pkl", "wb") as f:
        pickle.dump((sm_R, yR_mu, yR_sig), f)
    with open("rbf_volume.pkl", "wb") as f:
        pickle.dump((sm_V, yV_mu, yV_sig), f)

    print("Saved:")
    print("  rbf_resistance.pkl")
    print("  rbf_volume.pkl")


if __name__ == "__main__":
    main()