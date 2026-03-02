import numpy as np
from truth_resistance_model import truth_resistance_and_volume
def sample_designs_halton(n_samples: int, n_dim: int = 29, seed: int = 0) -> np.ndarray:
    """
    Returns samples in [0,1]^n_dim using Halton low-discrepancy sampling.
    Tries scipy; if not available, falls back to a simple LHS.
    """
    try:
        from scipy.stats import qmc
        sampler = qmc.Halton(d=n_dim, scramble=True, seed=seed)
        X = sampler.random(n=n_samples)
        return X.astype(float)
    except Exception:
        # Fallback: basic Latin Hypercube Sampling (LHS)
        rng = np.random.default_rng(seed)
        # LHS: for each dim, stratify and shuffle
        cut = np.linspace(0.0, 1.0, n_samples + 1)
        u = rng.random((n_samples, n_dim))
        X = np.zeros((n_samples, n_dim))
        for j in range(n_dim):
            a = cut[:n_samples]
            b = cut[1:n_samples + 1]
            pts = a + (b - a) * u[:, j]
            rng.shuffle(pts)
            X[:, j] = pts
        return X.astype(float)


def generate_dataset(
    n_samples: int,
    truth_fn,
    seed: int = 0,
    n_dim: int = 29,
):
    """
    Builds a dataset:
      inputs: h1..h29 in [0,1]
      outputs: R_total, V
    Returns: (H, R, V, meta_list)
    """
    H = sample_designs_halton(n_samples, n_dim=n_dim, seed=seed)
    R = np.zeros(n_samples, dtype=float)
    V = np.zeros(n_samples, dtype=float)
    meta_list = []

    for i in range(n_samples):
        Ri, Vi, meta = truth_fn(H[i])
        R[i] = Ri
        V[i] = Vi
        meta_list.append(meta)

    return H, R, V, meta_list


def save_dataset_csv(path: str, H: np.ndarray, R: np.ndarray, V: np.ndarray):
    """
    Writes CSV with columns: h1..h29,R,V
    """
    import csv
    n_samples, n_dim = H.shape
    header = [f"h{k+1}" for k in range(n_dim)] + ["R", "V"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n_samples):
            w.writerow(list(H[i]) + [float(R[i]), float(V[i])])
# assumes truth_resistance_and_volume(h, ...) is defined (the one I gave you earlier)
H, R, V, meta = generate_dataset(
    n_samples=500,
    truth_fn=lambda h: truth_resistance_and_volume(h, U=2.0, L=1.0, B_max=0.75, T_max=0.5),
    seed=1
)

save_dataset_csv("hull_dataset.csv", H, R, V)
print("Saved dataset:", H.shape, R.shape, V.shape)
