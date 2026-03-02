import os
import numpy as np
import pandas as pd

# --- marimo bootstrap (keeps file runnable even if opened as plain python) ---
try:
    import marimo
except Exception as e:  # pragma: no cover
    raise ImportError(
        "This script is a marimo app. Install marimo and run:\n"
        "  marimo run surrogate_hull_surrogate.py\n"
        f"Original import error: {e}"
    )

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
# Hull surrogate model (R_total + Volume)

This marimo app:
1) Loads or generates `hull_dataset.csv` (29 design variables in \[0,1\], outputs **R** and **V**)  
2) Fits two surrogates:
- **Resistance**: Gaussian Process Regression (Matern kernel)
- **Volume**: Ridge regression (fast + stable)

Your dataset generation follows `Data_Training_set.py` (Halton/LHS sampling) and
the "truth" model `truth_resistance_model.py`. 
"""
    )
    return


@app.cell
def _():
    import marimo as mo
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    import joblib

    from pathlib import Path

    # local project files
    from Data_Training_set import generate_dataset, save_dataset_csv
    from truth_resistance_model import truth_resistance_and_volume

    # sklearn (surrogate)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from sklearn.linear_model import Ridge

    return (
        ConstantKernel,
        GaussianProcessRegressor,
        Matern,
        Path,
        Pipeline,
        Ridge,
        StandardScaler,
        WhiteKernel,
        generate_dataset,
        go,
        joblib,
        mean_squared_error,
        mo,
        np,
        pd,
        r2_score,
        save_dataset_csv,
        train_test_split,
        truth_resistance_and_volume,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## 1) Dataset

- If `hull_dataset.csv` exists, we load it.
- Otherwise we generate it using the same calls as your script (29D Halton/LHS + truth model).
"""
    )
    return


@app.cell
def _(Path, mo):
    dataset_path = Path("hull_dataset.csv")
    n_samples_ui = mo.ui.slider(100, 3000, value=500, step=50, label="n_samples (if generating)")
    seed_ui = mo.ui.number(value=1, step=1, label="seed (if generating)")
    regenerate_ui = mo.ui.checkbox(value=False, label="force regenerate dataset")
    mo.hstack([n_samples_ui, seed_ui, regenerate_ui])
    return dataset_path, n_samples_ui, regenerate_ui, seed_ui


@app.cell
def _(
    dataset_path,
    generate_dataset,
    mo,
    n_samples_ui,
    np,
    pd,
    regenerate_ui,
    save_dataset_csv,
    seed_ui,
    truth_resistance_and_volume,
):
    def _maybe_generate():
        if dataset_path.exists() and not regenerate_ui.value:
            return

        n_samples = int(n_samples_ui.value)
        seed = int(seed_ui.value)

        H, R, V, _meta = generate_dataset(
            n_samples=n_samples,
            truth_fn=lambda h: truth_resistance_and_volume(
                h, U=2.0, L=1.0, B_max=0.75, T_max=0.5
            ),
            seed=seed,
            n_dim=29,
        )
        save_dataset_csv(str(dataset_path), H, R, V)

    _maybe_generate()

    df = pd.read_csv(dataset_path)
    x_cols = [c for c in df.columns if c.startswith("h")]
    X = df[x_cols].to_numpy(dtype=float)
    y_R = df["R"].to_numpy(dtype=float)
    y_V = df["V"].to_numpy(dtype=float)

    mo.md(
        f"Loaded `{dataset_path}` with **{len(df)}** samples and **{len(x_cols)}** design variables."
    )
    return X, df, x_cols, y_R, y_V


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## 2) Train/validation split + surrogate fit

Notes:
- We scale inputs (important for GP length-scales).
- Resistance is more nonlinear → GP (Matern ν=2.5 is a common default).
- Volume here is very smooth → Ridge works extremely well and is cheap.
"""
    )
    return


@app.cell
def _(
    ConstantKernel,
    GaussianProcessRegressor,
    Matern,
    Pipeline,
    Ridge,
    StandardScaler,
    WhiteKernel,
    X,
    mean_squared_error,
    np,
    r2_score,
    train_test_split,
    y_R,
    y_V,
):
    X_train, X_val, yR_train, yR_val, yV_train, yV_val = train_test_split(
        X, y_R, y_V, test_size=0.2, random_state=0
    )

    # --- Resistance GP ---
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))

    gp_R = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "gpr",
                GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    n_restarts_optimizer=2,
                    random_state=0,
                ),
            ),
        ]
    )
    gp_R.fit(X_train, yR_train)

    # --- Volume Ridge (fast + robust) ---
    ridge_V = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=1e-6, random_state=0)),
        ]
    )
    ridge_V.fit(X_train, yV_train)

    # Metrics
    yR_hat = gp_R.predict(X_val)
    yV_hat = ridge_V.predict(X_val)

    rmse_R = float(np.sqrt(mean_squared_error(yR_val, yR_hat)))
    r2_R = float(r2_score(yR_val, yR_hat))

    rmse_V = float(np.sqrt(mean_squared_error(yV_val, yV_hat)))
    r2_V = float(r2_score(yV_val, yV_hat))

    metrics = {
        "R_total": {"rmse": rmse_R, "r2": r2_R},
        "V": {"rmse": rmse_V, "r2": r2_V},
    }

    return (
        X_train,
        X_val,
        gp_R,
        metrics,
        ridge_V,
        rmse_R,
        rmse_V,
        r2_R,
        r2_V,
        yR_hat,
        yR_val,
        yV_hat,
        yV_val,
    )


@app.cell(hide_code=True)
def _(metrics, mo):
    mo.md(
        f"""
### Validation metrics (20% holdout)

- **R_total**: RMSE = `{metrics['R_total']['rmse']:.6g}`, R² = `{metrics['R_total']['r2']:.4f}`
- **V**: RMSE = `{metrics['V']['rmse']:.6g}`, R² = `{metrics['V']['r2']:.4f}`

If R_total error is too high near the optimum region, increase `n_samples`, or switch to a more expressive surrogate (e.g., NN) **only after** you have enough data.
"""
    )
    return


@app.cell(hide_code=True)
def _(go, mo, yR_hat, yR_val):
    figR = go.Figure()
    figR.add_scatter(x=yR_val, y=yR_hat, mode="markers", name="val")
    figR.update_layout(
        title="Parity plot: Resistance surrogate",
        xaxis_title="Truth R (val)",
        yaxis_title="Predicted R̂ (val)",
        height=450,
    )
    mo.ui.plotly(figR)
    return (figR,)


@app.cell(hide_code=True)
def _(go, mo, yV_hat, yV_val):
    figV = go.Figure()
    figV.add_scatter(x=yV_val, y=yV_hat, mode="markers", name="val")
    figV.update_layout(
        title="Parity plot: Volume surrogate",
        xaxis_title="Truth V (val)",
        yaxis_title="Predicted V̂ (val)",
        height=450,
    )
    mo.ui.plotly(figV)
    return (figV,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## 3) Use the surrogate (for your optimizers)

Call `predict_RV(h)` with `h.shape == (29,)` in \[0,1\].

In your optimization script, you can import this file's saved models (see next cell),
or just copy the `predict_RV` function into the optimizer code.
"""
    )
    return


@app.cell
def _(gp_R, np, ridge_V):
    def predict_RV(h: np.ndarray):
        """Return (R_hat, V_hat) for a single design vector h in [0,1]^29."""
        h = np.asarray(h, dtype=float).reshape(1, -1)
        if h.shape[1] != 29:
            raise ValueError(f"Expected 29 design variables, got {h.shape[1]}")
        h = np.clip(h, 0.0, 1.0)
        R_hat = float(gp_R.predict(h)[0])
        V_hat = float(ridge_V.predict(h)[0])
        return R_hat, V_hat

    return (predict_RV,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## 4) Save models to disk (optional)

This writes:
- `surrogate_R_gp.joblib`
- `surrogate_V_ridge.joblib`
"""
    )
    return


@app.cell
def _(Path, gp_R, joblib, mo, ridge_V):
    save_btn = mo.ui.button(label="Save surrogates to .joblib")
    out = mo.md("")
    if save_btn.value:
        joblib.dump(gp_R, Path("surrogate_R_gp.joblib"))
        joblib.dump(ridge_V, Path("surrogate_V_ridge.joblib"))
        out = mo.md("Saved `surrogate_R_gp.joblib` and `surrogate_V_ridge.joblib`.")
    mo.hstack([save_btn, out])
    return out, save_btn


if __name__ == "__main__":
    app.run()
