import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
# Two-surrogate modeling (29D hull params → R and V)

We train:
- **R surrogate**:  \hat{R}(h)
- **V surrogate**:  \hat{V}(h)

This is the correct setup when **volume is a constraint**:
\[
\min_h \; \hat{R}(h)\quad \text{s.t.}\quad \hat{V}(h)\ge V_{\min}
\]
"""
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import os

    import plotly.graph_objects as go
    import smt.surrogate_models as sm
    from timeit import default_timer as timer

    # Your modules
    from truth_resistance_model import truth_resistance_and_volume
    from Data_Training_set import generate_dataset, save_dataset_csv

    return (
        generate_dataset,
        go,
        mo,
        np,
        os,
        pd,
        save_dataset_csv,
        sm,
        timer,
        truth_resistance_and_volume,
    )


@app.cell(hide_code=True)
def _(mo):
    n_samples = mo.ui.slider(
        label="Number of samples",
        start=50,
        stop=2000,
        step=50,
        value=500,
        show_value=True,
        full_width=True,
        debounce=True,
    )

    test_frac = mo.ui.slider(
        label="Test fraction",
        start=0.1,
        stop=0.5,
        step=0.05,
        value=0.2,
        show_value=True,
        full_width=True,
        debounce=True,
    )

    seed = mo.ui.number(label="Random seed", value=1, step=1)

    speed_U = mo.ui.number(label="Speed U (m/s)", value=2.0, step=0.1)
    length_L = mo.ui.number(label="Length L (m)", value=1.0, step=0.1)
    beam_Bmax = mo.ui.number(label="Beam max B_max (m)", value=0.75, step=0.05)
    draft_Tmax = mo.ui.number(label="Draft max T_max (m)", value=0.50, step=0.05)

    dataset_path = mo.ui.text(label="CSV path", value="hull_dataset.csv")

    dataset_mode = mo.ui.dropdown(
        label="Dataset source",
        options=["Generate now (truth model)", "Load from CSV (if exists)"],
        value="Load from CSV (if exists)",
        allow_select_none=False,
    )

    build_data = mo.ui.run_button(label="Build / Load dataset")

    mo.vstack(
        [
            mo.md("### Dataset"),
            mo.hstack([dataset_mode, dataset_path]),
            mo.hstack([n_samples, test_frac]),
            mo.hstack([seed, build_data]),
            mo.md("#### Truth model settings (used only when generating)"),
            mo.hstack([speed_U, length_L, beam_Bmax, draft_Tmax]),
        ]
    )
    return (
        beam_Bmax,
        build_data,
        dataset_mode,
        dataset_path,
        draft_Tmax,
        length_L,
        n_samples,
        seed,
        speed_U,
        test_frac,
    )


@app.cell(hide_code=True)
def _(mo):
    model_type = mo.ui.dropdown(
        label="Surrogate type (SMT) for BOTH R and V",
        options=[
            "KRG (Gaussian Process / Kriging)",
            "RBF (Radial Basis Function)",
            "IDW (Inverse Distance Weighting)",
            "LS (Linear Regression)",
            "QP (Quadratic Regression)",
        ],
        value="KRG (Gaussian Process / Kriging)",
        allow_select_none=False,
    )

    # Model knobs (used depending on choice)
    rbf_d0 = mo.ui.number(label="RBF d0", value=1.0, step=0.1)
    idw_p = mo.ui.number(label="IDW p", value=2.5, step=0.1)

    train_models = mo.ui.run_button(label="Train BOTH surrogates")

    mo.vstack(
        [
            mo.md("### Surrogate"),
            mo.hstack([model_type, train_models]),
            mo.accordion(
                {
                    "**Hyperparameters**": mo.vstack(
                        [mo.hstack([rbf_d0, idw_p])]
                    )
                },
                lazy=True,
            ),
        ]
    )
    return idw_p, model_type, rbf_d0, train_models


@app.cell(hide_code=True)
def _(
    beam_Bmax,
    build_data,
    dataset_mode,
    dataset_path,
    draft_Tmax,
    generate_dataset,
    length_L,
    mo,
    n_samples,
    np,
    os,
    save_dataset_csv,
    seed,
    speed_U,
    test_frac,
    truth_resistance_and_volume,
):
    build_data.value  # trigger

    def _load_csv(path: str):
        import pandas as pd

        df = pd.read_csv(path)
        h_cols = [c for c in df.columns if c.startswith("h")]
        H = df[h_cols].to_numpy(dtype=float)
        R = df["R"].to_numpy(dtype=float)
        V = df["V"].to_numpy(dtype=float)
        return H, R, V

    with mo.status.progress_bar(title="Preparing dataset", total=1) as bar:
        path = dataset_path.value.strip()

        if dataset_mode.value.startswith("Load") and os.path.exists(path):
            H, R, V = _load_csv(path)
            msg = f"Loaded {H.shape[0]} samples from {path}"
        else:
            H, R, V, _meta = generate_dataset(
                n_samples=int(n_samples.value),
                truth_fn=lambda h: truth_resistance_and_volume(
                    h,
                    U=float(speed_U.value),
                    L=float(length_L.value),
                    B_max=float(beam_Bmax.value),
                    T_max=float(draft_Tmax.value),
                ),
                seed=int(seed.value),
                n_dim=29,
            )
            save_dataset_csv(path, H, R, V)
            msg = f"Generated {H.shape[0]} samples and saved to {path}"

        n = H.shape[0]
        rng = np.random.default_rng(int(seed.value))
        idx = rng.permutation(n)

        n_test = max(1, int(round(float(test_frac.value) * n)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        H_train = H[train_idx]
        H_test = H[test_idx]

        R_train, R_test = R[train_idx], R[test_idx]
        V_train, V_test = V[train_idx], V[test_idx]

        bar.update(title=msg)

    return H_test, H_train, R_test, R_train, V_test, V_train, msg


@app.cell(hide_code=True)
def _(H_train, mo, msg):
    mo.md(
        f"""
**Dataset status:** {msg}

- Train samples: **{H_train.shape[0]}**
- Test samples: **(set by slider)**
- Dimensionality: **{H_train.shape[1]}** (h1..h29)
"""
    )
    return


@app.cell(hide_code=True)
def _(sm, model_type, rbf_d0, idw_p):
    def make_model():
        mt = model_type.value
        if mt.startswith("KRG"):
            return sm.KRG()
        if mt.startswith("RBF"):
            # robust: keep it simple
            return sm.RBF(d0=float(rbf_d0.value), poly_degree=-1, reg=1e-10)
        if mt.startswith("IDW"):
            return sm.IDW(p=float(idw_p.value))
        if mt.startswith("LS"):
            return sm.LS()
        if mt.startswith("QP"):
            return sm.QP()
        return sm.KRG()

    return (make_model,)


@app.cell(hide_code=True)
def _(
    H_test,
    H_train,
    R_test,
    R_train,
    V_test,
    V_train,
    make_model,
    mo,
    np,
    timer,
    train_models,
):
    train_models.value  # trigger

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def r2(y, yhat):
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-16
        return float(1.0 - ss_res / ss_tot)

    def rel_l2(y, yhat):
        return float(np.linalg.norm(y - yhat) / (np.linalg.norm(y) + 1e-16))

    with mo.status.progress_bar(title="Training both surrogates", total=2) as bar:
        # ---- R model ----
        R_model = make_model()
        R_model.set_training_values(H_train, R_train.reshape(-1, 1))
        t0 = timer()
        R_model.train()
        t1 = timer()
        bar.update(advance=1)

        # ---- V model ----
        V_model = make_model()
        V_model.set_training_values(H_train, V_train.reshape(-1, 1))
        t2 = timer()
        V_model.train()
        t3 = timer()
        bar.update(advance=1)

        bar.update(
            title=f"Done. R train: {(t1-t0):.2e}s | V train: {(t3-t2):.2e}s"
        )

    # Predictions
    R_train_hat = R_model.predict_values(H_train).reshape(-1)
    R_test_hat = R_model.predict_values(H_test).reshape(-1)

    V_train_hat = V_model.predict_values(H_train).reshape(-1)
    V_test_hat = V_model.predict_values(H_test).reshape(-1)

    metrics = {
        "R": {
            "train_rmse": rmse(R_train, R_train_hat),
            "test_rmse": rmse(R_test, R_test_hat),
            "train_r2": r2(R_train, R_train_hat),
            "test_r2": r2(R_test, R_test_hat),
            "train_rel": rel_l2(R_train, R_train_hat),
            "test_rel": rel_l2(R_test, V_test_hat if False else R_test_hat),
        },
        "V": {
            "train_rmse": rmse(V_train, V_train_hat),
            "test_rmse": rmse(V_test, V_test_hat),
            "train_r2": r2(V_train, V_train_hat),
            "test_r2": r2(V_test, V_test_hat),
            "train_rel": rel_l2(V_train, V_train_hat),
            "test_rel": rel_l2(V_test, V_test_hat),
        },
    }

    return (
        R_model,
        R_test_hat,
        R_train_hat,
        V_model,
        V_test_hat,
        V_train_hat,
        metrics,
    )


@app.cell(hide_code=True)
def _(metrics, mo, model_type):
    def table_for(name):
        m = metrics[name]
        return f"""
| Split | RMSE | Relative L2 | R² |
|---|---:|---:|---:|
| Train | {m["train_rmse"]:.3e} | {m["train_rel"]:.3e} | {m["train_r2"]:.4f} |
| Test  | {m["test_rmse"]:.3e} | {m["test_rel"]:.3e} | {m["test_r2"]:.4f} |
"""

    mo.md(
        f"""
## Accuracy report

**Model type:** `{model_type.value}`

### Resistance surrogate (R)
{table_for("R")}

### Volume surrogate (V)
{table_for("V")}
"""
    )
    return


@app.cell(hide_code=True)
def _(go, mo, np, R_test, R_test_hat, R_train, R_train_hat, V_test, V_test_hat, V_train, V_train_hat):
    def parity_fig(y, yhat, title, xlab, ylab):
        lo = float(min(y.min(), yhat.min()))
        hi = float(max(y.max(), yhat.max()))
        pad = 0.02 * (hi - lo + 1e-16)
        lo -= pad
        hi += pad

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y, y=yhat, mode="markers", name="points"))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y=x"))
        fig.update_layout(
            title=title,
            xaxis_title=xlab,
            yaxis_title=ylab,
            height=360,
            width=420,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        )
        return fig

    r_train_fig = parity_fig(R_train, R_train_hat, "R parity (train)", "True R", "Predicted R")
    r_test_fig = parity_fig(R_test, R_test_hat, "R parity (test)", "True R", "Predicted R")

    v_train_fig = parity_fig(V_train, V_train_hat, "V parity (train)", "True V", "Predicted V")
    v_test_fig = parity_fig(V_test, V_test_hat, "V parity (test)", "True V", "Predicted V")

    mo.accordion(
        {
            "**Parity plots (R)**": mo.hstack([r_train_fig, r_test_fig]),
            "**Parity plots (V)**": mo.hstack([v_train_fig, v_test_fig]),
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## How you’ll use this in the constrained optimization notebook

Once trained, you have two callable predictors:

- `R_model.predict_values(h.reshape(1,-1))[0,0]`
- `V_model.predict_values(h.reshape(1,-1))[0,0]`

Typical constraint form (optimizer expects `g(h) <= 0`):
\[
g(h) = V_{\min} - \hat{V}(h) \le 0
\]
"""
    )
    return


if __name__ == "__main__":
    app.run()