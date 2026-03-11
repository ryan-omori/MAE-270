import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
# **Hull surrogate training + CSDL constrained optimization (29D)**
Dataset CSV must contain columns: `h1..h29`, `R`, `V`.

Workflow:
1) Load CSV  
2) Train surrogate:
   - MLP for resistance (NumPy)
   - Linear regression for volume
3) Freeze weights -> `surrogate_weights.npz`
4) CSDL-native constrained optimization:
   - minimize `R_hat(h)`
   - s.t. volume within ±γ of `Vref`
""")
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import plotly.graph_objects as go
    import plotly.colors as pc
    import numpy as np
    import pandas as pd
    import csdl_alpha as csdl
    import modopt as opt
    import modopt.postprocessing as post

    return csdl, go, mo, np, opt, pc, pd, post


@app.cell(hide_code=True)
def _(mo):
    dataset_path = mo.ui.text(value="hull_dataset.csv", label="Dataset CSV path")
    weights_path = mo.ui.text(value="surrogate_weights.npz", label="Save weights (.npz)")

    seed = mo.ui.number(value=1, label="Seed", step=1)
    val_frac = mo.ui.slider(start=0.1, stop=0.4, value=0.2, step=0.05, label="Validation fraction")

    # MLP shape
    width1 = mo.ui.number(value=16, label="MLP hidden width 1", step=1)
    width2 = mo.ui.number(value=16, label="MLP hidden width 2", step=1)

    epochs = mo.ui.number(value=1500, label="Epochs", step=100)
    lr = mo.ui.text(value="1e-3", label="Adam lr")
    batch_size = mo.ui.number(value=64, label="Batch size", step=16)
    l2 = mo.ui.text(value="1e-5", label="L2 (weights)")

    train_btn = mo.ui.button(label="Train surrogate + save weights")

    mo.accordion({
        "**Training inputs**": mo.vstack([
            dataset_path,
            weights_path,
            mo.hstack([seed, val_frac]),
            mo.hstack([width1, width2]),
            mo.hstack([epochs, batch_size]),
            mo.hstack([lr, l2]),
            train_btn,
        ])
    })
    return (
        batch_size,
        dataset_path,
        epochs,
        l2,
        lr,
        seed,
        train_btn,
        val_frac,
        weights_path,
        width1,
        width2,
    )


@app.cell(hide_code=True)
def _(np, pd, dataset_path, seed, train_btn, val_frac):
    # Load/split only when training is clicked
    _data = None
    if train_btn.value:
        df = pd.read_csv(dataset_path.value)
        hcols = [f"h{i}" for i in range(1, 30)]
        for c in hcols + ["R", "V"]:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in CSV.")

        H = df[hcols].to_numpy(float)
        R = df["R"].to_numpy(float).reshape(-1, 1)
        V = df["V"].to_numpy(float).reshape(-1, 1)

        rng = np.random.default_rng(int(seed.value))
        perm = np.arange(H.shape[0])
        rng.shuffle(perm)

        n_val = int(round(float(val_frac.value) * H.shape[0]))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        _data = {
            "H_tr": H[tr_idx], "R_tr": R[tr_idx], "V_tr": V[tr_idx],
            "H_va": H[val_idx], "R_va": R[val_idx], "V_va": V[val_idx],
            "n_total": H.shape[0],
        }
    return (_data,)


@app.cell(hide_code=True)
def _(mo, _data):
    if _data is None:
        mo.info("Click **Train surrogate + save weights** to load the CSV and train.")
    else:
        mo.success(f"Loaded {_data['n_total']} samples. Train={_data['H_tr'].shape[0]}, Val={_data['H_va'].shape[0]}")
    return


@app.cell(hide_code=True)
def _(np):
    # ---- Training utilities (NumPy) ----
    def standardize(y, eps=1e-12):
        mu = float(np.mean(y))
        sig = float(np.std(y) + eps)
        return (y - mu) / sig, mu, sig

    def linreg_fit(X, y, ridge=1e-10):
        n = X.shape[0]
        Xb = np.hstack([X, np.ones((n, 1))])
        I = np.eye(Xb.shape[1])
        I[-1, -1] = 0.0  # don't regularize bias
        wfull = np.linalg.solve(Xb.T @ Xb + ridge * I, Xb.T @ y)
        w = wfull[:-1].reshape(-1)
        b = float(wfull[-1])
        return w, b

    def tanh(x):
        return np.tanh(x)

    def tanh_grad(z):
        t = np.tanh(z)
        return 1.0 - t * t

    def init_mlp(n_in, n_h1, n_h2, seed=0):
        rng = np.random.default_rng(seed)
        W0 = rng.normal(0, 1.0 / np.sqrt(n_in), size=(n_in, n_h1))
        b0 = np.zeros((n_h1,))
        W1 = rng.normal(0, 1.0 / np.sqrt(n_h1), size=(n_h1, n_h2))
        b1 = np.zeros((n_h2,))
        W2 = rng.normal(0, 1.0 / np.sqrt(n_h2), size=(n_h2, 1))
        b2 = np.zeros((1,))
        return {"W0": W0, "b0": b0, "W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def mlp_forward(X, p):
        z0 = X @ p["W0"] + p["b0"]
        a0 = tanh(z0)
        z1 = a0 @ p["W1"] + p["b1"]
        a1 = tanh(z1)
        z2 = a1 @ p["W2"] + p["b2"]  # linear output
        cache = (X, z0, a0, z1, a1)
        return z2, cache

    def loss_grads(X, y, p, l2=0.0):
        yhat, cache = mlp_forward(X, p)
        N = X.shape[0]
        err = yhat - y
        loss = float(np.mean(err * err))
        loss += float(l2 * (np.sum(p["W0"]**2) + np.sum(p["W1"]**2) + np.sum(p["W2"]**2)))

        X0, z0, a0, z1, a1 = cache
        dY = (2.0 / N) * err

        dW2 = a1.T @ dY + 2.0 * l2 * p["W2"]
        db2 = np.sum(dY, axis=0)

        da1 = dY @ p["W2"].T
        dz1 = da1 * tanh_grad(z1)
        dW1 = a0.T @ dz1 + 2.0 * l2 * p["W1"]
        db1 = np.sum(dz1, axis=0)

        da0 = dz1 @ p["W1"].T
        dz0 = da0 * tanh_grad(z0)
        dW0 = X0.T @ dz0 + 2.0 * l2 * p["W0"]
        db0 = np.sum(dz0, axis=0)

        grads = {"W0": dW0, "b0": db0, "W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return loss, grads

    def adam_init(p):
        m = {k: np.zeros_like(v) for k, v in p.items()}
        v = {k: np.zeros_like(v) for k, v in p.items()}
        return m, v

    def adam_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        for k in p:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * (g[k] ** 2)
            mhat = m[k] / (1 - b1**t)
            vhat = v[k] / (1 - b2**t)
            p[k] = p[k] - lr * mhat / (np.sqrt(vhat) + eps)
        return p, m, v

    return adam_init, adam_step, init_mlp, linreg_fit, loss_grads, mlp_forward, standardize, tanh


@app.cell(hide_code=True)
def _(
    adam_init,
    adam_step,
    init_mlp,
    linreg_fit,
    loss_grads,
    mlp_forward,
    np,
    standardize,
    tanh,
    time,
    _data,
    batch_size,
    epochs,
    l2,
    lr,
    seed,
    weights_path,
    width1,
    width2,
):
    _trained = None

    if _data is not None:
        H_tr = _data["H_tr"]; R_tr = _data["R_tr"]; V_tr = _data["V_tr"]
        H_va = _data["H_va"]; R_va = _data["R_va"]; V_va = _data["V_va"]

        # Standardize R for training
        R_tr_std, R_mean, R_std = standardize(R_tr)
        R_va_std = (R_va - R_mean) / R_std

        # Volume linear regression
        V_w, V_b = linreg_fit(H_tr, V_tr, ridge=1e-10)

        # MLP
        p = init_mlp(29, int(width1.value), int(width2.value), seed=int(seed.value))
        m, v = adam_init(p)

        lr_f = float(lr.value)
        l2_f = float(l2.value)
        bs = int(batch_size.value)

        rng = np.random.default_rng(int(seed.value))
        n = H_tr.shape[0]
        train_loss = []
        val_loss = []

        t0 = time.time()
        for ep in range(1, int(epochs.value) + 1):
            perm = np.arange(n)
            rng.shuffle(perm)

            for start in range(0, n, bs):
                j = perm[start:start+bs]
                Xb = H_tr[j]
                yb = R_tr_std[j]
                L, grads = loss_grads(Xb, yb, p, l2=l2_f)
                p, m, v = adam_step(p, grads, m, v, t=ep, lr=lr_f)

            Ltr, _ = loss_grads(H_tr, R_tr_std, p, l2=0.0)
            Lva, _ = loss_grads(H_va, R_va_std, p, l2=0.0)
            train_loss.append(Ltr)
            val_loss.append(Lva)

        seconds = time.time() - t0

        # Save weights to npz for CSDL
        np.savez(
            weights_path.value,
            W0=p["W0"], b0=p["b0"],
            W1=p["W1"], b1=p["b1"],
            W2=p["W2"], b2=p["b2"],
            V_w=V_w, V_b=V_b,
            R_mean=R_mean, R_std=R_std,
        )

        _trained = {
            "p": p,
            "V_w": V_w,
            "V_b": V_b,
            "R_mean": R_mean,
            "R_std": R_std,
            "train_loss": np.array(train_loss),
            "val_loss": np.array(val_loss),
            "seconds": seconds,
        }

    return (_trained,)


@app.cell(hide_code=True)
def _(mo, _trained, weights_path):
    if _trained is None:
        mo.info("No training yet.")
    else:
        mo.success(f"Training done in {_trained['seconds']:.2f}s. Saved: {weights_path.value}")
    return


@app.cell(hide_code=True)
def _(go, mlp_forward, np, _data, _trained):
    _diag_figs = None
    if _trained is not None and _data is not None:
        H_va = _data["H_va"]
        R_va = _data["R_va"].reshape(-1)
        V_va = _data["V_va"].reshape(-1)

        R_hat_std, _ = mlp_forward(H_va, _trained["p"])
        R_hat = (R_hat_std * _trained["R_std"] + _trained["R_mean"]).reshape(-1)

        V_hat = (H_va @ _trained["V_w"].reshape(-1, 1) + _trained["V_b"]).reshape(-1)

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=_trained["train_loss"], mode="lines", name="train"))
        fig_loss.add_trace(go.Scatter(y=_trained["val_loss"], mode="lines", name="val"))
        fig_loss.update_layout(title="Training loss (MSE on standardized R)", xaxis_title="epoch", yaxis_title="MSE")

        fig_R = go.Figure()
        fig_R.add_trace(go.Scatter(x=R_va, y=R_hat, mode="markers", name="val"))
        lo = float(min(R_va.min(), R_hat.min())); hi = float(max(R_va.max(), R_hat.max()))
        fig_R.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y=x"))
        fig_R.update_layout(title="Resistance parity (val)", xaxis_title="True R", yaxis_title="Pred R")

        fig_V = go.Figure()
        fig_V.add_trace(go.Scatter(x=V_va, y=V_hat, mode="markers", name="val"))
        lo = float(min(V_va.min(), V_hat.min())); hi = float(max(V_va.max(), V_hat.max()))
        fig_V.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y=x"))
        fig_V.update_layout(title="Volume parity (val)", xaxis_title="True V", yaxis_title="Pred V")

        _diag_figs = (fig_loss, fig_R, fig_V)

    return (_diag_figs,)


@app.cell(hide_code=True)
def _(mo, _diag_figs):
    mo.md("## Diagnostics")
    if _diag_figs is None:
        mo.info("Train first to see plots.")
    else:
        fig_loss, fig_R, fig_V = _diag_figs
        mo.vstack([fig_loss, mo.hstack([fig_R, fig_V])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Constrained optimization settings")
    return


@app.cell(hide_code=True)
def _(mo):
    gamma = mo.ui.text(value="0.02", label="γ (volume tolerance)")
    Vref_mode = mo.ui.dropdown(options=["mean(V_train)", "user_defined"], value="mean(V_train)", label="Vref source")
    Vref_user = mo.ui.text(value="0.10", label="Vref if user_defined (m^3)")

    tol = mo.ui.text(value="1e-10", label="Optimizer tolerance")
    maxiter = mo.ui.slider(start=50, stop=1000, step=50, value=300, label="Max iter (grad-based)")
    gf_maxiter = mo.ui.slider(start=200, stop=5000, step=200, value=1200, label="Max iter (grad-free)")

    run_btn = mo.ui.button(label="Run optimizers")

    mo.accordion({
        "**Optimization inputs**": mo.vstack([
            mo.hstack([gamma, Vref_mode, Vref_user]),
            mo.hstack([tol, maxiter, gf_maxiter]),
            run_btn,
        ])
    })
    return Vref_mode, Vref_user, gf_maxiter, gamma, maxiter, run_btn, tol


@app.cell(hide_code=True)
def _(np, _data, Vref_mode, Vref_user):
    _Vref = None
    if _data is not None:
        if Vref_mode.value == "mean(V_train)":
            _Vref = float(np.mean(_data["V_tr"]))
        else:
            _Vref = float(Vref_user.value)
    return (_Vref,)


@app.cell(hide_code=True)
def _(csdl, np, opt, run_btn, tol, maxiter, gf_maxiter, gamma, _Vref, weights_path, _trained):
    # Build CSDL optimization problem only when trained + run clicked
    _prob = None
    _sim = None
    _h_var = None

    if _trained is not None and run_btn.value and _Vref is not None:
        _rec = csdl.Recorder(inline=True)
        _rec.start()

        h = csdl.Variable(value=np.full(29, 0.5), name="h")
        h.set_as_design_variable(lower=0.0, upper=1.0)

        data = np.load(weights_path.value, allow_pickle=True)

        # CSDL MLP forward (tanh activations)
        a = h
        for k in [0, 1, 2]:
            W = data[f"W{k}"]
            b = data[f"b{k}"]
            z = csdl.matmul(a, W) + b
            if k < 2:
                a = csdl.tanh(z)
            else:
                a = z  # output

        R_hat_std = csdl.reshape(a, (1,))
        R_hat = R_hat_std * float(data["R_std"]) + float(data["R_mean"])
        R_hat = csdl.reshape(R_hat, ())

        V_w = data["V_w"].reshape((29,))
        V_b = float(data["V_b"])
        V_hat = V_b + csdl.sum(h * V_w)
        V_hat = csdl.reshape(V_hat, ())

        gam = float(gamma.value)
        Vref = float(_Vref)

        g1 = (1.0 - gam) * Vref - V_hat
        g2 = V_hat - (1.0 + gam) * Vref
        g1.set_as_constraint(upper=0.0)
        g2.set_as_constraint(upper=0.0)

        R_hat.set_as_objective()

        _rec.stop()
        _sim = csdl.experimental.PySimulator(_rec)
        _prob = opt.CSDLAlphaProblem(problem_name="HullSurrogate", simulator=_sim)
        _h_var = h

    return _h_var, _prob, _sim


@app.cell(hide_code=True)
def _(opt, pc, _prob, tol, maxiter, gf_maxiter):
    optimizer_color_map = pc.qualitative.Plotly + pc.qualitative.Plotly

    def get_available_optimizers():
        if _prob is None:
            return {}

        opt_tol = float(tol.value)
        mi = int(maxiter.value)
        gmi = int(gf_maxiter.value)

        opts = {
            # Gradient-based constrained
            "SQP": opt.SQP(_prob, recording=True, maxiter=mi, opt_tol=opt_tol),
            "pySLSQP": opt.PySLSQP(_prob, recording=True, solver_options={"maxiter": mi, "acc": opt_tol}),
            # Gradient-free constrained
            "COBYQA": opt.COBYQA(_prob, recording=True, solver_options={"maxiter": gmi, "feasibility_tol": opt_tol}),
            "PSO": opt.PSO(_prob, recording=True, maxiter=gmi, tol=opt_tol, readable_outputs=["x"]),
            "NelderMeadSimplex": opt.NelderMeadSimplex(_prob, recording=True, maxiter=gmi, tol=opt_tol),
        }
        return opts

    return get_available_optimizers, optimizer_color_map


@app.cell(hide_code=True)
def _(get_available_optimizers, mo):
    optimizer_dropdown = mo.ui.multiselect(
        options=list(get_available_optimizers().keys()),
        value=["pySLSQP"] if "pySLSQP" in get_available_optimizers() else [],
        label="Optimizers to run",
        max_selections=10,
    )
    return (optimizer_dropdown,)


@app.cell(hide_code=True)
def _(mo, optimizer_dropdown):
    mo.accordion({"**Choose optimizers**": mo.vstack([optimizer_dropdown])})
    return


@app.cell(hide_code=True)
def _(get_available_optimizers, mo, optimizer_dropdown, run_btn):
    _optimizers = {}
    if run_btn.value:
        avail = get_available_optimizers()
        for name in optimizer_dropdown.value:
            _optimizers[name] = avail[name]

        with mo.status.progress_bar(total=len(_optimizers)) as bar:
            for name in _optimizers:
                bar.update(title=f"Running {name}", subtitle="")
                _optimizers[name].solve()
            bar.update(increment=0, title="Finished running optimizers", subtitle="")

    return (_optimizers,)


@app.cell(hide_code=True)
def _(mo, _optimizers):
    if not _optimizers:
        mo.info("Click **Run optimizers** (and select optimizers) to compute results.")
    else:
        mo.success(f"Ran {len(_optimizers)} optimizers.")
    return


@app.cell(hide_code=True)
def _(np, opt, post, _optimizers):
    # Load results from record.hdf5 like your template
    _results = {}
    _largest = 0

    if _optimizers:
        quantity_names = ["x", "opt", "obj", "optimality", "feasibility"]

        for name, optimizer in _optimizers.items():
            out_dir = optimizer.results["out_dir"]
            record_path = out_dir + "/record.hdf5"

            _, opt_vars, _, _ = post.print_record_contents(record_path, suppress_print=True)

            vars_to_load = [q for q in quantity_names if q in opt_vars]
            loaded = post.load_variables(record_path, vars_to_load)

            for q in loaded:
                loaded[q] = np.array(loaded[q])

            _results[name] = loaded
            if "x" in loaded:
                _largest = max(_largest, len(loaded["x"]))

    return _largest, _results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"### **Results**")
    return


@app.cell(hide_code=True)
def _(mo, _largest):
    if _largest <= 1:
        iteration_slider = None
    else:
        iteration_slider = mo.ui.slider(
            label="Iteration",
            start=0, stop=_largest - 1, step=1,
            value=_largest - 1,
            show_value=True,
            full_width=True,
            debounce=True,
        )
    return (iteration_slider,)


@app.cell(hide_code=True)
def _(go, iteration_slider, _largest, optimizer_color_map, _optimizers, _results):
    convergence_figs = {}

    if iteration_slider is None or not _optimizers:
        return (convergence_figs,)

    it = iteration_slider.value

    for key, label in [
        ("optimality", "Optimality"),
        ("feasibility", "Feasibility"),
        ("obj", "Objective"),
    ]:
        fig = go.Figure()
        for i, name in enumerate(_optimizers.keys()):
            if name not in _results:
                continue
            if key not in _results[name]:
                continue
            y = _results[name][key][: it + 1].reshape(-1)
            fig.add_trace(go.Scatter(
                x=list(range(len(y))),
                y=y,
                mode="lines+markers",
                line=dict(color=optimizer_color_map[i], width=2),
                name=name,
            ))
        fig.update_layout(
            title=label,
            xaxis_title="Iteration",
            yaxis_title=label,
            margin=dict(l=0, r=0, t=40, b=0),
            height=300,
        )
        convergence_figs[key] = fig

    return (convergence_figs,)


@app.cell(hide_code=True)
def _(convergence_figs, mo, iteration_slider):
    if iteration_slider is None or not convergence_figs:
        mo.info("Run optimizers to see convergence plots.")
    else:
        mo.accordion(
            {"**Convergence histories**": mo.vstack([
                convergence_figs.get("obj"),
                convergence_figs.get("optimality"),
                convergence_figs.get("feasibility"),
            ])},
            lazy=True,
        )
    return
if __name__ == "__main__":
    app.run()