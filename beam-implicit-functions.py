import marimo

__generated_with = "0.10.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# **Understanding implicit functions through the beam problem**""")
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import plotly.graph_objects as go
    import plotly.colors as pc
    import numpy as np
    import pandas as pd
    import os
    from collections import namedtuple

    import jax
    import jax.numpy as jnp 

    import modopt as opt
    import modopt.postprocessing
    return go, jax, jnp, mo, modopt, namedtuple, np, opt, os, pc, pd


@app.cell(hide_code=True)
def _(mo):
    # Model parameters inputs

    n_input = mo.ui.number(start=1, stop=300, value=20, label=r"Number of elements, $n$")
    b_input = mo.ui.text(value="0.1", placeholder="0.1", label=r"Width, $b$ [m]")
    l_input = mo.ui.text(value="1.0", placeholder="1.0", label=r"Beam length, $l$ [m]")
    m0_input = mo.ui.text(value="0.1", placeholder="0.1", label=r"Target mass, $m_0$ [kg]")
    rho_input = mo.ui.text(value="10.", placeholder="10.", label=r"Density, $\rho$ [kg/m^3]")
    E_input = mo.ui.text(value="1.0e7", placeholder="1.0e7", label=r"Young's modulus, $E$ [Pa]")
    a0_input = mo.ui.text(value="10.", placeholder="10.", label=r"Tip load, $a_0$ [N]")
    # obj_scaling_input = mo.ui.text(value="1e6", placeholder="1e6", label=r"Objective scaling")
    # con_scaling_input = mo.ui.text(value="1.0", placeholder="1.0", label=r"Constraint scaling")
    obj_scaling_input = mo.ui.slider(
        label="Obj. scaling (exp. of 10)",
        start=-16, stop=16, step=1, value=6,
        show_value=True,
        # full_width=True,
    )
    con_scaling_input = mo.ui.slider(
        label="Constr. scaling (exp. of 10)",
        start=-16, stop=16, step=1, value=0,
        show_value=True,
        # full_width=True,
    )

    parameters_stack = mo.vstack([
        # mo.md(r"""Parameters"""),
        mo.hstack([
            n_input,
            b_input,
            l_input,
        ]),
        mo.hstack([
            m0_input,
            rho_input,
            E_input,
        ]),
        mo.hstack([
            a0_input,
            obj_scaling_input,
            con_scaling_input,
        ]),
    ])

    mo.accordion({r"**Model parameters**": parameters_stack})
    return (
        E_input,
        a0_input,
        b_input,
        con_scaling_input,
        l_input,
        m0_input,
        n_input,
        obj_scaling_input,
        parameters_stack,
        rho_input,
    )


@app.cell(hide_code=True)
def _(
    E_input,
    a0_input,
    b_input,
    con_scaling_input,
    jax,
    jnp,
    l_input,
    m0_input,
    n_input,
    np,
    obj_scaling_input,
    opt,
    rho_input,
):
    # Model implementation

    _n = int(n_input.value)
    _b = float(b_input.value)
    _l = float(l_input.value)
    _m0 = float(m0_input.value)
    _rho = float(rho_input.value)
    _E = float(E_input.value)
    _a0 = float(a0_input.value)
    _obj_scaling = 10 ** float(obj_scaling_input.value)
    _con_scaling = 10 ** float(con_scaling_input.value)

    _a = np.zeros((2 * _n))
    _a[-2] = _a0 / _E / _b / _l

    def _compute_stiffness_matrix(x, dtype):
        _Ki_wox3 = 1. / 12. * np.array([
            [12 * _n ** 3, 6 * _n ** 2, -12 * _n ** 3, 6 * _n ** 2],
            [6 * _n ** 2, 4 * _n, -6 * _n ** 2, 2 * _n],
            [-12 * _n ** 3, -6 * _n ** 2, 12 * _n ** 3, -6 * _n ** 2],
            [6 * _n ** 2, 2 * _n, -6 * _n ** 2, 4 * _n],
        ])
        _K = jnp.zeros((_n * 2 + 2, _n * 2 + 2), dtype=dtype)
        for i in range(_n):
            _K = _K.at[2*i:2*i+4, 2*i:2*i+4].add(_Ki_wox3 * x[i] ** 3)

        return _K[2:, 2:]

    def compute_state(x, dtype=float):
        _K = _compute_stiffness_matrix(x, dtype)
        _y = jnp.linalg.solve(_K, _a)
        return _y

    def compute_residual(x, y):
        _K = _compute_stiffness_matrix(x, float)
        _R = jnp.dot(_K, y) - _a
        return _R

    def compute_objective(x, y):
        _F = jnp.dot(y, _a) * _obj_scaling
        return _F

    def compute_constraint(x, y):
        _C = (_rho * _b * _l * jnp.sum(x) / _n - _m0) * _con_scaling
        return jnp.array([_C])

    def compute_composite_objective(x):
        return compute_objective(x, compute_state(x, float))

    def compute_composite_constraint(x):
        return compute_constraint(x, compute_state(x, float))

    def compute_composite_objective_cs(x):
        return compute_objective(x, compute_state(x, complex))

    prob = opt.JaxProblem(
        x0=np.ones(_n), nc=1, jax_obj=compute_composite_objective, jax_con=compute_composite_constraint,
        name=f'cantilever_{_n}_jax', order=1,
        xl=1e-3, cl=0., cu=0.,
    )

    jit_objective = jax.jit(compute_objective)
    jit_constraint = jax.jit(compute_constraint)
    jit_state = jax.jit(compute_state)
    jit_residual = jax.jit(compute_residual)
    return (
        compute_composite_constraint,
        compute_composite_objective,
        compute_composite_objective_cs,
        compute_constraint,
        compute_objective,
        compute_residual,
        compute_state,
        jit_constraint,
        jit_objective,
        jit_residual,
        jit_state,
        prob,
    )


@app.cell(hide_code=True)
def _(mo, n_input):
    # Pertubation inputs

    _n = int(n_input.value)

    recompute_state_checkbox = mo.ui.checkbox(
        value=True, label=r"Recompute state",
    )

    delta_index_slider = mo.ui.slider(
        start=0, stop=_n-1, step=1, value=0,
        label=r"Perturbation index",
        show_value=True,
    )

    delta_slider = mo.ui.slider(
        start=-10, stop=2, step=0.1, value=-1,
        label=r"Perturbation ($10^x$)",
    )

    sigfigs_slider = mo.ui.slider(
        start=1, stop=10, step=1, value=3,
        label=r"Sig. figs.",
        show_value=True,
    )
    return (
        delta_index_slider,
        delta_slider,
        recompute_state_checkbox,
        sigfigs_slider,
    )


@app.cell(hide_code=True)
def _(np, sigfigs_slider):
    def numpy_round_sigfig(arr, _sig=None):
        """
        Round a NumPy array to the specified number of significant figures.
        Parameters:
            arr (array-like): Input array
        Returns:
            numpy.ndarray: Array rounded to the specified significant figures
        """
        if _sig is None:
            _sig = sigfigs_slider.value
        arr = np.asarray(arr, dtype=float)
        _nonzero = arr != 0
        _scale = 10 ** (_sig - 1 - np.floor(np.log10(np.abs(arr[_nonzero]))))
        arr[_nonzero] = np.round(arr[_nonzero] * _scale) / _scale
        return arr
    return (numpy_round_sigfig,)


@app.cell(hide_code=True)
def _(
    delta_index_slider,
    delta_slider,
    mo,
    numpy_round_sigfig,
    recompute_state_checkbox,
    sigfigs_slider,
):
    _delta = numpy_round_sigfig(10 ** float(delta_slider.value))
    _delta_md = mo.md(f"{_delta}")

    mo.vstack([
        mo.hstack([delta_index_slider, delta_slider, _delta_md], justify="start"),
        mo.hstack([sigfigs_slider, recompute_state_checkbox]),
    ])
    return


@app.cell(hide_code=True)
def _(
    E_input,
    a0_input,
    b_input,
    con_scaling_input,
    delta_index_slider,
    delta_slider,
    jit_constraint,
    jit_objective,
    jit_residual,
    jit_state,
    l_input,
    m0_input,
    mo,
    n_input,
    np,
    numpy_round_sigfig,
    obj_scaling_input,
    pd,
    recompute_state_checkbox,
    rho_input,
):
    # Apply perturbations and compute all quantities

    _n = int(n_input.value)
    _b = float(b_input.value)
    _l = float(l_input.value)
    _m0 = float(m0_input.value)
    _rho = float(rho_input.value)
    _E = float(E_input.value)
    _a0 = float(a0_input.value)
    _obj_scaling = 10 ** float(obj_scaling_input.value)
    _con_scaling = 10 ** float(con_scaling_input.value)

    _recompute_state = recompute_state_checkbox.value
    _delta_index = delta_index_slider.value
    _delta = 10 ** float(delta_slider.value)

    _volume = _m0 / _rho
    _h = _volume / _l / _b

    x0 = _h * np.ones(_n)
    x = np.array(x0)

    x[_delta_index] += _delta

    y0 = jit_state(x0)
    if _recompute_state:
        y = jit_state(x)
    else:
        y = y0

    r0 = jit_residual(x0, y0)
    r = jit_residual(x, y)

    f0 = jit_objective(x0, y0)
    f = jit_objective(x, y)

    c0 = jit_constraint(x0, y0)[0]
    c = jit_constraint(x, y)[0]

    _raw_table_dict = {
        'x0': x0, 'x': x,
        'y0': y0, 'y': y,
        'r0': r0, 'r': r,
        '||r0||': np.linalg.norm(r0), '||r||': np.linalg.norm(r),
        'f0': f0, 'f': f,
        'c0': c0, 'c': c,
    }

    for _key in _raw_table_dict:
        _raw_table_dict[_key] = numpy_round_sigfig(np.array(_raw_table_dict[_key]))

    _pd_series_list = [pd.Series(_raw_table_dict[key], name=key) for key in _raw_table_dict]
    _df = pd.concat(_pd_series_list, axis=1)

    table = mo.ui.table(_df, show_column_summaries=False)
    return c, c0, f, f0, r, r0, table, x, x0, y, y0


@app.cell(hide_code=True)
def _(mo, table):
    mo.accordion({r"**All vectors in tabular form**":
        table
    }, lazy=True)
    return


@app.cell(hide_code=True)
def _(go, mo, r, r0, x, x0, y, y0):
    # Thickness profile figure

    figs = {}

    for _fig_name, _fig_data in [
        (r"Heights ($x$)", [
            (x0, "x0", "red"),
            (x, "x", "blue"),
        ]),
        (r"Displacements ($y$)", [
            (y0[::2], "y0", "red"),
            (y[::2], "y", "blue"),
        ]),
        (r"Residuals ($R(x, y)$)", [
            (r0, "r0", "red"),
            (r, "r", "blue"),
        ]),   
    ]:    
        figs[_fig_name] = _fig = go.Figure()

        for _data, _name, _color in _fig_data:
            _fig.add_trace(go.Scatter(
                y=_data,
                mode="lines+markers",
                line=dict(color=_color, width=2),
                name=_name,
            ))

        _fig.update_layout(
            xaxis_title="Vector index",
            yaxis_title=_fig_name,
            margin=dict(l=0, r=0, t=0, b=0),
            height=200, width=1080,
        )

    mo.accordion({r"**Visualizations of design, state, and residual vectors**":
        mo.vstack([figs[_key] for _key in figs])
    }, lazy=True)
    return (figs,)


if __name__ == "__main__":
    app.run()
