import marimo

__generated_with = "0.19.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Cantilever beam height optimization in CSDL**
    """)
    return


@app.cell
def _():
    import marimo as mo
    import plotly.graph_objects as go
    import plotly.colors as pc
    import matplotlib.colors as mcolors
    import plotly.subplots as sp
    import numpy as np
    import pandas as pd
    import os
    from collections import namedtuple

    import csdl_alpha as csdl

    import modopt as opt
    import modopt.postprocessing
    return csdl, go, mcolors, mo, modopt, np, opt, pc, sp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Problem description**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _overview = r"""
    Consider a solid cantilever beam modeled using $n$ Euler--Bernoulli beam elements of equal length.
    The design problem is to find the beam height distribution that minimizes the structural compliance
    subject to a mass constraint.
    The beam is clamped on the left end and is subjected to a upward tip load on the right end.
    The finite element model has $m$ state variables, which are the $2n$ nodal values of 
    transverse displacements and rotations excluding those of the clamped node.
    """

    mo.accordion({r"**Overview**": _overview})
    return


@app.cell(hide_code=True)
def _(mo):
    _nomenclature_note = r"""
    *In this notebook, variables are represented by lower-case letters and functions by upper-case letters.*
    """

    _inputs = r"""
    | Type | Variable | Description |
    | :--- | :--- | :--- |
    | Design variable | $h \in \mathbb{R}^n$ | Element height vector |
    | Parameter | $b \in \mathbb{R}$ | Beam width |
    | Parameter | $l \in \mathbb{R}$ | Beam length |
    | Parameter | $m_0 \in \mathbb{R}$ | Target mass |
    | Parameter | $\rho \in \mathbb{R}$ | Beam material density |
    | Parameter | $E \in \mathbb{R}$ | Young's modulus |
    | Parameter | $a \in \mathbb{R}^m$ | External force vector |
    | State variable | $u \in \mathbb{R}^m$ | Displacement vector |

    | Model equations/functions | Description |
    | :--- | :--- |
    | $K(h) u = a$ | Finite element equations |
    | $P(h, u) := u^T K(h) u$ | Structural compliance |
    | $M(h) := \rho * b * l * \left( \sum_{i=1}^n h_i \right) / n$ | Beam mass |
    """

    _general_notation = r"""
    | Variable/function (general notation) | Description |
    | :--- | :--- |
    | $x \in \mathcal{D} \subseteq \mathbb{R}^n, \; x := h$ | Design vector |
    | $y \in \mathbb{R}^m, \; y := u$ | State vector |
    | $R(x, y) := K(x) y - a$ | Residual function |
    | $F(x, y) := P(x, y)$ | Objective function |
    | $C(x, y) := M(x) - m_0$ | Constraint function |

    | Composite function | Description |
    | :--- | :--- |
    | $\mathcal{Y}: \mathcal{D} \mapsto \mathbb{R}^m \, \mid \, R(x, \mathcal{Y}(x)) = 0$ | Implicit function defining $y$ |
    | $\mathcal{R}(x) := R(x, \mathcal{Y}(x))$ | Composite residual function |
    | $\mathcal{F}(x) := F(x, \mathcal{Y}(x))$ | Composite objective function |
    | $\mathcal{C}(x) := C(x, \mathcal{Y}(x))$ | Composite constraint function |
    """

    mo.accordion({r"**Nomenclature**": mo.vstack([
        mo.md(_nomenclature_note),
        mo.hstack([mo.md(_inputs), mo.md(_general_notation)], justify="center", gap=2.),
    ])})
    return


@app.cell(hide_code=True)
def _(mo):
    _general_notation = r"""
    ### 1. Local stiffness matrix

    The global finite element equation $K(h)u=a$ can be formed by summing the local contributions of each element.
    The stiffness matrix Ki(hi)∈R4×4K_i(h_i) \in \mathbb{R}^{4 \times 4} for the iith beam element is:

    $$
    K_i(h_i) = 
    \frac{E\,I(h_i)}{l_0^3}
    \begin{bmatrix}
    12 & 6\,l_0 & -12 & 6\,l_0 \\
    6\,l_0 & 4\,l_0^2 & -6\,l_0 & 2\,l_0^2 \\
    -12 & -6\,l_0 & 12 & -6\,l_0 \\
    6\,l_0 & 2\,l_0^2 & -6\,l_0 & 4\,l_0^2 
    \end{bmatrix},
    $$

    where
    $E$ is Young’s modulus,
    $I(h_i)$ is the moment of inertia of the cross-section (often dependent on $h_i$, the element “height” or thickness),
    $l_0$ is the length of a single beam element (assumed constant across elements).

    The local element equation before normalization is:

    $$
    K_i(h_i)\,u_i = a_i,
    $$

    where
    $u_i = [\,d_{1i},\, \theta_{1i},\, d_{2i},\, \theta_{2i}\,]^T$ is the local displacement vector (translational and rotational DOFs at each node),
    $a_i = [\,F_{1i},\, M_{1i},\, F_{2i},\, M_{2i}\,]^T$ is the local force vector (forces and moments at each node).

    ### 2. Displacement and Force Normalization

    We want to nondimensionalize $u$ such that
    displacements are scaled by $1/l$ and
    rotations remain unscaled.

    Define the diagonal matrix $S_u$ for a single beam element with 4 DOFs:

    $$
    S_u
    =
    \mathrm{diag}\!\bigl(\tfrac{1}{l},\,1,\,\tfrac{1}{l},\,1\bigr).
    $$

    Then the scaled local displacement vector is

    $$
    \bar{u}_i = S_u\,u_i,
    \quad\Longrightarrow\quad
    \bar{u}_i
    =
    \begin{bmatrix}
    d_{1i}/l \\[4pt]
    \theta_{1i} \\[4pt]
    d_{2i}/l \\[4pt]
    \theta_{2i}
    \end{bmatrix}.
    $$

    Because the force vector $a_i$ has entries corresponding to forces (e.g., $F_{1i}, F_{2i}$) and moments (e.g., $M_{1i}, M_{2i}$), we may want to use different scaling factors:
    forces might be normalized by a characteristic quantity $E\,b\,l$; and
    moments might be normalized by $E\,b\,l^2$.

    Hence, define

    $$
    S_a
    =
    \mathrm{diag}\!\Bigl(\tfrac{1}{E\,b\,l},\;\tfrac{1}{E\,b\,l^2},\;\tfrac{1}{E\,b\,l},\;\tfrac{1}{E\,b\,l^2}\Bigr).
    $$

    Then the scaled local force vector is

    $$
    \bar{a}_i = S_a\,a_i,
    \quad\Longrightarrow\quad
    \bar{a}_i
    =
    \begin{bmatrix}
    F_{1i}/(E\,b\,l) \\[4pt]
    M_{1i}/(E\,b\,l^2) \\[4pt]
    F_{2i}/(E\,b\,l) \\[4pt]
    M_{2i}/(E\,b\,l^2)
    \end{bmatrix}.
    $$

    ### 3. Local Stiffness Matrix After Normalization

    Starting with the local system

    $$
    K_i(h_i)\,u_i = a_i,
    $$

    substitute $u_i = S_u^{-1}\,\bar{u}_i$ and $a_i = S_a^{-1}\,\bar{a}_i$:

    $$
    K_i(h_i)\,\bigl(S_u^{-1}\,\bar{u}_i\bigr) = S_a^{-1}\,\bar{a}_i.
    $$

    Multiply on the left by $S_a$:

    $$
    S_a\,K_i(h_i)\,S_u^{-1}\,\bar{u}_i = \bar{a}_i.
    $$

    Define the scaled local stiffness matrix:

    $$
    \bar{K}_i(h_i)
    \; := \;
    S_a\,K_i(h_i)\,S_u^{-1}.
    $$

    Expanding,

    $$
    \bar{K}_i(h_i)
    = 
    \frac{I(h_i)}{bl^3}
    \begin{bmatrix}
    12n^3 & 6n^2 & -12n^3 & 6n^2 \\
    6n^2 & 4\,n & -6n^2 & 2n \\
    -12n^3 & -6n^2 & 12n^3 & -6n^2 \\
    6n^2 & 2n & -6n^2 & 4n
    \end{bmatrix}
    =
    \frac{\bar h_i^3}{12}
    \begin{bmatrix}
    12n^3 & 6n^2 & -12n^3 & 6n^2 \\
    6n^2 & 4\,n & -6n^2 & 2n \\
    -12n^3 & -6n^2 & 12n^3 & -6n^2 \\
    6n^2 & 2n & -6n^2 & 4n
    \end{bmatrix},
    $$

    where $\bar h_i:=h_i/l$ is the normalized height of the $i$th beam element.

    The normalized local element equation is

    $$
    \bar{K}_i(h_i)\,\bar{u}_i = \bar{a}_i.
    $$

    The code implementation uses normalized quantities and equations throughout.
    """

    mo.accordion({r"**Model**": mo.vstack([
        mo.md(_general_notation),
        # mo.md(_nomenclature_note),
        # mo.hstack([mo.md(_inputs), mo.md(_general_notation)], justify="center", gap=2.),
    ])})
    return


@app.cell(hide_code=True)
def _(mo):
    _problem = r"""
    ||||
    | ---: | :--- | : --- |
    | $\underset{h}{\text{minimize}}$ | $P(h, \mathcal{Y}(h))$ | Structural compliance |
    | $\text{subject to}$ | $M(h) - m_0 = 0$ | Mass constraint |
    """

    mo.accordion({r"**Optimization problem**": mo.hstack([
        mo.md(_problem)
    ], justify="center")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Execution and Results**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Define model parameter inputs

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

    parameters_stack
    return (
        E_input,
        a0_input,
        b_input,
        con_scaling_input,
        l_input,
        m0_input,
        n_input,
        obj_scaling_input,
        rho_input,
    )


@app.cell
def _(
    E_input,
    a0_input,
    b_input,
    con_scaling_input,
    csdl,
    l_input,
    m0_input,
    n_input,
    np,
    obj_scaling_input,
    rho_input,
):
    # Implement model

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
        _Ki_wo_x3 = 1. / 12. * np.array([
            [12 * _n ** 3, 6 * _n ** 2, -12 * _n ** 3, 6 * _n ** 2],
            [6 * _n ** 2, 4 * _n, -6 * _n ** 2, 2 * _n],
            [-12 * _n ** 3, -6 * _n ** 2, 12 * _n ** 3, -6 * _n ** 2],
            [6 * _n ** 2, 2 * _n, -6 * _n ** 2, 4 * _n],
        ])
        _K = csdl.Variable(value=np.zeros((_n * 2 + 2, _n * 2 + 2), dtype=dtype))
        for i in range(_n):
            _K = _K.set(csdl.slice[2*i:2*i+4, 2*i:2*i+4], _K[2*i:2*i+4, 2*i:2*i+4] + (_Ki_wo_x3 * x[i] ** 3))

        return _K[2:, 2:]

    def compute_state(x, dtype=float):
        _K = _compute_stiffness_matrix(x, dtype)
        _y = csdl.solve_linear(_K, _a)
        return _y

    def compute_objective(x, y):
        _F = csdl.inner(y, _a) * _obj_scaling
        return _F

    def compute_constraint(x, y):
        _C = (_rho * _b * _l * csdl.sum(x) / _n - _m0) * _con_scaling
        return _C
    return compute_constraint, compute_objective, compute_state


@app.cell
def _(
    compute_constraint,
    compute_objective,
    compute_state,
    csdl,
    n_input,
    np,
    opt,
):
    # Instantiate the model

    _n = int(n_input.value)

    _recorder = csdl.Recorder(inline=True)
    _recorder.start()

    x = csdl.Variable(value=np.ones(_n), name='height vector')
    y = compute_state(x)
    f = compute_objective(x, y)
    c = compute_constraint(x, y)

    x.set_as_design_variable(lower=1.e-3)
    f.set_as_objective()
    c.set_as_constraint(equals=0.)

    _recorder.stop()

    simulator = csdl.experimental.PySimulator(_recorder)
    prob = opt.CSDLAlphaProblem(problem_name='beam', simulator=simulator)
    return c, f, prob, simulator, x, y


@app.cell
def _(opt, prob):
    # Run optimization algorithm

    _maxiter = 300
    _opt_tol = 1e-10

    optimizer = opt.PySLSQP(prob, recording=True, solver_options={
        'maxiter': _maxiter, 'acc': _opt_tol, 'save_vars': ['optimality'],
    })
    # optimizer = opt.OpenSQP(prob, recording=True, maxiter=_maxiter, opt_tol=_opt_tol)

    results = optimizer.solve()
    return (results,)


@app.cell(hide_code=True)
def _(modopt, n_input, np, prob, results, x, y):
    # Load results

    results_dict = modopt.postprocessing.load_variables(
    # results_dict = load_variables(
        results['out_dir']+'/record.hdf5',
        ['x'],
    )

    for _quantity_name in results_dict:
        results_dict[_quantity_name] = np.array(results_dict[_quantity_name])

    _num_iterations = len(results_dict['x'])
    _num_constraints = 1

    _n = int(n_input.value)

    _sim = prob.options["simulator"]

    results_dict['y'] = np.zeros((_num_iterations, 2 * _n))
    for _index in range(_num_iterations):
        _x = results_dict['x'][_index, :]
        _sim[x] = _x
        _sim.run()
    
        results_dict['y'][_index, :] = _sim[y]
    return (results_dict,)


@app.cell(hide_code=True)
def _(mo, results_dict):
    # Create iteration slider

    _num_iterations = len(results_dict['x'])

    iteration_slider = mo.ui.slider(
        start=0, stop=_num_iterations - 1, step=1, 
        value=_num_iterations - 1, full_width=True,
        label="Optimization iteration",
        debounce=True,
    )

    iteration_slider
    return (iteration_slider,)


@app.cell(hide_code=True)
def _(
    E_input,
    a0_input,
    b_input,
    go,
    iteration_slider,
    l_input,
    m0_input,
    mcolors,
    mo,
    n_input,
    np,
    pc,
    results_dict,
    rho_input,
):
    # Plot beam visualization

    _n = int(n_input.value)
    _b = float(b_input.value)
    _l = float(l_input.value)
    _m0 = float(m0_input.value)
    _rho = float(rho_input.value)
    _E = float(E_input.value)
    _a0 = float(a0_input.value)

    def _lighten_color(color, amount):
        """
        Lightens the given color by blending it with white.
        :param color: The base color (named color, hex code, or RGB tuple).
        :param amount: A float from 0 to 1 specifying the amount of lightening (higher = lighter).
        :return: The lightened color in hex format.
        """
        try:
            c = mcolors.cnames[color]  # Named color
        except KeyError:
            c = color  # Hex or RGB tuple
        c = mcolors.to_rgb(c)  # Convert to RGB
        return mcolors.to_hex([1 - (1 - x) * (1 - amount) for x in c])

    _fig = go.Figure()

    _iteration = min(iteration_slider.value, len(results_dict['x']) - 1)

    _x = np.linspace(0, _l, _n + 1)
    _y = np.concatenate([np.zeros(1), results_dict['y'][_iteration, ::2]])
    _phi = np.concatenate([np.zeros(1), results_dict['y'][_iteration, 1::2]])
    _h = results_dict['x'][_iteration]

    for _name, _deformed, _line_lighten, _fill_lighten in [
        ("Undeformed", False, 0.6, 0.8),
        ("Deformed", True, 0., 0.2),
    ]:
        _all_points_list = []
        for _i_elem in range(_n):        
            if _deformed:
                def _get_point(_ix, _iy):
                    return [
                        _x[_i_elem + _ix] + (1 - 2 * _iy) * _h[_i_elem] / 2. * np.tan(_phi[_i_elem + _ix]), 
                        _y[_i_elem + _ix] - (1 - 2 * _iy) * _h[_i_elem] / 2. * np.cos(_phi[_i_elem + _ix]),
                    ]
            else:
                def _get_point(_ix, _iy):
                    return [
                        _x[_i_elem + _ix], 
                        -(1 - 2 * _iy) * _h[_i_elem] / 2.,
                    ]

            _points = np.array([
                [None, None],
                _get_point(0, 0),
                _get_point(1, 0),
                _get_point(1, 1),
                _get_point(0, 1),
                _get_point(0, 0),
            ])
            _all_points_list.append(_points)

        _all_points = np.vstack(_all_points_list)

        _line_color = _lighten_color(pc.qualitative.Plotly[0], _line_lighten)
        _fill_color = _lighten_color(pc.qualitative.Plotly[0], _fill_lighten)

        _fig.add_trace(go.Scatter(
            x=_all_points[:, 0],
            y=_all_points[:, 1],
            mode="lines",
            fill="toself",
            fillcolor=_fill_color,
            line=dict(color=_line_color, width=2),
            name=f"{_name}",
        ))

    _fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        xaxis=dict(scaleratio=1, range=[-0.05 * _l, 1.05 * _l]),
        yaxis=dict(scaleratio=1, range=[-0.2, 0.2], scaleanchor="x"),
        legend=dict(
            x=0.5,  # Center the legend horizontally
            y=-0.2,  # Place the legend below the plot
            xanchor="center",  # Align legend to its center
            orientation="h",  # Horizontal legend layout
            bordercolor="Black",
            borderwidth=1
        ),
        showlegend=True,
        margin=dict(l=0, r=0, t=0, b=0),
        height=500, width=1000,
    )

    fig_beam = _fig

    mo.accordion({r"**Beam visualization**": mo.vstack([
        fig_beam
    ])})
    return


@app.cell(hide_code=True)
def _(
    go,
    iteration_slider,
    l_input,
    mo,
    n_input,
    np,
    results_dict,
    simulator,
    sp,
    x,
    y,
):
    # Plot height and displacement profiles

    _n = int(n_input.value)
    _l = float(l_input.value)
    _iteration = int(iteration_slider.value)

    simulator[x] = results_dict['x'][_iteration]
    simulator.run()

    _fig = sp.make_subplots(rows=1, cols=2)

    _fig.add_trace(go.Scatter(
        x=np.linspace(0, _l, _n + 1),
        y=results_dict['x'][_iteration],
        mode="lines+markers",
        line=dict(width=2),
    ), row=1, col=1)
    _fig.update_xaxes(title_text="Lengthwise coordinate", row=1, col=1)
    _fig.update_yaxes(title_text="Normalized height", row=1, col=1)

    _fig.add_trace(go.Scatter(
        x=np.linspace(0, _l, _n + 1),
        y=simulator[y][::2],
        mode="lines+markers",
        line=dict(width=2),
    ), row=1, col=2)
    _fig.update_xaxes(title_text="Lengthwise coordinate", row=1, col=2)
    _fig.update_yaxes(title_text="Normalized displacement", row=1, col=2)

    _fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=400, width=1000,
        showlegend=False,
    )

    mo.accordion({r"**Height and displacement profiles**": mo.vstack([
        mo.hstack([_fig], justify="center"),
    ])})
    return


@app.cell(hide_code=True)
def _(c, f, go, iteration_slider, mo, np, results_dict, simulator, sp, x):
    # Plot convergence histories

    _num_iterations = len(results_dict['x'])

    _obj_history = np.empty(_num_iterations)
    _con_history = np.empty(_num_iterations)

    for _iteration in range(_num_iterations):
        simulator[x] = results_dict['x'][_iteration]
        simulator.run()
        _obj_history[_iteration] = simulator[f]
        _con_history[_iteration] = simulator[c]

    _iteration = int(iteration_slider.value)

    _fig = sp.make_subplots(rows=1, cols=2)

    _fig.add_trace(go.Scatter(
        x=list(range(_num_iterations)),
        y=_obj_history,
        mode="lines+markers",
        line=dict(width=2),
    ), row=1, col=1)
    _fig.add_trace(go.Scatter(
        x=[_iteration],
        y=[_obj_history[_iteration]],
        mode="markers",
        marker=dict(size=10, color="black"),
    ), row=1, col=1)
    _fig.update_xaxes(title_text="Iteration", row=1, col=1)
    _fig.update_yaxes(title_text="Objective", row=1, col=1, type="log", exponentformat="power")

    _fig.add_trace(go.Scatter(
        x=list(range(_num_iterations)),
        y=_con_history,
        mode="lines+markers",
        line=dict(width=2),
    ), row=1, col=2)
    _fig.add_trace(go.Scatter(
        x=[_iteration],
        y=[_con_history[_iteration]],
        mode="markers",
        marker=dict(size=10, color="black"),
    ), row=1, col=2)
    _fig.update_xaxes(title_text="Iteration", row=1, col=2)
    _fig.update_yaxes(title_text="Constraint", row=1, col=2, type="log", exponentformat="power")

    _fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=400, width=1000,
        showlegend=False,
    )

    mo.accordion({r"**Convergence histories**": mo.vstack([
        mo.hstack([_fig], justify="center")
    ])})
    return


if __name__ == "__main__":
    app.run()
