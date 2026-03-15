import numpy as np
import pickle
import matplotlib.pyplot as plt
import modopt as mo

from truth_model_model import (
    HullSettings,
    design_bounds,
    baseline_design,
    hull_sections,
)


class RBFWrapper:
    def __init__(self, resistance_model, volume_model):
        self.sm_R, self.yR_mu, self.yR_sig = resistance_model
        self.sm_V, self.yV_mu, self.yV_sig = volume_model

    def predict_resistance(self, x):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        ys = self.sm_R.predict_values(x)
        return float(ys[0, 0] * self.yR_sig[0] + self.yR_mu[0])

    def predict_volume(self, x):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        ys = self.sm_V.predict_values(x)
        return float(ys[0, 0] * self.yV_sig[0] + self.yV_mu[0])

    def grad_fd(self, fun, x, h=1e-6):
        x = np.asarray(x, dtype=float).copy()
        g = np.zeros_like(x)
        f0 = fun(x)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += h
            g[i] = (fun(xp) - f0) / h
        return g


def make_problem(wrapper, settings):
    lb, ub = design_bounds(settings)
    x0 = baseline_design(settings)

    def obj(x):
        return wrapper.predict_resistance(x)

    def grad(x):
        return wrapper.grad_fd(wrapper.predict_resistance, x)

    def con(x):
        return np.array([wrapper.predict_volume(x) - settings.target_volume])

    def jac(x):
        return wrapper.grad_fd(wrapper.predict_volume, x).reshape(1, -1)

    prob = mo.ProblemLite(
        x0=x0,
        name="hull_rbf_problem",
        obj=obj,
        grad=grad,
        con=con,
        jac=jac,
        xl=lb,
        xu=ub,
        cl=np.array([0.0]),
        cu=np.array([0.0]),
    )
    return prob


def plot_side_profile(z, settings, title="Optimized hull side profile"):
    x_hat, _, draft = hull_sections(z, settings)
    x = x_hat * settings.L
    keel = -draft
    deck = np.zeros_like(x)

    plt.figure(figsize=(10, 4))
    plt.plot(x, deck, linewidth=2, label="Deck / waterline reference")
    plt.plot(x, keel, linewidth=2, label="Keel")
    plt.fill_between(x, keel, deck, alpha=0.3)
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    settings = HullSettings(
        n_beam_cp=4,
        n_draft_cp=4,
        L=30.0,
        V_ship=6.0,
        target_volume=280.0,
    )

    with open("rbf_resistance.pkl", "rb") as f:
        R_model = pickle.load(f)
    with open("rbf_volume.pkl", "rb") as f:
        V_model = pickle.load(f)

    wrapper = RBFWrapper(R_model, V_model)
    prob = make_problem(wrapper, settings)

    opt = mo.SLSQP(
        prob,
        solver_options={"maxiter": 200, "ftol": 1e-8},
        turn_off_outputs=True,
    )
    opt.solve()
    res = opt.results

    x_best = np.array(res.get("x", res.get("x_star")))
    f_best = res.get("objective", res.get("f_star"))

    print("Best predicted resistance:", f_best)
    print("Predicted volume:", wrapper.predict_volume(x_best))
    print("Best design vector:", x_best)

    plot_side_profile(x_best, settings, title="Optimized hull side profile (RBF surrogate)")


if __name__ == "__main__":
    main()