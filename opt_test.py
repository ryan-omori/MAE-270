import numpy as np
import matplotlib.pyplot as plt
import time

import csdl_alpha as csdl
import modopt as opt

from TrainSurrogate_test import load_surrogate
from test import true_model, to_physical

N_VARS = 12


# ======================================================
# SURROGATE PREDICTION
# ======================================================

def surrogate_predict(h, models):

    preds = []

    for model in models:
        preds.append(model.predict(h.reshape(1,-1))[0])

    return np.array(preds)


# ======================================================
# CSDL COMPONENT
# ======================================================

class HullSurrogateComp(csdl.CustomExplicitOperation):

    def __init__(self, models):
        super().__init__()
        self.models = models

    def evaluate(self, h):

        self.declare_input("h", h)

        R = self.create_output("R", shape=(1,))
        V = self.create_output("V", shape=(1,))

        return R, V

    def compute(self, inputs, outputs):

        h = inputs["h"]

        pred = surrogate_predict(h, self.models)

        V = pred[0]
        R = pred[-1]

        outputs["R"] = np.array([R])
        outputs["V"] = np.array([V])


# ======================================================
# BUILD OPTIMIZATION PROBLEM
# ======================================================

def build_problem(models, V_ref, gamma, h0):

    rec = csdl.Recorder(inline=True)
    rec.start()

    h = csdl.Variable(value=h0.copy(), name="h")

    h.set_as_design_variable(
        lower=np.zeros(N_VARS),
        upper=np.ones(N_VARS)
    )

    comp = HullSurrogateComp(models)

    R, V = comp.evaluate(h)

    R.set_as_objective()

    ((1-gamma)*V_ref - V).set_as_constraint(upper=0.0)
    (V - (1+gamma)*V_ref).set_as_constraint(upper=0.0)

    rec.stop()

    sim = csdl.experimental.PySimulator(rec)

    prob = opt.CSDLAlphaProblem(
        problem_name="HullMDO",
        simulator=sim
    )

    prob.x0 = h0.copy()

    return prob


# ======================================================
# MULTI START
# ======================================================

def run_multistart(models, V_ref, gamma, n_starts=6):

    best = None

    for i in range(n_starts):

        h0 = np.random.rand(N_VARS)

        prob = build_problem(models, V_ref, gamma, h0)

        solver = opt.SQP(
            prob,
            maxiter=200
        )

        solver.solve()

        h_opt = solver.results["x"]
        R_opt = solver.results["f"]

        if best is None or R_opt < best[0]:
            best = (R_opt, h_opt)

        print(f"Start {i+1} -> R = {R_opt:.3f}")

    return best


# ======================================================
# VERIFY TRUE MODEL
# ======================================================

def verify_solution(h_opt):

    R, V, *_ = true_model(h_opt)

    print("\nTrue model evaluation")
    print("---------------------")
    print("Resistance:", R)
    print("Volume:", V)

    return R, V


# ======================================================
# MAIN
# ======================================================

def main():

    print("\nLoading surrogate...")

    models = load_surrogate("hull_surrogate.pkl")

    print("Models loaded:", len(models))

    print("\nComputing reference design...")

    h_ref = np.ones(N_VARS)*0.5

    R_ref, V_ref, *_ = true_model(h_ref)

    print("Reference resistance:", R_ref)
    print("Reference volume:", V_ref)

    gamma = 0.03

    print("\nRunning optimizer...")

    start = time.time()

    R_best, h_best = run_multistart(models, V_ref, gamma)

    print("\nOptimization finished")
    print("Best resistance:", R_best)

    verify_solution(h_best)

    print("\nRuntime:", time.time()-start)


if __name__ == "__main__":
    main()