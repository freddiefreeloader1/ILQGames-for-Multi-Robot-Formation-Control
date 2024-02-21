import numpy as np
from scipy.optimize import approx_fprime

class ProximityCost:
    def __init__(self, d_threshold=0.5, idx1 = 0, idx2 = 0):
        self.d_threshold = 0.5 
        self.idx1 = idx1
        self.idx2 = idx2

    def evaluate(self, x, u):
        dist = np.sqrt((x[4*self.idx1] - x[4*self.idx2])**2 + (x[4*self.idx1 + 1] - x[4*self.idx2 + 1])**2)
        return dist if dist > self.d_threshold else 0.0

class ReferenceCost:
    def __init__(self, d_threshold=0.5, idx = 0, x_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0])):
        self.idx = idx
        self.x_ref = x_ref

    def evaluate(self, x, u):
        dist = np.sqrt((x[4*self.idx] - self.x_ref[4*self.idx])**2 + (x[4*self.idx + 1] - self.x_ref[4*self.idx + 1])**2)
        return dist

class TrialCost:
    def __init__(self, d_threshold=0.5):
        self.d_threshold = 0.5
    def evaluate(self, x, u):
        dist = x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2
        return dist

class OverallCost:
    def __init__(self, subsystem_cost_functions):
        self.subsystem_cost_functions = subsystem_cost_functions

    def evaluate(self, x, u):
        total_cost = 0.0
        for subsystem_cost in self.subsystem_cost_functions:
            total_cost += subsystem_cost.evaluate(x, u)
        return total_cost

    def gradient_x(self, x, u):
        grad_x = approx_fprime(x, lambda x: self.evaluate(x, u), epsilon=1e-6)
        return grad_x
    def gradient_u(self, x, u):
        grad_u = approx_fprime(u, lambda u: self.evaluate(x, u), epsilon=1e-6)
        return grad_u
    def hessian_x(self, x, u):
        hessian_x = approx_fprime(x, lambda x: self.gradient_x(x, u), epsilon=1e-6)
        return hessian_x

    def hessian_u(self, x, u):
        hessian_u = approx_fprime(u, lambda u: self.gradient_u(x, u), epsilon=1e-6)
        return hessian_u

def trial():
    trial_cost = TrialCost()
    overall_cost = OverallCost([trial_cost])

    x_example = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    u_example = np.array([1, 1])

    total_cost = overall_cost.evaluate(x_example, u_example)
    gradient_x = overall_cost.gradient_x(x_example, u_example)
    hessian_x = overall_cost.hessian_x(x_example, u_example)
    hessian_u = overall_cost.hessian_u(x_example, u_example)

    print("Total Cost:", total_cost)
    print("Gradient with respect to x:", gradient_x)
    print("Hessian with respect to x:", hessian_x)
    print("Hessian with respect to u:", hessian_u)

