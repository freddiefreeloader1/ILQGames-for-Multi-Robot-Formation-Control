import autograd.numpy as np
from autograd import grad, elementwise_grad, hessian
from scipy.optimize import approx_fprime

class ProximityCost:
    def __init__(self, d_threshold=0.5, idx1=0, idx2=0, weight=1.0):
        self.d_threshold = d_threshold
        self.idx1 = idx1
        self.idx2 = idx2
        self.weight = weight

    def evaluate(self, x, u):
        dist = np.sqrt((x[4 * self.idx1] - x[4 * self.idx2])**2 + (x[4 * self.idx1 + 1] - x[4 * self.idx2 + 1])**2)
        return 0.0 if dist > self.d_threshold else self.weight * (self.d_threshold - dist)**2

    def gradient_x(self, x, u):
        grad_x = elementwise_grad(self.evaluate, 0)
        return grad_x(x, u)

    def hessian_x(self, x, u):
        hessian_x_fn = hessian(self.evaluate, 0)
        return hessian_x_fn(x, u)

class ReferenceCost:
    def __init__(self, idx=0, x_ref=np.array([0, 0, 0, 0, 0, 0, 0, 0]), weight=1.0):
        self.idx = idx
        self.x_ref = x_ref
        self.weight = weight

    def evaluate(self, x, u):
        dist = np.sqrt((x[4 * self.idx] - self.x_ref[4 * self.idx])**2 +
                       (x[4 * self.idx + 1] - self.x_ref[4 * self.idx + 1])**2 +
                       (x[4 * self.idx + 2] - self.x_ref[4 * self.idx + 2])**2 +
                       (x[4 * self.idx + 3] - self.x_ref[4 * self.idx + 3])**2)
        return dist * self.weight

    def gradient_x(self, x, u):
        grad_x = elementwise_grad(self.evaluate, 0)
        return grad_x(x, u)

    def hessian_x(self, x, u):
        hessian_x_fn = hessian(self.evaluate, 0)
        return hessian_x_fn(x, u)

class InputCost:
    def __init__(self, idx, weight=1.0):
        self.weight = weight
        self.idx = idx

    def evaluate(self, x, u):
        return self.weight * (3*u[0]**2 + 2*u[1]**2)

    def gradient_x(self, x, u):
        grad_x = elementwise_grad(self.evaluate, 0)
        return grad_x(x, u)

    def hessian_x(self, x, u):
        hessian_x_fn = hessian(self.evaluate, 0)
        return hessian_x_fn(x, u)

class WallCost:
    def __init__(self, idx, weight=1.0):
        self.idx = idx
        self.weight = weight

    def evaluate(self, x, u):
        x_robot = x[4 * self.idx]
        y_robot = x[4 * self.idx + 1]
        side_length = 7.0
        x_center = 0.0
        y_center = 0.0
        dx = max(0, abs(x_robot - x_center) - 0.5 * side_length)
        dy = max(0, abs(y_robot - y_center) - 0.5 * side_length)
        dist_penalty = np.sqrt(dx**2 + dy**2)
        return self.weight * dist_penalty

    def gradient_x(self, x, u):
        grad_x = elementwise_grad(self.evaluate, 0)
        return grad_x(x, u)

    def hessian_x(self, x, u):
        hessian_x_fn = hessian(self.evaluate, 0)
        return hessian_x_fn(x, u)

class TrialCost:
    def __init__(self, d_threshold=0.5):
        self.d_threshold = 0.5

    def evaluate(self, x, u):
        dist = x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2
        return dist

    def gradient_x(self, x, u):
        grad_x = elementwise_grad(self.evaluate, 0)
        return grad_x(x, u)

    def hessian_x(self, x, u):
        hessian_x_fn = hessian(self.evaluate, 0)
        return hessian_x_fn(x, u)

class OverallCost:
    def __init__(self, subsystem_cost_functions):
        self.subsystem_cost_functions = subsystem_cost_functions

    def evaluate(self, x, u):
        total_cost = 0.0
        for subsystem_cost in self.subsystem_cost_functions:
            total_cost += subsystem_cost.evaluate(x, u)
            # print(subsystem_cost.evaluate(x, u))
        return total_cost

    def gradient_x(self, x, u):
        grad_x = elementwise_grad(self.evaluate, 0)
        return grad_x(x, u)

    def gradient_u(self, x, u):
        grad_u = elementwise_grad(self.evaluate, 1)
        return grad_u(x, u)

    def hessian_x(self, x, u):
        hessian_x_fn = hessian(self.evaluate, 0)
        return hessian_x_fn(x, u)

    def hessian_u(self, x, u):
        hessian_u_fn = hessian(self.evaluate, 1)
        return hessian_u_fn(x, u)

def trial():
    trial_cost = TrialCost()
    overall_cost = OverallCost([ProximityCost(idx1=0, idx2=1)])

    x_example = np.array([1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    u_example = np.array([1, 1])

    total_cost = overall_cost.evaluate(x_example, u_example)
    gradient_x = overall_cost.gradient_x(x_example, u_example)
    hessian_x = overall_cost.hessian_x(x_example, u_example)
    gradient_u = overall_cost.gradient_u(x_example, u_example)
    hessian_u = overall_cost.hessian_u(x_example, u_example)

    print("Total Cost:", total_cost)
    print("Gradient with respect to x:", gradient_x)
    print("Hessian with respect to x:", hessian_x)
    print("Gradient with respect to u:", gradient_u)
    print("Hessian with respect to u:", hessian_u)

# Example usage
# trial()
