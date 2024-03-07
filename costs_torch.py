import torch
import numpy as np

class ProximityCost:
    def __init__(self, d_threshold=0.5, idx1=0, idx2=0, weight=1.0):
        self.d_threshold = d_threshold
        self.idx1 = idx1
        self.idx2 = idx2
        self.weight = weight

    def evaluate(self, x, u):
        dist = torch.sqrt((x[4 * self.idx1] - x[4 * self.idx2])**2 + (x[4 * self.idx1 + 1] - x[4 * self.idx2 + 1])**2)
        return 0.0 if dist > self.d_threshold else self.weight * (self.d_threshold - dist)

    def gradient_x(self, x, u):
        grad_x = torch.autograd.grad(self.evaluate(x, u), x, create_graph=True)[0]
        return grad_x.detach().numpy()

class ReferenceCost:
    def __init__(self, idx=0, x_ref=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), weight=1.0):
        self.idx = idx
        self.x_ref = x_ref
        self.weight = weight

    def evaluate(self, x, u):
        dist = torch.sqrt((x[4 * self.idx] - self.x_ref[4 * self.idx])**2 + (x[4 * self.idx + 1] - self.x_ref[4 * self.idx + 1])**2 + (x[4 * self.idx + 2] - self.x_ref[4 * self.idx + 2])**2 + (x[4 * self.idx + 3] - self.x_ref[4 * self.idx + 3])**2)
        return dist * self.weight

    def gradient_x(self, x, u):
        grad_x = torch.autograd.grad(self.evaluate(x, u), x, create_graph=True)[0]
        return grad_x.detach().numpy()

class InputCost:
    def __init__(self, idx = 0, weight=1.0):
        self.weight = weight
        self.idx = idx

    def evaluate(self, x, u):
        return self.weight * (3 * u[0]**2 + 2 * u[1]**2)

    def gradient_u(self, x, u):
        grad_u = torch.autograd.grad(self.evaluate(x, u), u, create_graph=True)[0]
        return grad_u.detach().numpy()

class WallCost:
    def __init__(self, idx = 0, weight=1.0):
        self.idx = idx
        self.weight = weight

    def evaluate(self, x, u):
        x_robot = x[4 * self.idx]
        y_robot = x[4 * self.idx + 1]

        side_length = 7.0
        x_center = 0.0
        y_center = 0.0

        dx = torch.max(torch.tensor(0.0), torch.abs(x_robot - x_center) - 0.5 * side_length)
        dy = torch.max(torch.tensor(0.0), torch.abs(y_robot - y_center) - 0.5 * side_length)

        dist_penalty = torch.sqrt(dx**2 + dy**2)

        return self.weight * dist_penalty

    def gradient_x(self, x, u):
        grad_x = torch.autograd.grad(self.evaluate(x, u), x, create_graph=True)[0]
        return grad_x.detach().numpy()

class TrialCost:
    def __init__(self, d_threshold=0.5):
        self.d_threshold = 0.5

    def evaluate(self, x, u):
        dist = x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2
        return dist

    def gradient_x(self, x, u):
        grad_x = torch.autograd.grad(self.evaluate(x, u), x, create_graph=True)[0]
        return grad_x.detach().numpy()

class OverallCost:
    def __init__(self, subsystem_cost_functions):
        self.subsystem_cost_functions = subsystem_cost_functions

    def evaluate(self, x, u):
        total_cost = 0.0
        for subsystem_cost in self.subsystem_cost_functions:
            total_cost += subsystem_cost.evaluate(x, u)
        return total_cost

    def gradient_x(self, x, u):
        grad_x = torch.autograd.grad(self.evaluate(x, u), x, create_graph=True)[0]
        return grad_x.detach().numpy()

    def gradient_u(self, x, u):
        grad_u = torch.autograd.grad(self.evaluate(x, u), u, create_graph=True)[0]
        return grad_u.detach().numpy()

    def hessian_x(self, x, u):
        hess_x = torch.autograd.functional.hessian(self.evaluate, (x, u))
        return hess_x

    def hessian_u(self, x, u):
        hess_u = torch.autograd.functional.hessian(self.evaluate, (x, u))
        return hess_u



