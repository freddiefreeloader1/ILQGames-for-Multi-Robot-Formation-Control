import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class UnicycleRobot:

    def __init__(self, x0, xref, dt=0.1):
        self.x0 = x0
        self.state = torch.tensor([x0[0], x0[1], x0[2], x0[3]], requires_grad=True)  # (x, y, theta, v)
        self.xref = xref
        self.dt = dt

    def dynamics(self, u1, u2):
        x, y, theta, v  = self.state
        x_dot = v * torch.cos(theta)
        y_dot = v * torch.sin(theta)
        theta_dot = torch.tensor(u1)  # Convert theta_dot to a tensor
        v_dot = torch.tensor(u2)  # Convert v_dot to a tensor
        return torch.stack([x_dot, y_dot, theta_dot, v_dot])

    def dynamics_for_given_state(self, state, u1, u2):
        x, y, theta, v  = state
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = u1  # Convert theta_dot to a tensor
        v_dot = u2
        return [x_dot, y_dot, theta_dot, v_dot]

    def integrate_dynamics_clone(self, u1, u2, dt):
        x_dot = self.dynamics(u1, u2)
        updated_state = self.state + self.dt * x_dot.detach().clone()
        return updated_state.data  

    def integrate_dynamics_for_given_state(self, state, u1, u2, dt):
        x_dot = self.dynamics_for_given_state(state, u1, u2)
        updated_state = [self.dt*i for i in x_dot] 
        updated_state= [i + j for i, j in zip(state, updated_state)]
        return updated_state

    def integrate_dynamics_for_initial_state(self, state, u1s, u2s, dt, TIMESTEP):
        states = []
        for i in range(TIMESTEP):
            state = self.integrate_dynamics_for_given_state(state, u1s[i], u2s[i], dt)
            states.append(state)
        return states

    def integrate_dynamics(self, u1, u2, dt):
        # Integrate forward in time using Euler method
        x_dot = self.dynamics(u1, u2)
        updated_state = self.state + self.dt * x_dot.detach().clone()
        self.state.data =  updated_state.data  # Update state without creating a view

    def linearize_autograd(self, x_torch, u_torch):
        
        updated_state = self.integrate_dynamics_clone(u_torch[0], u_torch[1], self.dt)

        A = np.array([[0, 0, updated_state[3].detach().numpy() * -torch.sin(updated_state[2]).item(), torch.cos(updated_state[2]).item()], 
                     [0, 0, updated_state[3].item() * torch.cos(updated_state[2]).item(), torch.sin(updated_state[2]).item()],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])

        B = np.array([[0, 0],
                     [0, 0],
                     [1, 0],
                     [0, 1]])
        return A, B

    def linearize(self, x, u):
        
        A = np.array([[0, 0, x[3] * -np.sin(x[2]), np.cos(x[2])], 
                    [0, 0, x[3] * np.cos(x[2]), np.sin(x[2])],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])

        B = np.array([[0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1]])
        return A, B
    
    def linearize_discrete(self, A, B, dt):

        A_d = scipy.linalg.expm(A * dt)
        B_d = np.linalg.pinv(A) @ (scipy.linalg.expm(A * dt) - np.eye(4)) @ B
        # make the values of A_d and B_d to be 0 if they are very close to 0
        A_d[np.abs(A_d) < 1e-10] = 0
        B_d[np.abs(B_d) < 1e-10] = 0

        return A_d, B_d
        
    def linearize_dynamics_along_trajectory(self, u1_traj, u2_traj, dt):
        # Linearize dynamics along the trajectory
        num_steps = len(u2_traj)

        A_list = []
        B_list = []
        A_d_list = []
        B_d_list = []

        for t in range(num_steps):
            # Integrate forward in time
            updated_state = self.integrate_dynamics_clone(u1_traj[t], u2_traj[t], self.dt)

            # Linearize at the current state and control
            x_torch = updated_state.clone().detach().requires_grad_(True)
            u_torch = torch.tensor([u1_traj[t], u2_traj[t]], requires_grad=True)

            A, B = self.linearize_autograd(x_torch, u_torch)
            A_d, B_d = self.linearize_discrete(A, B, dt)
            A_d_list.append(A_d)
            B_d_list.append(B_d)
            A_list.append(A)
            B_list.append(B)
          
        return np.array(A_list), np.array(B_list), np.array(A_d_list), np.array(B_d_list)

    def linearize_dynamics_along_trajectory_for_states(self,states, u1_traj, u2_traj, dt):
        # Linearize dynamics along the trajectory
        num_steps = len(u2_traj)

        A_list = []
        B_list = []
        A_d_list = []
        B_d_list = []

        for t in range(num_steps):
            u = [u1_traj[t], u2_traj[t]]
            A, B = self.linearize(states[t], u)
            A_d, B_d = self.linearize_discrete(A, B, dt)
            A_d_list.append(A_d)
            B_d_list.append(B_d)
            A_list.append(A)
            B_list.append(B)
          
        return np.array(A_list), np.array(B_list), np.array(A_d_list), np.array(B_d_list)

# Example usage:
""" robot = UnicycleRobot(0.0,0.0,0.0,0.0)

# Define a trajectory of control inputs (v, theta_dot)
v_trajectory = [0.1]*3
theta_dot_trajectory = [0.1]*3

# Time step for integration
dt = 0.1

# Linearize dynamics along the trajectory
A_traj, B_traj, A_d_list, B_d_list = robot.linearize_dynamics_along_trajectory(v_trajectory, theta_dot_trajectory, dt)

for t in range(len(A_traj)):
    print(f"Time Step {t + 1}:")
    print("Linearization Matrix A:")
    print(A_traj[t])
    print("Linearization Matrix B:")
    print(B_traj[t])


    print("Discrete Linearization Matrix A_d:")
    print(A_d_traj[t])
    print("Discrete Linearization Matrix B_d:")
    print(B_d_traj[t])
    print("\n")



# Define the initial state of the robot
initial_state = torch.tensor([0.0, 0.0, 0.0, 0.0])

# Create a figure and axis for the animation
fig, ax = plt.subplots()

# Set the limits of the plot
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# Create a line object for the robot's trajectory
line, = ax.plot([], [], 'b')

# Function to update the animation at each frame
def update(frame):
    # Integrate forward in time using the discrete matrices
    if frame == 0:
        robot.state = initial_state.clone()
    else:
        A_d = torch.tensor(A_d_traj[frame - 1], dtype=torch.double)
        B_d = torch.tensor(B_d_traj[frame - 1], dtype=torch.double)
        robot.state = torch.matmul(A_d, robot.state.double()) + torch.matmul(B_d, torch.tensor([v_trajectory[frame - 1], theta_dot_trajectory[frame - 1]], dtype=torch.double))

    # Clear the previous plot
    ax.clear()

    # Plot the new robot position
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.scatter(robot.state[0].item(), robot.state[1].item(), c='r')

    return ax,

# ani = animation.FuncAnimation(fig, update, frames=len(A_d_traj) + 1, interval=dt, blit=True)

# plt.show()
 """
