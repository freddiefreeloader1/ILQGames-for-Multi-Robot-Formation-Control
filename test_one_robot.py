import numpy as np
import matplotlib.pyplot as plt

from solve_lq_problem import solve_lq_game
from Diff_robot import UnicycleRobot
import torch
import matplotlib.pyplot as plt
import time


dt = 0.1
HORIZON = 10.0
TIMESTEPS = int(HORIZON / dt)


x0 = [3.0, 2.0, 1.0, 0.0]
x_traj = [x0[0]]
y_traj = [x0[1]]
robot = UnicycleRobot(x0[0], x0[1], x0[2], x0[3])

u1 = [0.0]*TIMESTEPS
u2 = [0.0]*TIMESTEPS

Q1 = np.diag([20.0, 25.0, 4.0, 2.0])
Q1s = [Q1] * TIMESTEPS
l1 = np.zeros((4, 1))
l1s = [l1] * TIMESTEPS
R1 = np.eye(2)
R1s = [R1] * TIMESTEPS

us = np.zeros((TIMESTEPS, 2))
x_ref = np.array([1, 0, 0, 0])
for t in range(200):
    # Step 1: Start with initial us with horizon
    _, _, A_traj, B_traj = robot.linearize_dynamics_along_trajectory(u1, u2, dt)

    # Step 2: Linearize the system with respect to this and get the Atraj and Btraj
    [Ps], [alphas] = solve_lq_game(A_traj, [B_traj], [Q1s], [l1s], [[R1s]])

    # Step 3: Solve the LQ problem using the function
    for ii in range(TIMESTEPS):
        us[ii, :] = - np.transpose(alphas[ii]) - Ps[ii] @ (robot.state.detach().numpy() - x_ref)

    # u1 and u2 are the first and second columns of us, make sure to reshape them to be of shape (TIMESTEPS, 1) but in list form
    u1 = us[:, 0].tolist()
    u2 = us[:, 1].tolist()

    # Update the robot's state
    robot.integrate_dynamics(us[0][0], us[0][1], dt)
    print(robot.state.detach().numpy())
    x_traj.append(robot.state[0].item())
    y_traj.append(robot.state[1].item())
    

plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.grid(True)

for kk in range(200):
    ax.plot(x_traj[:kk + 1], y_traj[:kk + 1], 'ro')  # Plot the trajectory up to the current step
    plt.pause(0.01)
    fig.canvas.draw()
    time.sleep(0.01)

plt.ioff()
plt.show()





