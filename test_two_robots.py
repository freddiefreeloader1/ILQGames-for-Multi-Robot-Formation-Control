import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import block_diag

from solve_lq_problem import solve_lq_game
from Diff_robot import UnicycleRobot

dt = 0.1
HORIZON = 10.0
TIMESTEPS = int(HORIZON / dt)


x0_1 = [3.0, 2.0, 3.14, 0.0]
x0_2 = [-2.0, -2.0, 0.0, 0.0]

x_traj_1 = [x0_1[0]]
y_traj_1 = [x0_1[1]]
x_traj_2 = [x0_2[0]]
y_traj_2 = [x0_2[1]]
heading_1 = [x0_1[2]]
heading_2 = [x0_2[2]]

robot1 = UnicycleRobot(x0_1)
robot2 = UnicycleRobot(x0_2)


u1_1 = [0.0] * TIMESTEPS
u1_2 = [0.0] * TIMESTEPS
u2_1 = [0.0] * TIMESTEPS
u2_2 = [0.0] * TIMESTEPS

Q1 = np.diag([4.0, 4.0, 0.0, 0.0])
Q2 = np.diag([4.0, 4.0, 0.0, 0.0])
Q1s = [Q1] * TIMESTEPS
Q2s = [Q2] * TIMESTEPS

l1 = np.zeros((4, 1))
l2 = np.zeros((4, 1))
l1s = [l1] * TIMESTEPS
l2s = [l2] * TIMESTEPS

R11 = np.eye(2)
R11s = [R11] * TIMESTEPS
R12 = np.zeros((2, 2))
R12s = [R12] * TIMESTEPS
R21 = np.zeros((2, 2))
R21s = [R21] * TIMESTEPS
R22 = np.eye(2)
R22s = [R22] * TIMESTEPS

Q1_concat = [block_diag(Q1, np.zeros((4, 4))) for Q1 in Q1s]
Q2_concat = [block_diag(np.zeros((4, 4)), Q2) for Q2 in Q2s]
l1s = [np.concatenate((l1, np.zeros((4, 1))), axis=0) for l1 in l1s]
l2s = [np.concatenate((np.zeros((4, 1)), l2), axis=0) for l2 in l2s]

us_1 = np.zeros((TIMESTEPS, 2))
us_2 = np.zeros((TIMESTEPS, 2))

x_ref_1 = np.array([2, 0, 0, 0])
x_ref_2 = np.array([-1, 0, 0, 0])

for t in range(200):
    # Step 1: linearize the system around the operating point
    _, _, A_traj_1, B_traj_1 = robot1.linearize_dynamics_along_trajectory(u1_1, u1_2, dt)
    _, _, A_traj_2, B_traj_2 = robot2.linearize_dynamics_along_trajectory(u2_1, u2_2, dt)

    B_traj_1 = [np.concatenate((B, np.zeros((4, 2))), axis=0) for B in B_traj_1]
    B_traj_2 = [np.concatenate((np.zeros((4, 2)), B), axis=0) for B in B_traj_2]
    

    A_traj_mp = [block_diag(A1, A2) for A1, A2 in zip(A_traj_1, A_traj_2)]
    # Step 2: solve the LQ game
    [Ps_1, Ps_2], [alphas_1, alphas_2] = solve_lq_game(A_traj_mp, [B_traj_1,B_traj_2], [Q1_concat, Q2_concat], [l1s, l2s], [[R11s, R12s], [R21s, R22s]])

    # Step 3: Update the control inputs
    for ii in range(TIMESTEPS):
        us_1[ii, :] = -np.transpose(alphas_1[ii]) - Ps_1[ii][1][0:4] @ (robot1.state.detach().numpy() - x_ref_1)
        us_2[ii, :] = -np.transpose(alphas_2[ii]) - Ps_2[ii][1][4:8] @ (robot2.state.detach().numpy() - x_ref_2)

    # u1_1, u1_2, u2_1, and u2_2 are the first and second columns of us_1 and us_2,
    # make sure to reshape them to be of shape (TIMESTEPS, 1) but in list form
    u1_1 = us_1[:, 0].tolist()
    u1_2 = us_1[:, 1].tolist()
    u2_1 = us_2[:, 0].tolist()
    u2_2 = us_2[:, 1].tolist()

    # Update the robot's state
    robot1.integrate_dynamics(us_1[0][0], us_1[0][1], dt)
    robot2.integrate_dynamics(us_2[0][0], us_2[0][1], dt)

    x_traj_1.append(robot1.state[0].item())
    y_traj_1.append(robot1.state[1].item())
    x_traj_2.append(robot2.state[0].item())
    y_traj_2.append(robot2.state[1].item())
    heading_1.append(robot1.state[2].item())
    heading_2.append(robot2.state[2].item())

plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(True)

for kk in range(200):
    ax.clear()
    ax.grid(True)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.plot(x_traj_1[:kk + 1], y_traj_1[:kk + 1], 'ro', label='Robot 1')
    ax.plot(x_traj_2[:kk + 1], y_traj_2[:kk + 1], 'bo', label='Robot 2')
    # put an direction arrow based on the third state of the robot on the dot
    ax.arrow(x_traj_1[kk], y_traj_1[kk], 0.3 * np.cos(heading_1[kk]), 0.3 * np.sin(heading_1[kk]), head_width=0.1)
    ax.arrow(x_traj_2[kk], y_traj_2[kk], 0.3 * np.cos(heading_2[kk]), 0.3 * np.sin(heading_2[kk]), head_width=0.1)
    plt.pause(0.01)
    fig.canvas.draw()
    time.sleep(0.01)

plt.ioff()
plt.show()
