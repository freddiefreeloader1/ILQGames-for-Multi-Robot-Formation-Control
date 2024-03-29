import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import block_diag

from solve_lq_problem import solve_lq_game
from Diff_robot import UnicycleRobot
from Costs import ProximityCost, OverallCost, ReferenceCost

dt = 0.1
HORIZON = 3.0
TIMESTEPS = int(HORIZON / dt)


x0_1 = [-2.0, -2.0, 0.0, 0.0]
x0_2 = [2.0, 2.0, 3.14, 0.0]
x0_mp = x0_1 + x0_2
x_ref_1 = np.array([2, 2, 0, 0])
x_ref_2 = np.array([-2, -2, 0, 0])
xref_mp = np.concatenate((x_ref_1, x_ref_2))

x_traj_1 = [x0_1[0]]
y_traj_1 = [x0_1[1]]
x_traj_2 = [x0_2[0]]
y_traj_2 = [x0_2[1]]
heading_1 = [x0_1[2]]
heading_2 = [x0_2[2]]

robot1 = UnicycleRobot(x0_1, dt)
robot2 = UnicycleRobot(x0_2, dt)

ProximityCost1 = ProximityCost(1.7, 0, 1, 2)
ProximityCost2 = ProximityCost(1.7, 1, 0, 2)
ReferenceCost1 = ReferenceCost(0, xref_mp)
ReferenceCost2 = ReferenceCost(1, xref_mp)

overall_cost_1 = OverallCost([ReferenceCost1, ProximityCost1])
overall_cost_2 = OverallCost([ReferenceCost2, ProximityCost2])

u1_1 = [0.0] * TIMESTEPS
u1_2 = [0.0] * TIMESTEPS
u2_1 = [0.0] * TIMESTEPS
u2_2 = [0.0] * TIMESTEPS

Q1 = overall_cost_1.hessian_x(x0_mp, [0]*8)
Q2 = overall_cost_2.hessian_x(x0_mp, [0]*8)
Q1s = [Q1] * TIMESTEPS
Q2s = [Q2] * TIMESTEPS

l1 = overall_cost_1.gradient_x(x0_mp, [0]*8)
l2 = overall_cost_2.gradient_x(x0_mp, [0]*8)
l1s = [l1] * TIMESTEPS
l2s = [l2] * TIMESTEPS

R11 = np.eye(2)*3
R11s = [R11] * TIMESTEPS
R12 = np.zeros((2, 2))
R12s = [R12] * TIMESTEPS
R21 = np.zeros((2, 2))
R21s = [R21] * TIMESTEPS
R22 = np.eye(2)*3
R22s = [R22] * TIMESTEPS



us_1 = np.zeros((TIMESTEPS, 2))
us_2 = np.zeros((TIMESTEPS, 2))

total_time_steps = 0

while ((np.abs(robot1.state[0].item() - x_ref_1[0]) > 1e-2 or np.abs(robot1.state[1].item() - x_ref_1[1]) > 1e-2 or np.abs(robot2.state[0].item() - x_ref_2[0]) > 1e-2 or np.abs(robot2.state[1].item() - x_ref_2[1]) > 1e-2)) and (total_time_steps < 200):
    # Step 1: linearize the system around the operating point
    _, _, A_traj_1, B_traj_1 = robot1.linearize_dynamics_along_trajectory(u1_1, u1_2, dt)
    _, _, A_traj_2, B_traj_2 = robot2.linearize_dynamics_along_trajectory(u2_1, u2_2, dt)

    B_traj_1 = [np.concatenate((B, np.zeros((4, 2))), axis=0) for B in B_traj_1]
    B_traj_2 = [np.concatenate((np.zeros((4, 2)), B), axis=0) for B in B_traj_2]
    

    A_traj_mp = [block_diag(A1, A2) for A1, A2 in zip(A_traj_1, A_traj_2)]
    # Step 2: solve the LQ game
    [Ps_1, Ps_2], [alphas_1, alphas_2] = solve_lq_game(A_traj_mp, [B_traj_1,B_traj_2], [Q1s, Q2s], [l1s, l2s], [[R11s, R12s], [R21s, R22s]])

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

    # update the Q and l values
    states1 = robot1.state.detach().numpy().tolist()  
    states2 = robot2.state.detach().numpy().tolist()
    for ii in range(TIMESTEPS):
        states1 = robot1.integrate_dynamics_for_given_state(states1, us_1[ii][0], us_1[ii][1], dt) 
        states2 = robot2.integrate_dynamics_for_given_state(states2, us_2[ii][0], us_2[ii][1],dt)
        states = states1 + states2
        Q1s[ii] = overall_cost_1.hessian_x(states, us_1[ii] + us_2[ii])
        Q2s[ii] = overall_cost_2.hessian_x(states, us_1[ii] + us_2[ii])
        l1s[ii] = overall_cost_1.gradient_x(states, us_1[ii] + us_2[ii])
        l2s[ii] = overall_cost_2.gradient_x(states, us_1[ii] + us_2[ii])

    x_traj_1.append(robot1.state[0].item())
    y_traj_1.append(robot1.state[1].item())
    x_traj_2.append(robot2.state[0].item())
    y_traj_2.append(robot2.state[1].item())
    heading_1.append(robot1.state[2].item())
    heading_2.append(robot2.state[2].item())
    total_time_steps += 1
    print(total_time_steps)

plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(True)

for kk in range(total_time_steps):    
    ax.clear()
    ax.grid(True)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.plot(x_traj_1[kk], y_traj_1[kk], 'ro', label='Robot 1', markersize=25)
    ax.plot(x_traj_2[kk], y_traj_2[kk], 'bo', label='Robot 2', markersize=25)
    # put an direction arrow based on the third state of the robot on the dot
    ax.arrow(x_traj_1[kk], y_traj_1[kk], 0.3 * np.cos(heading_1[kk]), 0.3 * np.sin(heading_1[kk]), head_width=0.1)
    ax.arrow(x_traj_2[kk], y_traj_2[kk], 0.3 * np.cos(heading_2[kk]), 0.3 * np.sin(heading_2[kk]), head_width=0.1)
    plt.pause(0.01)
    # fig.canvas.draw()
    time.sleep(0.01)
    plt.show()
    
plt.ioff()
