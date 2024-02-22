import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import block_diag

from solve_lq_problem import solve_lq_game
from Diff_robot import UnicycleRobot
from Costs import ProximityCost, OverallCost, ReferenceCost

dt = 0.1
HORIZON = 3
TIMESTEPS = int(HORIZON / dt)


x0_1 = [-2.0, -2.0, 0.0, 0.0]
x0_2 = [2.0, 2.0, 3.14, 0.0]
x0_3 = [3.0, -2.0, 3.14, 0.0]
x0_4 = [-2.0, 3.0, 0.0, 0.0]
x0_5 = [0.0, 1.0, 0.0, 0.0]
x0_6 = [1.0, 0.0, 0.0, 0.0]

x0_mp = x0_1 + x0_2 + x0_3 + x0_4 + x0_5 + x0_6

x_ref_1 = np.array([0, 0, 0, 0])
x_ref_2 = np.array([0, -1, 0, 0])
x_ref_3 = np.array([0, 1, 0, 0])
x_ref_4 = np.array([1, 0, 0, 0])
x_ref_5 = np.array([-1, 0, 0, 0])
x_ref_6 = np.array([0, -2, 0, 0])

xref_mp = np.concatenate((x_ref_1, x_ref_2, x_ref_3, x_ref_4, x_ref_5, x_ref_6))

x_traj_1 = [x0_1[0]]
y_traj_1 = [x0_1[1]]
x_traj_2 = [x0_2[0]]
y_traj_2 = [x0_2[1]]
x_traj_3 = [x0_3[0]]
y_traj_3 = [x0_3[1]]
x_traj_4 = [x0_4[0]]
y_traj_4 = [x0_4[1]]
x_traj_5 = [x0_5[0]]
y_traj_5 = [x0_5[1]]
x_traj_6 = [x0_6[0]]
y_traj_6 = [x0_6[1]]

heading_1 = [x0_1[2]]
heading_2 = [x0_2[2]]
heading_3 = [x0_3[2]]
heading_4 = [x0_4[2]]
heading_5 = [x0_5[2]]
heading_6 = [x0_6[2]]


robot1 = UnicycleRobot(x0_1[0], x0_1[1], x0_1[2], x0_1[3], dt)
robot2 = UnicycleRobot(x0_2[0], x0_2[1], x0_2[2], x0_2[3], dt)
robot3 = UnicycleRobot(x0_3[0], x0_3[1], x0_3[2], x0_3[3], dt)
robot4 = UnicycleRobot(x0_4[0], x0_4[1], x0_4[2], x0_4[3], dt)
robot5 = UnicycleRobot(x0_5[0], x0_5[1], x0_5[2], x0_5[3], dt)
robot6 = UnicycleRobot(x0_6[0], x0_6[1], x0_6[2], x0_6[3], dt)

prox_cost_list = [[] for _ in range(6)]
# Ensure prox_cost_list has enough elements
for i in range(6):
    while len(prox_cost_list[i]) < 6:
        prox_cost_list[i].append(ProximityCost(0.4, i, i, 1))

ReferenceCost1 = ReferenceCost(0.5, 0, xref_mp)
ReferenceCost2 = ReferenceCost(0.5, 1, xref_mp)
ReferenceCost3 = ReferenceCost(0.5, 2, xref_mp)
ReferenceCost4 = ReferenceCost(0.5, 3, xref_mp)
ReferenceCost5 = ReferenceCost(0.5, 4, xref_mp)
ReferenceCost6 = ReferenceCost(0.5, 5, xref_mp)
overall_cost_1 = OverallCost([ReferenceCost1, prox_cost_list[0][1], prox_cost_list[0][2], prox_cost_list[0][3], prox_cost_list[0][4], prox_cost_list[0][5]])
overall_cost_2 = OverallCost([ReferenceCost2, prox_cost_list[1][0], prox_cost_list[1][2], prox_cost_list[1][3], prox_cost_list[1][4], prox_cost_list[1][5]])
overall_cost_3 = OverallCost([ReferenceCost3, prox_cost_list[2][0], prox_cost_list[2][1], prox_cost_list[2][3], prox_cost_list[2][4], prox_cost_list[2][5]])
overall_cost_4 = OverallCost([ReferenceCost4, prox_cost_list[3][0], prox_cost_list[3][1], prox_cost_list[3][2], prox_cost_list[3][4], prox_cost_list[3][5]])
overall_cost_5 = OverallCost([ReferenceCost5, prox_cost_list[4][0], prox_cost_list[4][1], prox_cost_list[4][2], prox_cost_list[4][3], prox_cost_list[4][5]])
overall_cost_6 = OverallCost([ReferenceCost6, prox_cost_list[5][0], prox_cost_list[5][1], prox_cost_list[5][2], prox_cost_list[5][3], prox_cost_list[5][4]])


""" overall_cost_1 = OverallCost([ReferenceCost1])
overall_cost_2 = OverallCost([ReferenceCost2])
overall_cost_3 = OverallCost([ReferenceCost3])
overall_cost_4 = OverallCost([ReferenceCost4])
overall_cost_5 = OverallCost([ReferenceCost5])
overall_cost_6 = OverallCost([ReferenceCost6]) """

u1_1 = [0.0] * TIMESTEPS
u1_2 = [0.0] * TIMESTEPS
u2_1 = [0.0] * TIMESTEPS
u2_2 = [0.0] * TIMESTEPS
u3_1 = [0.0] * TIMESTEPS
u3_2 = [0.0] * TIMESTEPS
u4_1 = [0.0] * TIMESTEPS
u4_2 = [0.0] * TIMESTEPS
u5_1 = [0.0] * TIMESTEPS
u5_2 = [0.0] * TIMESTEPS
u6_1 = [0.0] * TIMESTEPS
u6_2 = [0.0] * TIMESTEPS


Q1 = overall_cost_1.hessian_x(x0_mp, [0]*24)
Q2 = overall_cost_2.hessian_x(x0_mp, [0]*24)
Q3 = overall_cost_3.hessian_x(x0_mp, [0]*24)
Q4 = overall_cost_4.hessian_x(x0_mp, [0]*24)
Q5 = overall_cost_5.hessian_x(x0_mp, [0]*24)
Q6 = overall_cost_6.hessian_x(x0_mp, [0]*24)

Q1s = [Q1] * TIMESTEPS
Q2s = [Q2] * TIMESTEPS
Q3s = [Q3] * TIMESTEPS
Q4s = [Q4] * TIMESTEPS
Q5s = [Q5] * TIMESTEPS
Q6s = [Q6] * TIMESTEPS
Qs = [Q1s, Q2s, Q3s, Q4s, Q5s, Q6s]
l1 = overall_cost_1.gradient_x(x0_mp, [0]*24)
l2 = overall_cost_2.gradient_x(x0_mp, [0]*24)
l3 = overall_cost_3.gradient_x(x0_mp, [0]*24)
l4 = overall_cost_4.gradient_x(x0_mp, [0]*24)
l5 = overall_cost_5.gradient_x(x0_mp, [0]*24)
l6 = overall_cost_6.gradient_x(x0_mp, [0]*24)

l1s = [l1] * TIMESTEPS
l2s = [l2] * TIMESTEPS
l3s = [l3] * TIMESTEPS
l4s = [l4] * TIMESTEPS
l5s = [l5] * TIMESTEPS
l6s = [l6] * TIMESTEPS
ls  = [l1s, l2s, l3s, l4s, l5s, l6s]

R_eye = np.eye(2) * 3
R_zeros = np.zeros((2, 2))

R_matrices = [R_eye.copy() for _ in range(TIMESTEPS)]
Z_matrices = [R_zeros.copy() for _ in range(TIMESTEPS)]

R11s, R22s, R33s, R44s, R55s, R66s = R_matrices, R_matrices, R_matrices, R_matrices, R_matrices, R_matrices
R12s, R13s, R14s, R15s, R16s = Z_matrices, Z_matrices, Z_matrices, Z_matrices, Z_matrices
R21s, R23s, R24s, R25s, R26s = Z_matrices, Z_matrices, Z_matrices, Z_matrices, Z_matrices
R31s, R32s, R34s, R35s, R36s = Z_matrices, Z_matrices, Z_matrices, Z_matrices, Z_matrices
R41s, R42s, R43s, R45s, R46s = Z_matrices, Z_matrices, Z_matrices, Z_matrices, Z_matrices
R51s, R52s, R53s, R54s, R56s = Z_matrices, Z_matrices, Z_matrices, Z_matrices, Z_matrices
R61s, R62s, R63s, R64s, R65s = Z_matrices, Z_matrices, Z_matrices, Z_matrices, Z_matrices

Rs = [[R11s, R12s, R13s, R14s, R15s, R16s], [R21s, R22s, R23s, R24s, R25s, R26s], [R31s, R32s, R33s, R34s, R35s, R36s], [R41s, R42s, R43s, R44s, R45s, R46s], [R51s, R52s, R53s, R54s, R55s, R56s], [R61s, R62s, R63s, R64s, R65s, R66s]]

us_1 = np.zeros((TIMESTEPS, 2))
us_2 = np.zeros((TIMESTEPS, 2))
us_3 = np.zeros((TIMESTEPS, 2))
us_4 = np.zeros((TIMESTEPS, 2))
us_5 = np.zeros((TIMESTEPS, 2))
us_6 = np.zeros((TIMESTEPS, 2))

total_time_steps = 0

while (total_time_steps < 100):
    # Step 1: linearize the system around the operating point
    _, _, A_traj_1, B_traj_1 = robot1.linearize_dynamics_along_trajectory(u1_1, u1_2, dt)
    _, _, A_traj_2, B_traj_2 = robot2.linearize_dynamics_along_trajectory(u2_1, u2_2, dt)
    _, _, A_traj_3, B_traj_3 = robot3.linearize_dynamics_along_trajectory(u3_1, u3_2, dt)
    _, _, A_traj_4, B_traj_4 = robot4.linearize_dynamics_along_trajectory(u4_1, u4_2, dt)
    _, _, A_traj_5, B_traj_5 = robot5.linearize_dynamics_along_trajectory(u5_1, u5_2, dt)
    _, _, A_traj_6, B_traj_6 = robot6.linearize_dynamics_along_trajectory(u6_1, u6_2, dt)

    B_traj_1 = [np.concatenate((B, np.zeros((20, 2))), axis=0) for B in B_traj_1]
    B_traj_2 = [np.concatenate((np.zeros((4, 2)), B, np.zeros((16,2))), axis=0) for B in B_traj_2]
    B_traj_3 = [np.concatenate((np.zeros((8, 2)), B, np.zeros((12,2))), axis=0) for B in B_traj_3]
    B_traj_4 = [np.concatenate((np.zeros((12, 2)), B, np.zeros((8,2))), axis=0) for B in B_traj_4]
    B_traj_5 = [np.concatenate((np.zeros((16, 2)), B, np.zeros((4,2))), axis=0) for B in B_traj_5]
    B_traj_6 = [np.concatenate((np.zeros((20, 2)), B), axis=0) for B in B_traj_6]
    
    Bs = [B_traj_1, B_traj_2, B_traj_3, B_traj_4, B_traj_5, B_traj_6]
    
    

    A_traj_mp = [block_diag(*A_list) for A_list in zip(A_traj_1, A_traj_2, A_traj_3, A_traj_4, A_traj_5, A_traj_6)]
    # Step 2: solve the LQ game
    [Ps_1, Ps_2, Ps_3, Ps_4, Ps_5, Ps_6], [alphas_1, alphas_2, alphas_3, alphas_4, alphas_5, alphas_6] = solve_lq_game(A_traj_mp, Bs, Qs, ls, Rs)

    # Step 3: Update the control inputs
    for ii in range(TIMESTEPS):
        us_1[ii, :] = -np.transpose(alphas_1[ii]) - Ps_1[ii][1][0:4] @ (robot1.state.detach().numpy() - x_ref_1)
        us_2[ii, :] = -np.transpose(alphas_2[ii]) - Ps_2[ii][1][4:8] @ (robot2.state.detach().numpy() - x_ref_2)
        us_3[ii, :] = -np.transpose(alphas_3[ii]) - Ps_3[ii][1][8:12] @ (robot3.state.detach().numpy() - x_ref_3)
        us_4[ii, :] = -np.transpose(alphas_4[ii]) - Ps_4[ii][1][12:16] @ (robot4.state.detach().numpy() - x_ref_4)
        us_5[ii, :] = -np.transpose(alphas_5[ii]) - Ps_5[ii][1][16:20] @ (robot5.state.detach().numpy() - x_ref_5)
        us_6[ii, :] = -np.transpose(alphas_6[ii]) - Ps_6[ii][1][20:24] @ (robot6.state.detach().numpy() - x_ref_6)

    # u1_1, u1_2, u2_1, and u2_2 are the first and second columns of us_1 and us_2,
    # make sure to reshape them to be of shape (TIMESTEPS, 1) but in list form
    u1_1 = us_1[:, 0].tolist()
    u1_2 = us_1[:, 1].tolist()
    u2_1 = us_2[:, 0].tolist()
    u2_2 = us_2[:, 1].tolist()
    u3_1 = us_3[:, 0].tolist()
    u3_2 = us_3[:, 1].tolist()
    u4_1 = us_4[:, 0].tolist()
    u4_2 = us_4[:, 1].tolist()
    u5_1 = us_5[:, 0].tolist()
    u5_2 = us_5[:, 1].tolist()
    u6_1 = us_6[:, 0].tolist()
    u6_2 = us_6[:, 1].tolist()


    # Update the robot's state
    robot1.integrate_dynamics(us_1[0][0], us_1[0][1], dt)
    robot2.integrate_dynamics(us_2[0][0], us_2[0][1], dt)
    robot3.integrate_dynamics(us_3[0][0], us_3[0][1], dt)
    robot4.integrate_dynamics(us_4[0][0], us_4[0][1], dt)
    robot5.integrate_dynamics(us_5[0][0], us_5[0][1], dt)
    robot6.integrate_dynamics(us_6[0][0], us_6[0][1], dt)

    # update the Q and l values
    states1 = robot1.state.detach().numpy().tolist()  
    states2 = robot2.state.detach().numpy().tolist()
    states3 = robot3.state.detach().numpy().tolist()
    states4 = robot4.state.detach().numpy().tolist()
    states5 = robot5.state.detach().numpy().tolist()
    states6 = robot6.state.detach().numpy().tolist()

    for ii in range(TIMESTEPS):
        states1 = robot1.integrate_dynamics_for_given_state(states1, us_1[ii][0], us_1[ii][1],dt) 
        states2 = robot2.integrate_dynamics_for_given_state(states2, us_2[ii][0], us_2[ii][1],dt)
        states3 = robot3.integrate_dynamics_for_given_state(states3, us_3[ii][0], us_3[ii][1],dt)
        states4 = robot4.integrate_dynamics_for_given_state(states4, us_4[ii][0], us_4[ii][1],dt)
        states5 = robot5.integrate_dynamics_for_given_state(states5, us_5[ii][0], us_5[ii][1],dt)
        states6 = robot6.integrate_dynamics_for_given_state(states6, us_6[ii][0], us_6[ii][1],dt)

        states = states1 + states2 + states3 + states4 + states5 + states6
        Q1s[ii] = overall_cost_1.hessian_x(states, us_1[ii] + us_2[ii])
        Q2s[ii] = overall_cost_2.hessian_x(states, us_1[ii] + us_2[ii])
        Q3s[ii] = overall_cost_3.hessian_x(states, us_1[ii] + us_2[ii])
        Q4s[ii] = overall_cost_4.hessian_x(states, us_1[ii] + us_2[ii])
        Q5s[ii] = overall_cost_5.hessian_x(states, us_1[ii] + us_2[ii])
        Q6s[ii] = overall_cost_6.hessian_x(states, us_1[ii] + us_2[ii])

        l1s[ii] = overall_cost_1.gradient_x(states, us_1[ii] + us_2[ii])
        l2s[ii] = overall_cost_2.gradient_x(states, us_1[ii] + us_2[ii])
        l3s[ii] = overall_cost_3.gradient_x(states, us_1[ii] + us_2[ii])
        l4s[ii] = overall_cost_4.gradient_x(states, us_1[ii] + us_2[ii])
        l5s[ii] = overall_cost_5.gradient_x(states, us_1[ii] + us_2[ii])
        l6s[ii] = overall_cost_6.gradient_x(states, us_1[ii] + us_2[ii])

    x_traj_1.append(robot1.state[0].item())
    y_traj_1.append(robot1.state[1].item())
    x_traj_2.append(robot2.state[0].item())
    y_traj_2.append(robot2.state[1].item())
    x_traj_3.append(robot3.state[0].item())
    y_traj_3.append(robot3.state[1].item())
    x_traj_4.append(robot4.state[0].item())
    y_traj_4.append(robot4.state[1].item())
    x_traj_5.append(robot5.state[0].item())
    y_traj_5.append(robot5.state[1].item())
    x_traj_6.append(robot6.state[0].item())
    y_traj_6.append(robot6.state[1].item())

    heading_1.append(robot1.state[2].item())
    heading_2.append(robot2.state[2].item())
    heading_3.append(robot3.state[2].item())
    heading_4.append(robot4.state[2].item())
    heading_5.append(robot5.state[2].item())
    heading_6.append(robot6.state[2].item())

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
    ax.plot(x_traj_3[kk], y_traj_3[kk], 'go', label='Robot 3', markersize=25)
    ax.plot(x_traj_4[kk], y_traj_4[kk], 'yo', label='Robot 4', markersize=25)
    ax.plot(x_traj_5[kk], y_traj_5[kk], 'mo', label='Robot 5', markersize=25)
    ax.plot(x_traj_6[kk], y_traj_6[kk], 'co', label='Robot 6', markersize=25)

    # put an direction arrow based on the third state of the robot on the dot
    ax.arrow(x_traj_1[kk], y_traj_1[kk], 0.3 * np.cos(heading_1[kk]), 0.3 * np.sin(heading_1[kk]), head_width=0.1)
    ax.arrow(x_traj_2[kk], y_traj_2[kk], 0.3 * np.cos(heading_2[kk]), 0.3 * np.sin(heading_2[kk]), head_width=0.1)
    ax.arrow(x_traj_3[kk], y_traj_3[kk], 0.3 * np.cos(heading_3[kk]), 0.3 * np.sin(heading_3[kk]), head_width=0.1)
    ax.arrow(x_traj_4[kk], y_traj_4[kk], 0.3 * np.cos(heading_4[kk]), 0.3 * np.sin(heading_4[kk]), head_width=0.1)
    ax.arrow(x_traj_5[kk], y_traj_5[kk], 0.3 * np.cos(heading_5[kk]), 0.3 * np.sin(heading_5[kk]), head_width=0.1)
    ax.arrow(x_traj_6[kk], y_traj_6[kk], 0.3 * np.cos(heading_6[kk]), 0.3 * np.sin(heading_6[kk]), head_width=0.1)

    plt.pause(0.01)
    # fig.canvas.draw()
    time.sleep(0.01)
    plt.show()
    
plt.ioff()
