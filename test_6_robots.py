import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import block_diag

from solve_lq_problem import solve_lq_game
from Diff_robot import UnicycleRobot
from Costs import ProximityCost, OverallCost, ReferenceCost
from MultiAgentDynamics import MultiAgentDynamics

dt = 0.1
HORIZON = 3.0
TIMESTEPS = int(HORIZON / dt)


x0_1 = [-2.0, -2.0, 0.0, 0.0]
x0_2 = [2.0, 2.0, 3.14, 0.0]
x0_3 = [3.0, -2.0, 3.14, 0.0]
x0_4 = [-2.0, 3.0, 0.0, 0.0]
x0_5 = [0.0, 1.0, 0.0, 0.0]
x0_6 = [1.0, 0.0, 0.0, 0.0]


x_ref_1 = np.array([0, 0, 0, 0])
x_ref_2 = np.array([0, -2, 0, 0])
x_ref_3 = np.array([0, 2, 0, 0])
x_ref_4 = np.array([2, 0, 0, 0])
x_ref_5 = np.array([-2, 0, 0, 0])
x_ref_6 = np.array([0, -1, 0, 0])


robot1 = UnicycleRobot(x0_1, x_ref_1, dt)
robot2 = UnicycleRobot(x0_2, x_ref_2, dt)
robot3 = UnicycleRobot(x0_3, x_ref_3, dt)
robot4 = UnicycleRobot(x0_4, x_ref_4, dt)
robot5 = UnicycleRobot(x0_5, x_ref_5, dt)
robot6 = UnicycleRobot(x0_6, x_ref_6, dt)


# mp_dynamics = MultiAgentDynamics([robot1, robot2, robot3, robot4, robot5, robot6], dt, HORIZON)
mp_dynamics = MultiAgentDynamics([robot1, robot2, robot3, robot4], dt, HORIZON)

costs = mp_dynamics.define_costs_lists()

x_traj = [[] for _ in range(mp_dynamics.num_agents)]
y_traj = [[] for _ in range(mp_dynamics.num_agents)]
headings = [[] for _ in range(mp_dynamics.num_agents)]

ls = []
Qs = []

for i in range(mp_dynamics.num_agents):
    Qs.append([costs[i][0].hessian_x(mp_dynamics.x0_mp, [0]*mp_dynamics.num_agents*4)]*mp_dynamics.TIMESTEPS)
    ls.append([costs[i][0].gradient_x(mp_dynamics.x0_mp, [0]*mp_dynamics.num_agents*4)]*mp_dynamics.TIMESTEPS)

Rs = mp_dynamics.get_control_cost_matrix()

total_time_steps = 0
reshaped_inputs = mp_dynamics.reshape_control_inputs()
flag = 0
# define ksi as [[[state1],[state2]]...,[[input1], [input2]]]

ksi = [[[] for agent in mp_dynamics.agent_list], [[0,0]*mp_dynamics.TIMESTEPS for agent in mp_dynamics.agent_list]]

try:
    while (flag == 0):


        As, Bs = mp_dynamics.get_linearized_dynamics(reshaped_inputs)

        # update the Q and l values
        states = [robot.state.detach().numpy().tolist() for robot in mp_dynamics.agent_list]
        # Initialize Qs and ls lists
        Qs = [[] for _ in range(mp_dynamics.num_agents)]
        ls = [[] for _ in range(mp_dynamics.num_agents)]

        # Iterate over timesteps
        for ii in range(mp_dynamics.TIMESTEPS):
            # Integrate dynamics for each robot
            for i, robot in enumerate(mp_dynamics.agent_list):
                states[i] = robot.integrate_dynamics_for_given_state(states[i], mp_dynamics.us[i][ii][0], mp_dynamics.us[i][ii][1], mp_dynamics.dt)
                ksi[0][i].append(states[i])
            # Concatenate states of all robots
            concatenated_states = [val for sublist in states for val in sublist]

            # Compute Hessian and gradients for each robot
            for i, robot in enumerate(mp_dynamics.agent_list):
                Qs[i].append(costs[i][0].hessian_x(concatenated_states, mp_dynamics.us[i][ii]))
                ls[i].append(costs[i][0].gradient_x(concatenated_states, mp_dynamics.us[i][ii]))

        # Step 2: solve the LQ game
        Ps, alphas = solve_lq_game(As, Bs, Qs, ls, Rs)

        # Step 3: Update the control inputs


        # Update the robot's state
        mp_dynamics.integrate_dynamics()

        mp_dynamics.compute_control_vector(Ps, alphas, ksi)

        for i, agent in enumerate(mp_dynamics.agent_list):
            ksi[1][i] = mp_dynamics.compute_control_vector(Ps, alphas, ksi)[i]

        reshaped_inputs = mp_dynamics.reshape_control_inputs()

        dist = []
    
        for robot in mp_dynamics.agent_list:
            dist.append(np.sqrt(np.linalg.norm(robot.state[0:2].detach().numpy() - robot.xref[0:2])))
        
        print(dist)
        # check all elements of dist are smaller than 
        if all(d < 0.3 for d in dist):
            flag = 1

        # Append the states to the trajectory lists
        for i, robot in enumerate(mp_dynamics.agent_list):
            x_traj[i].append(robot.state[0].item())
            y_traj[i].append(robot.state[1].item())
            headings[i].append(robot.state[2].item())

        total_time_steps += 1
        print(total_time_steps)

except KeyboardInterrupt:
    pass 

plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(True)
colors = ['ro', 'go', 'bo', 'co', 'mo', 'yo']

for kk in range(total_time_steps):    
    ax.clear()
    ax.grid(True)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    for i in range(mp_dynamics.num_agents):
        ax.plot(x_traj[i][kk], y_traj[i][kk], colors[i], label=f'Robot {i}', markersize=25)
        ax.arrow(x_traj[i][kk], y_traj[i][kk], 0.3 * np.cos(headings[i][kk]), 0.3 * np.sin(headings[i][kk]), head_width=0.1)

    plt.pause(0.01)
    time.sleep(0.01)
    plt.show()
    
plt.ioff()
