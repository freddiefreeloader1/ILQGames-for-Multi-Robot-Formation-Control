import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import block_diag
import torch

from solve_lq_problem import solve_lq_game
from Diff_robot import UnicycleRobot
from MultiAgentDynamics import MultiAgentDynamics



dt = 0.25
HORIZON = 6.0
TIMESTEPS = int(HORIZON / dt)
scenerio = "overtaking"

if scenerio == "intersection":
    x0_1 = [-2.0, -2.0, 0.0, 1.0]
    x0_2 = [-2.0, 2.0, 0.0, 1.0]
    x0_3 = [0.0, 4.0, -np.pi/2, 3.0]
    x0_4 = [-2.0, 3.0, 0.0, 0.0]
    x0_5 = [0.0, 1.0, 0.0, 0.0]
    x0_6 = [1.0, 0.0, 0.0, 0.0]


    x_ref_1 = np.array([3, -2, 0, 0])
    x_ref_2 = np.array([3, 2, 0, 0])
    x_ref_3 = np.array([0, -3, 0, 0])
    x_ref_4 = np.array([2, 0, 0, 0])
    x_ref_5 = np.array([-2, 0, 0, 0])
    x_ref_6 = np.array([0, -1, 0, 0])

if scenerio == "overtaking":
    x0_1 = [-3.0, -2.0, 0.0, 1]
    x0_2 = [-3.0, 2.0, 0.0, 1]
    x0_3 = [-3.0, 0.0, 0, 1]
    x0_4 = [-2.0, 3.0, 0.0, 0.0]
    x0_5 = [0.0, 1.0, 0.0, 0.0]
    x0_6 = [1.0, 0.0, 0.0, 0.0]


    x_ref_1 = np.array([3, 2, 0, 0])
    x_ref_2 = np.array([3, -2, 0, 0])
    x_ref_3 = np.array([3, 0, 0, 0])
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
mp_dynamics = MultiAgentDynamics([robot1, robot2, robot3], dt, HORIZON)

costs = mp_dynamics.define_costs_lists()

x_traj = [[] for _ in range(mp_dynamics.num_agents)]
y_traj = [[] for _ in range(mp_dynamics.num_agents)]
headings = [[] for _ in range(mp_dynamics.num_agents)]

ls = []
Qs = []
x0_tensor = torch.tensor(mp_dynamics.x0_mp, requires_grad=True)
u0_tensor = torch.tensor([0.0]*mp_dynamics.num_agents*4, requires_grad=True)
for i in range(mp_dynamics.num_agents):
    Qs.append([costs[i][0].hessian_x(x0_tensor, u0_tensor)]*mp_dynamics.TIMESTEPS)
    ls.append([costs[i][0].gradient_x(x0_tensor, u0_tensor)]*mp_dynamics.TIMESTEPS)

Rs = mp_dynamics.get_control_cost_matrix()

total_time_steps = 0
reshaped_inputs = mp_dynamics.reshape_control_inputs()
flag = 0
# define ksi as [[[state1],[state2]]...,[[input1], [input2]]]
last_points = None
current_points = None

ksi = [[[] for agent in mp_dynamics.agent_list], [[0,0]*mp_dynamics.TIMESTEPS for agent in mp_dynamics.agent_list]]
u1 = [[0]*mp_dynamics.TIMESTEPS for agent in mp_dynamics.agent_list]
u2 = [[0]*mp_dynamics.TIMESTEPS for agent in mp_dynamics.agent_list]
xs = [[0]*mp_dynamics.TIMESTEPS for agent in mp_dynamics.agent_list]

# define xs as sinusoidal trajectory for each robot 
for i, agent in enumerate(mp_dynamics.agent_list):
    xs[i] = [agent.x0 for _ in range(mp_dynamics.TIMESTEPS)]
    for t in range(mp_dynamics.TIMESTEPS):
        if t != 0:
            x = xs[i][0][0] + t * dt * 1
            y = xs[i][0][1] 
            xs[i][t] = [x, y, 0, 1]

# plot the xs first 
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(True)
colors = ['ro', 'go', 'bo', 'co', 'mo', 'yo']
for kk in range(mp_dynamics.TIMESTEPS):
    ax.clear()
    ax.grid(True)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    for i in range(mp_dynamics.num_agents):
        ax.plot(xs[i][kk][0], xs[i][kk][1], colors[i], label=f'Robot {i}', markersize=25)
        ax.arrow(xs[i][kk][0], xs[i][kk][1], 0.3 * np.cos(xs[i][kk][2]), 0.3 * np.sin(xs[i][kk][2]), head_width=0.1)

    plt.pause(0.01)
    time.sleep(0.01)
    plt.show()

plt.ioff()

# close the plot
plt.close()



prev_control_inputs = np.zeros((mp_dynamics.num_agents, mp_dynamics.TIMESTEPS, 2))
# make prev_control_inputs sinusoidal
for i in range(mp_dynamics.num_agents):
    for t in range(mp_dynamics.TIMESTEPS):
        prev_control_inputs[i][t] = [0, 0]

control_inputs = np.zeros((mp_dynamics.num_agents, mp_dynamics.TIMESTEPS, 2))
total_costs = []

try:
    while (flag == 0):
        start = time.time()

        last_points = current_points
        
        # integrate the dynamics
        if total_time_steps != 0:
            xs = mp_dynamics.integrate_dynamics_for_initial_mp(u1, u2, mp_dynamics.dt)

        current_points = xs

        # get the linearized dynamics
        As, Bs = mp_dynamics.get_linearized_dynamics_for_initial_state(xs,u1,u2)

        Qs = [[] for _ in range(mp_dynamics.num_agents)]
        ls = [[] for _ in range(mp_dynamics.num_agents)]
        Rs = [[[] for _ in range(mp_dynamics.num_agents)] for _ in range(mp_dynamics.num_agents)]

        # Iterate over timesteps
        total_costs.append([])
        '''cost_start = time.time()'''
        for ii in range(mp_dynamics.TIMESTEPS):
            concatenated_states = np.concatenate([state[ii] for state in xs])
            concatenated_states_t = torch.tensor(concatenated_states, requires_grad=True)
            prev_control_inputs_t = torch.tensor(prev_control_inputs, requires_grad=True)
            for i, robot in enumerate(mp_dynamics.agent_list):
                Qs[i].append(costs[i][0].hessian_x(concatenated_states_t, prev_control_inputs_t[i][ii])[0][0].detach().numpy())  
                ls[i].append(costs[i][0].gradient_x(concatenated_states_t, prev_control_inputs_t[i][ii]))
                Rs[i][i].append(costs[i][0].hessian_u(concatenated_states_t, prev_control_inputs_t[i][ii])[1][1].detach().numpy())
                total_costs[total_time_steps].append(costs[i][0].evaluate(concatenated_states_t, prev_control_inputs_t[i][ii]).detach().item())

        '''cost_end = time.time()
        print(f"Time taken for cost computation: {cost_end - cost_start}")'''

        for i in range(mp_dynamics.num_agents):
            for j in range(mp_dynamics.num_agents):
                if i != j:
                    Rs[i][j] = [np.zeros((2, 2)) for _ in range(mp_dynamics.TIMESTEPS)]    

        # sum the costs 
        total_costs[total_time_steps] = sum(total_costs[total_time_steps])
        Ps, alphas = solve_lq_game(As, Bs, Qs, ls, Rs)

        u1_array = np.array(u1)
        u2_array = np.array(u2)

        # Reshape u1 and u2
        u1_reshaped = np.reshape(u1_array, (u1_array.shape[0], u1_array.shape[1], 1))
        u2_reshaped = np.reshape(u2_array, (u2_array.shape[0], u2_array.shape[1], 1))

        # Combine u1_reshaped and u2_reshaped along the last axis to get (num_agents, timesteps, 2)
        control_inputs = np.concatenate((u1_reshaped, u2_reshaped), axis=-1)

        control_inputs = mp_dynamics.compute_control_vector_current(Ps, alphas, xs, current_points, prev_control_inputs)

        prev_control_inputs = control_inputs

        # get u1 and u2 from control_inputs
        u1 = control_inputs[:,:,0]
        u2 = control_inputs[:,:,1]
        
        flag = mp_dynamics.check_convergence(current_points, last_points)

        if flag == 1:
            for ii in range(mp_dynamics.TIMESTEPS):
                for i, agent in enumerate(mp_dynamics.agent_list):
                    x_traj[i].append(xs[i][ii][0])
                    y_traj[i].append(xs[i][ii][1])
                    headings[i].append(xs[i][ii][2])

        total_time_steps += 1
        end = time.time()
        print(f"Time taken for iteration {total_time_steps}: {end - start}")
        print(total_time_steps)

except KeyboardInterrupt:
    for ii in range(mp_dynamics.TIMESTEPS):
        for i, agent in enumerate(mp_dynamics.agent_list):
            x_traj[i].append(xs[i][ii][0])
            y_traj[i].append(xs[i][ii][1])
            headings[i].append(xs[i][ii][2])
    for ii in range(len(total_costs)):
        if type(total_costs[ii]) is list: 
            total_costs[ii] = sum(total_costs[ii])
    
# plot costs
plt.figure()
plt.plot(total_costs)
plt.xlabel('Time Step')
plt.ylabel('Cost')
plt.title('Costs over Time')
plt.show()


plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(True)
colors = ['ro', 'go', 'bo', 'co', 'mo', 'yo']
for kk in range(mp_dynamics.TIMESTEPS):    
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


# plot the whole state trajectorys
plt.figure()
for i in range(mp_dynamics.num_agents):
    plt.plot(x_traj[i], y_traj[i], label=f'Robot {i}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('State Trajectories')
plt.legend()
plt.show()
# plot the whole state trajectorys
plt.figure()
for i in range(mp_dynamics.num_agents):
    plt.plot(headings[i], label=f'Robot {i}')
plt.xlabel('Timestep')
plt.ylabel('Heading')
plt.title('Heading Trajectories')
plt.legend()
plt.show()
