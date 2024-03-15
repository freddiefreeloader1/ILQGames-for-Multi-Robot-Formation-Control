import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import block_diag

from solve_lq_problem import solve_lq_game
from Diff_robot_uncertainty import UnicycleRobotUncertain
from Costs import ProximityCost, OverallCost, ReferenceCost
from MultiAgentDynamics import MultiAgentDynamics



dt = 0.2
HORIZON = 4
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
    x0_1 = [-3.0, -2.0, 0.0, 1.1]
    x0_2 = [-3.0, 2.0, 0.0, 1]
    x0_3 = [-3.0, 0.0, 0, 0.8]
    x0_4 = [-2.0, 3.0, 0.0, 0.0]
    x0_5 = [0.0, 1.0, 0.0, 0.0]
    x0_6 = [1.0, 0.0, 0.0, 0.0]


    x_ref_1 = np.array([2, 2, 0, 0])
    x_ref_2 = np.array([2, -2, 0, 0])
    x_ref_3 = np.array([2, 0, 0, 0])
    x_ref_4 = np.array([2, 0, 0, 0])
    x_ref_5 = np.array([-2, 0, 0, 0])
    x_ref_6 = np.array([0, -1, 0, 0])

robot1 = UnicycleRobotUncertain(x0_1, x_ref_1, dt)
robot2 = UnicycleRobotUncertain(x0_2, x_ref_2, dt)
robot3 = UnicycleRobotUncertain(x0_3, x_ref_3, dt)
robot4 = UnicycleRobotUncertain(x0_4, x_ref_4, dt)
robot5 = UnicycleRobotUncertain(x0_5, x_ref_5, dt)
robot6 = UnicycleRobotUncertain(x0_6, x_ref_6, dt)


# mp_dynamics = MultiAgentDynamics([robot1, robot2, robot3, robot4, robot5, robot6], dt, HORIZON)
mp_dynamics = MultiAgentDynamics([robot1, robot2, robot3], dt, HORIZON)

costs = mp_dynamics.define_costs_lists(uncertainty=True)

x_traj = [[] for _ in range(mp_dynamics.num_agents)]
y_traj = [[] for _ in range(mp_dynamics.num_agents)]
headings = [[] for _ in range(mp_dynamics.num_agents)]

ls = []
Qs = []



Rs = mp_dynamics.get_control_cost_matrix()

total_time_steps = 0
reshaped_inputs = mp_dynamics.reshape_control_inputs()
flag = 0

last_points = None
current_points = None

u1 = [[0]*mp_dynamics.TIMESTEPS for agent in mp_dynamics.agent_list]
u2 = [[0]*mp_dynamics.TIMESTEPS for agent in mp_dynamics.agent_list]
xs = [[0]*mp_dynamics.TIMESTEPS for agent in mp_dynamics.agent_list]


for i, agent in enumerate(mp_dynamics.agent_list):
    xs[i] = [agent.x0 for _ in range(mp_dynamics.TIMESTEPS)]
    for t in range(mp_dynamics.TIMESTEPS):
        if t != 0:
            # the initial trajectort should be a straight line to reference
            x_dif = (agent.xref[0] - agent.x0[0])/mp_dynamics.TIMESTEPS
            y_dif = (agent.xref[1] - agent.x0[1])/mp_dynamics.TIMESTEPS
            heading = np.arctan2(y_dif, x_dif)
            xs[i][t] = [xs[i][t-1][0] + x_dif, xs[i][t-1][1] + y_dif, heading, agent.x0[3]]


prev_control_inputs = np.zeros((mp_dynamics.num_agents, mp_dynamics.TIMESTEPS, 2))
control_inputs = np.zeros((mp_dynamics.num_agents, mp_dynamics.TIMESTEPS, 2))
total_costs = []

mu = np.array([[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]])
phi = 1.2

Gs = np.empty((mp_dynamics.num_agents, mp_dynamics.TIMESTEPS, mp_dynamics.num_agents-1, 12), dtype=object)
qs = np.empty((mp_dynamics.num_agents, mp_dynamics.TIMESTEPS, mp_dynamics.num_agents-1), dtype=object)
rhos = np.empty((mp_dynamics.num_agents, mp_dynamics.TIMESTEPS, mp_dynamics.num_agents-1), dtype=object)

# make Gs all zeros
for i in range(mp_dynamics.num_agents):
    for t in range(mp_dynamics.TIMESTEPS):
        for j in range(mp_dynamics.num_agents-1):
            Gs[i][t][j] = np.zeros((1, 12))
            qs[i][t][j] = np.zeros(1)
            rhos[i][t][j] = 0.0


identity_size = 4

# Create a block diagonal matrix of size 4 * num_agents by 4 * TIMESTEPS
sigmas_block_diag = block_diag(*[np.eye(identity_size)*0.005 for _ in range(mp_dynamics.num_agents)])

sigmas = np.array([sigmas_block_diag for _ in range(mp_dynamics.TIMESTEPS)])

prox_cost_list = [[] for _ in range(len(mp_dynamics.agent_list))]
for i in range(len(mp_dynamics.agent_list)):
    for j in range(len(mp_dynamics.agent_list)):
        if i != j:
            prox_cost_list[i].append(ProximityCost(1.0, i, j, 5.0))


for i in range(mp_dynamics.num_agents):
    for t in range(mp_dynamics.TIMESTEPS):
        u1[i][t] = 0.5*np.sin(t*mp_dynamics.dt) 
        u1[i][t] = 0.0
        u2[i][t] = 0.0


xs = mp_dynamics.integrate_dynamics_for_initial_mp(u1, u2, mp_dynamics.dt, True)
current_points = xs
last_points = xs

Acs, Bcs, As, Bs = mp_dynamics.get_linearized_dynamics_for_initial_state(xs,u1,u2)

for ii in range(mp_dynamics.TIMESTEPS):
                sigmas[ii] = Acs[ii] @ sigmas[ii] @ Acs[ii].T

Gs, qs, rhos = mp_dynamics.get_Gs(xs, prox_cost_list, sigmas)

for i in range(mp_dynamics.num_agents):
    for t in range(mp_dynamics.TIMESTEPS):
        prev_control_inputs[i][t] = [u1[i][t], u2[i][t]]

lambdas = np.zeros((mp_dynamics.num_agents, mp_dynamics.num_agents-1))
Is = np.zeros((mp_dynamics.num_agents, mp_dynamics.num_agents-1))
Is = mu

# initialize the Ps with shape 3,50,2,12

Ps = np.zeros((mp_dynamics.num_agents, mp_dynamics.TIMESTEPS, 2, mp_dynamics.num_agents*4))

# initialize the alphas with shape 3,50,2
alphas = np.zeros((mp_dynamics.num_agents, mp_dynamics.TIMESTEPS, 2))

'''for i in range(mp_dynamics.num_agents):
    Qs.append([costs[i][0].hessian_x(mp_dynamics.x0_mp, [0]*mp_dynamics.num_agents*4)]*mp_dynamics.TIMESTEPS)
    ls.append([costs[i][0].gradient_x(mp_dynamics.x0_mp, [0]*mp_dynamics.num_agents*4)]*mp_dynamics.TIMESTEPS)'''

TOL_CC_ERROR = 1e-3
max_error = 100

total_costs = []
total_ref_costs = []
total_prox_costs = []
total_wall_costs = []
total_input_costs = []

try:
    while(max_error > TOL_CC_ERROR):
        # define errors list as agent * agent-1 as list
        errors = [[[] for _ in range(mp_dynamics.num_agents-1)] for _ in range(mp_dynamics.num_agents)]
        if Gs[0][0][0] is not None:
            for i, robot in enumerate(mp_dynamics.agent_list):
                for j in range(mp_dynamics.TIMESTEPS):
                    for k in range(mp_dynamics.num_agents-1):
                        concatenated_states = np.concatenate([state[j] for state in xs])
                        error = (Gs[i][j][k]@concatenated_states + qs[i][j][k] + rhos[i][j][k])
                        errors[i][k].append(error)
            max_errors = np.float32(np.max(np.array(errors), 2))
            max_error = np.max(max_errors)
        print(max_error)
        for i in range(mp_dynamics.num_agents):
            for j in range(mp_dynamics.num_agents-1):
                lambdas[i][j] = max(0,lambdas[i][j] + mu[i][j] * np.abs((0.70 - max_errors[i][j])))
                Is[i][j] = 0 if (0.7 - max_error < 0.0)&(lambdas[i][j] == 0) else mu[i][j]

        for i in range(mp_dynamics.num_agents):
            for j in range(mp_dynamics.num_agents-1):
                mu[i][j] *= phi
        flag = 0
        while (flag == 0):

            start = time.time()
            errors = [[[] for _ in range(mp_dynamics.num_agents-1)] for _ in range(mp_dynamics.num_agents)]
            if Gs[0][0][0] is not None:
                for i, robot in enumerate(mp_dynamics.agent_list):
                    for j in range(mp_dynamics.TIMESTEPS):
                        for k in range(mp_dynamics.num_agents-1):
                            concatenated_states = np.concatenate([state[j] for state in xs])
                            error = (Gs[i][j][k]@concatenated_states + qs[i][j][k] + rhos[i][j][k])
                            errors[i][k].append(error)
                max_errors = np.float32(np.max(np.array(errors), 2))
                max_error = np.max(max_errors)
            print('Max Error:', max_error)


           
            
            # integrate the dynamics
            

            xs, control_inputs = mp_dynamics.compute_op_point(Ps, alphas, current_points, prev_control_inputs, 0.02 , True)

            u1 = control_inputs[:,:,0]
            u2 = control_inputs[:,:,1]

            xs_real = mp_dynamics.integrate_dynamics_for_initial_mp(u1, u2, mp_dynamics.dt, True)

            last_points = current_points
            current_points = xs

            # get the linearized dynamics
            Acs, Bcs, As, Bs = mp_dynamics.get_linearized_dynamics_for_initial_state(xs,u1,u2)

            for ii in range(mp_dynamics.TIMESTEPS):
                sigmas[ii] = Acs[ii] @ sigmas[ii] @ Acs[ii].T 

            # get the linearized constraint matrices
            Gs, qs, rhos = mp_dynamics.get_Gs(xs, prox_cost_list, sigmas)

            Qs = [[] for _ in range(mp_dynamics.num_agents)]
            ls = [[] for _ in range(mp_dynamics.num_agents)]
            Rs = [[[] for _ in range(mp_dynamics.num_agents)] for _ in range(mp_dynamics.num_agents)]

            # Iterate over timesteps
            total_costs.append([])
            total_ref_costs.append([])
            total_prox_costs.append([])
            total_wall_costs.append([])
            total_input_costs.append([])

            for ii in range(mp_dynamics.TIMESTEPS):
                concatenated_states = np.concatenate([state[ii] for state in xs])
                for i, robot in enumerate(mp_dynamics.agent_list):
                    for k in range(len(mp_dynamics.agent_list)-1):
                        Qs[i].append(costs[i][0].hessian_x(concatenated_states, control_inputs[i][ii], Gs[i][ii][k], qs[i][ii][k], rhos[i][ii][k], lambdas[i][k], Is[i][k]))
                        ls[i].append(costs[i][0].gradient_x(concatenated_states, control_inputs[i][ii], Gs[i][ii][k], qs[i][ii][k], rhos[i][ii][k], lambdas[i][k], Is[i][k]))
                        Rs[i][i].append(costs[i][0].hessian_u(concatenated_states, control_inputs[i][ii]))
                        total_costs[total_time_steps].append(costs[i][0].evaluate(concatenated_states, control_inputs[i][ii], Gs[i][ii][k], qs[i][ii][k], rhos[i][ii][k], lambdas[i][k], Is[i][k]))
                        total_prox_costs[total_time_steps].append(costs[i][0].subsystem_cost_functions[1].evaluate(concatenated_states, Gs[i][ii][k], qs[i][ii][k], rhos[i][ii][k], lambdas[i][k]))
                        total_prox_costs[total_time_steps].append(costs[i][0].subsystem_cost_functions[3].evaluate(concatenated_states, Gs[i][ii][k], qs[i][ii][k], rhos[i][ii][k], lambdas[i][k]))
                        total_prox_costs[total_time_steps].append(costs[i][0].subsystem_cost_functions[2].evaluate(concatenated_states, Gs[i][ii][k], qs[i][ii][k], rhos[i][ii][k], Is[i][k]))
                        total_prox_costs[total_time_steps].append(costs[i][0].subsystem_cost_functions[4].evaluate(concatenated_states, Gs[i][ii][k], qs[i][ii][k], rhos[i][ii][k], Is[i][k]))
                        total_ref_costs[total_time_steps].append(costs[i][0].subsystem_cost_functions[0].evaluate(concatenated_states, control_inputs[i][ii]))
                        total_input_costs[total_time_steps].append(costs[i][0].subsystem_cost_functions[6].evaluate(concatenated_states, control_inputs[i][ii]))
                        total_wall_costs[total_time_steps].append(costs[i][0].subsystem_cost_functions[5].evaluate(concatenated_states, control_inputs[i][ii]))

            # sum the costs 
            for i in range(mp_dynamics.num_agents):
                for j in range(mp_dynamics.num_agents):
                    if i != j:
                        Rs[i][j] = [np.zeros((2, 2)) for _ in range(mp_dynamics.TIMESTEPS)]       

            total_costs[total_time_steps] = sum(total_costs[total_time_steps])
            total_prox_costs[total_time_steps] = sum(total_prox_costs[total_time_steps])
            total_ref_costs[total_time_steps] = sum(total_ref_costs[total_time_steps])
            total_input_costs[total_time_steps] = sum(total_input_costs[total_time_steps])
            total_wall_costs[total_time_steps] = sum(total_wall_costs[total_time_steps])


            Ps, alphas = solve_lq_game(As, Bs, Qs, ls, Rs)


            # Combine u1_reshaped and u2_reshaped along the last axis to get (num_agents, timesteps, 2)
            #control_inputs = np.concatenate((u1_reshaped, u2_reshaped), axis=-1)

            # control_inputs = mp_dynamics.compute_control_vector_current(Ps, alphas, xs, current_points, prev_control_inputs)

            prev_control_inputs = control_inputs

            # get u1 and u2 from control_inputs
            
            
            '''if total_time_steps > 0:
                flag = mp_dynamics.check_convergence(current_points, last_points)'''

            if flag == 1:
                for ii in range(mp_dynamics.TIMESTEPS):
                    for i, agent in enumerate(mp_dynamics.agent_list):
                        x_traj[i].append(xs[i][ii][0])
                        y_traj[i].append(xs[i][ii][1])
                        headings[i].append(xs[i][ii][2])

            total_time_steps += 1
            # print the iteration with text
            print(f'Iteration {total_time_steps}')
            
            end = time.time()
            print(f'Time: {end - start}')

except KeyboardInterrupt:
    for ii in range(mp_dynamics.TIMESTEPS):
        for i, agent in enumerate(mp_dynamics.agent_list):
            x_traj[i].append(xs_real[i][ii][0])
            y_traj[i].append(xs_real[i][ii][1])
            headings[i].append(xs_real[i][ii][2])
    for ii in range(len(total_costs)):
        if type(total_costs[ii]) is list: 
            total_costs[ii] = sum(total_costs[ii])
            total_prox_costs[ii] = sum(total_prox_costs[ii])
            total_ref_costs[ii] = sum(total_ref_costs[ii])
            total_input_costs[ii] = sum(total_input_costs[ii])
            total_wall_costs[ii] = sum(total_wall_costs[ii])


    
# plot costs
plt.figure()
plt.plot(total_costs)
plt.plot(total_prox_costs)
plt.plot(total_ref_costs)
plt.plot(total_input_costs)
plt.plot(total_wall_costs)

plt.legend(['Total Cost','Proximity Cost', 'Reference Cost', 'Input Cost', 'Wall Cost'])
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


plt.figure()
for i in range(mp_dynamics.num_agents):
    plt.plot(x_traj[i], y_traj[i], colors[i],  label=f'Robot {i}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('State Trajectories')
plt.legend()
plt.show()
