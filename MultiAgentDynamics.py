from scipy.linalg import block_diag
from Costs import ProximityCost, OverallCost, ReferenceCost, WallCost, InputCost
import numpy as np

class MultiAgentDynamics():
    def __init__(self, agent_list, dt, HORIZON=3.0):
        self.agent_list = agent_list
        self.dt = dt
        self.num_agents = len(agent_list)
        self.x0_mp = np.concatenate([agent.x0 for agent in agent_list])
        self.xref_mp = np.concatenate([agent.xref for agent in agent_list])
        self.TIMESTEPS = int(HORIZON/dt)
        self.us = self.get_control_vector()

    def get_linearized_dynamics(self, u_list):
        A_traj_mp = []
        As = []
        Bs = []
        for i, agent in enumerate(self.agent_list):
            _, _, A_traj, B_traj = agent.linearize_dynamics_along_trajectory(u_list[i][0], u_list[i][1], self.dt)
            A_traj_mp.append(A_traj)
            if i == 0:
                B_traj = [np.concatenate((B, np.zeros((4 * (self.num_agents -1), 2))), axis=0) for B in B_traj]
            else:
                B_traj = [np.concatenate((np.zeros((4 * i, 2)), B, np.zeros((4 * (self.num_agents - i - 1), 2))), axis=0) for B in B_traj]   

            Bs.append(B_traj)

        As = [block_diag(*A_list) for A_list in zip(*A_traj_mp)]

        return As, Bs

    def get_linearized_dynamics_for_initial_state(self, x_states, u1, u2):
        A_traj_mp = []
        As = []
        Bs = []
        for i, agent in enumerate(self.agent_list):
            _, _, A_traj, B_traj = agent.linearize_dynamics_along_trajectory_for_states(x_states[i], u1[i], u2[i], self.dt)
            A_traj_mp.append(A_traj)
            if i == 0:
                B_traj = [np.concatenate((B, np.zeros((4 * (self.num_agents -1), 2))), axis=0) for B in B_traj]
            else:
                B_traj = [np.concatenate((np.zeros((4 * i, 2)), B, np.zeros((4 * (self.num_agents - i - 1), 2))), axis=0) for B in B_traj]   

            Bs.append(B_traj)

        As = [block_diag(*A_list) for A_list in zip(*A_traj_mp)]

        return As, Bs

    def define_costs_lists(self):
        ref_cost_list = [[] for _ in range(len(self.agent_list))]
        prox_cost_list = [[] for _ in range(len(self.agent_list))]
        wall_cost_list = [[] for _ in range(len(self.agent_list))]
        input_cost_list = [[] for _ in range(len(self.agent_list))]
        overall_cost_list = [[] for _ in range(len(self.agent_list))]

        for i, agent in enumerate(self.agent_list):
            ref_cost_list[i].append(ReferenceCost(i, self.xref_mp, 15.0))
            input_cost_list[i].append(InputCost(i, 8.0))

        for i in range(len(self.agent_list)):
            for j in range(len(self.agent_list)):
                if i != j:
                    prox_cost_list[i].append(ProximityCost(1.0, i, j, 3.0))

        for i in range(len(self.agent_list)):
            wall_cost_list[i].append(WallCost(i, 8.0))

        for i in range(len(self.agent_list)):
            # add the reference cost and the proximity cost to the overall cost list
            cost_list = ref_cost_list[i] + prox_cost_list[i] + wall_cost_list[i] + input_cost_list[i]
            overall_cost_list[i].append(OverallCost(cost_list))
        return overall_cost_list
    
    def get_control_vector(self):
        us = []
        for agent in self.agent_list:
            us.append(np.zeros((self.TIMESTEPS, 2)))
        return us

    def compute_control_vector(self, Ps, alphas, ksi = 0):
        for i, agent in enumerate(self.agent_list):
            for ii in range(self.TIMESTEPS):
                self.us[i][ii, :] = ksi[1][i][ii] - np.transpose(0.4*alphas[i][ii]) - Ps[i][ii][1][4*i:4*(i+1)] @ (agent.state.detach().numpy() - ksi[0][i][ii])
        return self.us

    def compute_control_vector_current(self, Ps, alphas, xs, current_x, u_prev):
        u_next = np.zeros((self.num_agents, self.TIMESTEPS, 2))
        if current_x is not None:
            for i, agent in enumerate(self.agent_list):
                for ii in range(self.TIMESTEPS):
                    # concatenate the states of all the robots
                    concatenated_states = np.concatenate([state[ii] for state in xs])
                    concatenated_states_current = np.concatenate([state[ii] for state in current_x])
                    u_next[i][ii] = u_prev[i][ii] - 0.01*alphas[i][ii] - Ps[i][ii] @ (concatenated_states - concatenated_states_current)
        else:
            for i, agent in enumerate(self.agent_list):
                for ii in range(self.TIMESTEPS):
                    u_next[i][ii] = u_prev[i][ii] - 0.01*alphas[i][ii]
        return u_next

    def integrate_dynamics(self):
        for i, agent in enumerate(self.agent_list):
            agent.integrate_dynamics(self.us[i][0][0], self.us[i][0][1], self.dt)
        return None

    def integrate_dynamics_for_initial_mp(self, u1, u2, dt):
        xs = [[agent.x0] for agent in self.agent_list]
        for i, agent in enumerate(self.agent_list):
            xs[i] = xs[i] + (agent.integrate_dynamics_for_initial_state(agent.x0, u1[i], u2[i], dt, self.TIMESTEPS))
        return xs

    def reshape_control_inputs(self):
        reshaped_inputs = []

        for i in range(self.num_agents):
            robot_inputs_1 = [self.us[i][t][0] for t in range(self.TIMESTEPS)]
            robot_inputs_2 = [self.us[i][t][1] for t in range(self.TIMESTEPS)]
            reshaped_inputs.append([robot_inputs_1, robot_inputs_2])

        return reshaped_inputs

    def get_control_cost_matrix(self):
        R_eye = np.array([[1, 0],[0, 25]])
        R_zeros = np.zeros((2, 2))

        # Initialize R_matrices and Z_matrices lists
        R_matrices = [R_eye.copy() for _ in range(self.TIMESTEPS)]
        Z_matrices = [R_zeros.copy() for _ in range(self.TIMESTEPS)]

        # Initialize Rs list based on the number of robots
        Rs = []
        for i in range(self.num_agents):
            R_terms = [R_matrices.copy() if i == j else Z_matrices.copy() for j in range(self.num_agents)]
            Rs.append(R_terms)

        return Rs

    def check_convergence(self, current_points, last_points):

        if last_points is None:
            return 0
        for i in range(len(current_points)):
            for j in range(len(current_points[i])):
                for k in range(len(current_points[i][j])):
                    if np.abs(np.array(current_points[i][j][k]) - np.array(last_points[i][j][k])) > 0.001:
                        return 0
        return 1