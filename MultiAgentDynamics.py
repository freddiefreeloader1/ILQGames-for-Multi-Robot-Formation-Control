from scipy.linalg import block_diag
from Costs import ProximityCost, OverallCost, ReferenceCost, WallCost
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

    def define_costs_lists(self):
        ref_cost_list = [[] for _ in range(len(self.agent_list))]
        prox_cost_list = [[] for _ in range(len(self.agent_list))]
        wall_cost_list = [[] for _ in range(len(self.agent_list))]
        overall_cost_list = [[] for _ in range(len(self.agent_list))]

        for i, agent in enumerate(self.agent_list):
            ref_cost_list[i].append(ReferenceCost(0.5, i, self.xref_mp))

        for i in range(len(self.agent_list)):
            for j in range(len(self.agent_list)):
                if i != j:
                    prox_cost_list[i].append(ProximityCost(0.6, i, j, 1.0))

        for i in range(len(self.agent_list)):
            wall_cost_list[i].append(WallCost(i, 4.0))

        for i in range(len(self.agent_list)):
            # add the reference cost and the proximity cost to the overall cost list
            cost_list = ref_cost_list[i] + prox_cost_list[i] + wall_cost_list[i]
            overall_cost_list[i].append(OverallCost(cost_list))
        return overall_cost_list
    
    def get_control_vector(self):
        us = []
        for agent in self.agent_list:
            us.append(np.zeros((self.TIMESTEPS, 2)))
        return us

    def compute_control_vector(self, Ps, alphas):
        for i, agent in enumerate(self.agent_list):
            for ii in range(self.TIMESTEPS):
                self.us[i][ii, :] = -np.transpose(alphas[i][ii]) - Ps[i][ii][1][4*i:4*(i+1)] @ (agent.state.detach().numpy() - agent.xref)
        return None

    def integrate_dynamics(self):
        for i, agent in enumerate(self.agent_list):
            agent.integrate_dynamics(self.us[i][0][0], self.us[i][0][1], self.dt)
        return None

    def reshape_control_inputs(self):
        reshaped_inputs = []

        for i in range(self.num_agents):
            robot_inputs_1 = [self.us[i][t][0] for t in range(self.TIMESTEPS)]
            robot_inputs_2 = [self.us[i][t][1] for t in range(self.TIMESTEPS)]
            reshaped_inputs.append([robot_inputs_1, robot_inputs_2])

        return reshaped_inputs

    def get_control_cost_matrix(self):
        R_eye = np.eye(2)*4
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