from scipy.linalg import block_diag
from Costs import ProximityCost, OverallCost, ReferenceCost
import numpy as np

class MultiAgentDynamics():
    def __init__(self, agent_list, dt):
        self.agent_list = agent_list
        self.dt = dt
        self.num_agents = len(agent_list)
        self.x0_mp = np.concatenate([agent.x0 for agent in agent_list])
        self.xref_mp = np.concatenate([agent.xref for agent in agent_list])

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

        # return As like this     A_traj_mp = [block_diag(*A_list) for A_list in zip(A_traj_1, A_traj_2, A_traj_3, A_traj_4, A_traj_5, A_traj_6)]
        As = [block_diag(*A_list) for A_list in zip(*A_traj_mp)]

        return As, Bs

    def define_costs_lists(self, agent_list, x_ref_mp):
        ref_cost_list = [[] for _ in range(len(agent_list))]
        prox_cost_list = [[] for _ in range(len(agent_list))]
        overall_cost_list = [[] for _ in range(len(agent_list))]

        for i, agent in enumerate(agent_list):
            ref_cost_list[i].append(ReferenceCost(0.5, i, x_ref_mp))

        for i in range(len(agent_list)):
            for j in range(len(agent_list)):
                if i != j:
                    prox_cost_list[i].append(ProximityCost(0.6, i, j, 1.0))

        for i in range(len(agent_list)):
            # add the reference cost and the proximity cost to the overall cost list
            cost_list = ref_cost_list[i] + prox_cost_list[i]
            overall_cost_list[i].append(OverallCost(cost_list))
        return overall_cost_list

    