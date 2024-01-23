import argparse
import numpy as np
import networkx as nx
from coopucb2 import coopucb2
import matplotlib.pyplot as plt
from graph_optimization import (fastest_averaging_constant_weight,
                                fdla_weights_symmetric, fmmc_weights,
                                lmsc_weights, max_degree_weights,
                                metropolis_hastings_weights)
from fdpg import signal_smoothness, nodal_distance, fdpg

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--network', type=str, default='tadpole')
	args = parser.parse_args()

	np.set_printoptions(threshold=np.inf, linewidth=np.inf)

	num_runs = 1
	num_arms = 10
	num_timesteps = 500
	bandit_true_means = np.random.normal(0, 1, (num_runs, num_arms))

	adjacency_matrix = np.load(f'data/saved_networks/{args.network}_adj.npy')
	incidence_matrix = np.asarray(nx.linalg.graphmatrix.incidence_matrix(nx.from_numpy_array(adjacency_matrix), oriented=True).todense())
	num_agents = adjacency_matrix.shape[0]
	adjacency_with_self = adjacency_matrix + np.eye(num_agents)

	# _, weight_matrix, _ = fmmc_weights(incidence_matrix)
	# print(weight_matrix)
	weight_matrix = np.array([
		[0   , 0.33, 0   , 0   ],
		[0.33, 0   , 0.33, 0.33],
		[0   , 0.33, 0   , 0.66],
		[0   , 0.33, 0.66, 0   ],
	])
	print(weight_matrix)

	regret, s_n_error, percent_optimal_action, arm_pulled = coopucb2(bandit_true_means, num_timesteps, weight_matrix)

	print(f'signal smoothness: {signal_smoothness(arm_pulled[0], adjacency_matrix)}')

	E = nodal_distance(arm_pulled[0], adjacency_matrix)
	iu = np.triu_indices(num_agents, k=1)
	e = E[iu]
	print(E)
	pred_w_k = fdpg(e.reshape(e.shape[0], 1), 2000, 0.3, 0.001, num_agents, 0.001)
	print(f'pred_w_k: {pred_w_k}')

	regret = np.cumsum(regret, axis=2)

	mean_regret = np.mean(regret, axis=0)
	mean_s_n_error = np.mean(s_n_error, axis=0)
	mean_percent_optimal_action = np.mean(percent_optimal_action, axis=0)

	fig, axs = plt.subplots(4, 1, figsize=(8, 12))
	ax_regret, ax_s_n_error, ax_percent_optimal_action, ax_arm_pulled = axs
	fig.suptitle(f'CoopUCB2 on {args.network}')

	ax_regret.set_title('Regret')
	ax_regret.set_xlabel('Timestep')
	ax_regret.set_ylabel('Regret')
	for agent_idx, agent_regret in enumerate(mean_regret):
		ax_regret.plot(agent_regret)
	mean_agent_regret = np.mean(mean_regret, axis=0)
	ax_regret.plot(mean_agent_regret, color='black', linestyle='--')

	ax_s_n_error.set_title('s/n Error')
	ax_s_n_error.set_xlabel('Timestep')
	ax_s_n_error.set_ylabel('Regret')
	for agent_idx, agent_s_n_error in enumerate(mean_s_n_error):
		ax_s_n_error.plot(agent_s_n_error)
	mean_agent_s_n_error = np.mean(mean_s_n_error, axis=0)
	ax_s_n_error.plot(mean_agent_s_n_error, color='black', linestyle='--')

	ax_percent_optimal_action.set_title('Percent optimal action')
	ax_percent_optimal_action.set_xlabel('Timestep')
	ax_percent_optimal_action.set_ylabel('Regret')
	for agent_idx, agent_percent_optimal_action in enumerate(mean_percent_optimal_action):
		ax_percent_optimal_action.plot(agent_percent_optimal_action)
	mean_agent_percent_optimal_action = np.mean(mean_percent_optimal_action, axis=0)
	ax_percent_optimal_action.plot(mean_agent_percent_optimal_action, label='Mean', color='black', linestyle='--')

	ax_arm_pulled.set_title('Arm picked in run 0')
	ax_arm_pulled.set_xlabel('Timestep')
	ax_arm_pulled.set_ylabel('Regret')
	for agent_idx, agent_arm_pulled in enumerate(arm_pulled[0]):
		ax_arm_pulled.plot(agent_arm_pulled, label=f'Agent {agent_idx}')

	ax_regret.grid()
	ax_s_n_error.grid()
	ax_percent_optimal_action.grid()
	ax_arm_pulled.grid()

	lgd = fig.legend(loc='center', bbox_to_anchor=(0.5, -0.01), ncols=num_agents + 1)
	plt.tight_layout()
	plt.savefig(f'data/img/pdf/{args.network}_graph_id_custom.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')