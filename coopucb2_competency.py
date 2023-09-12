import logging
from time import ctime, time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_pydot import graphviz_layout
from numba import njit, prange

from graph_optimization import (fastest_averaging_constant_weight,
                                fdla_weights_symmetric, fmmc_weights,
                                lmsc_weights, max_degree_weights,
                                metropolis_hastings_weights)


def run(bandit_true_means: np.ndarray, changes_at: np.ndarray, competencies: np.ndarray, num_timesteps: int, adjacency_with_self: np.ndarray, weight_matrix: np.ndarray, experiment_name: str, axs=None):
	if axs is None:
		raise ValueError('ax must be provided')

	# reset plt color cycle
	for _ax in axs:
		_ax.set_prop_cycle(None)

	ax_top, ax_bottom = axs

	logging.info(f'started {experiment_name} version')
	regret, percent_optimal_action = coopucb2(
		bandit_true_means,
		competencies,
		num_timesteps,
		weight_matrix,
	)

	regret = np.cumsum(regret, axis=2) # cumulative regret over time
	regret = np.mean(regret, axis=0) # average over runs
	percent_optimal_action = np.mean(percent_optimal_action, axis=0) # average over runs

	# np.save('data/data/changing_P_regret.npy', regret)
	# regret = np.load('data/data/changing_P_regret.npy')

	for agent_idx, agent_regret in enumerate(regret):
		ax_top.plot(agent_regret, label=f'agent {agent_idx + 1}', linestyle='-')

	regret = np.mean(regret, axis=0) # average over agents
	ax_top.plot(regret, label=f'team average', color='black', linestyle='--')

	ax_top.title.set_text(experiment_name)
	ax_top.grid()

	for agent_idx, agent_percent_optimal_action in enumerate(percent_optimal_action):
		ax_bottom.plot(agent_percent_optimal_action, label=f'agent {agent_idx + 1}', linestyle='-')

	percent_optimal_action = np.mean(percent_optimal_action, axis=0) # average over agents
	ax_bottom.plot(percent_optimal_action, label=f'team average', color='black', linestyle='--')

	ax_bottom.grid()

	logging.info(f'ended {experiment_name} version')


def main():
	logging.info(f'started script')
	# network = nx.complete_graph(5) # all-to-all
	# adjacency_matrix = nx.to_numpy_array(network)

	# house
	adjacency_matrix = np.array([
		[0, 1, 0, 0],
		[1, 0, 1, 1],
		[0, 1, 0, 1],
		[0, 1, 1, 0],
	])
	incidence_matrix = np.asarray(nx.linalg.graphmatrix.incidence_matrix(nx.from_numpy_array(adjacency_matrix), oriented=True).todense())
	num_agents = adjacency_matrix.shape[0]

	adjacency_with_self = adjacency_matrix + np.eye(num_agents)
	neighbor_count = np.sum(adjacency_with_self, axis=1)

	# weight_matrix = adjacency_with_self / neighbor_count # try this
	_, _, weight_matrix, _ = fastest_averaging_constant_weight(incidence_matrix)
	logging.info(f'incidence matrix:\n{incidence_matrix}')
	logging.info(f'weight matrix:\n{weight_matrix}')

	num_runs = 100000
	num_arms = 2
	num_timesteps = 1000
	changes_at = np.arange(num_timesteps, step=1000)
	num_changes = changes_at.shape[0]
	bandit_true_means = np.random.normal(0, 1, (num_runs, num_changes, num_arms))

	# plt.figure(figsize=(8, 5))

	competencies_arrays = np.array([
		np.array([1.0, 1.0, 1.0, 1.0]),
		np.array([0.2, 1.0, 1.0, 1.0]),
		np.array([1.0, 0.2, 1.0, 1.0]),
		np.array([1.0, 1.0, 0.2, 1.0]),
		np.array([0.2, 0.2, 0.2, 0.2]),
	])

	num_experiments = competencies_arrays.shape[0]
	num_experiment_cols = competencies_arrays.shape[0]
	num_experiment_rows = 2

	fig, axs = plt.subplots(num_experiment_rows, num_experiment_cols, figsize=(num_experiment_cols * 8, num_experiment_rows * 5), sharey='row')
	# ax = ax.flatten()

	for competencies_arr_idx, competencies_arr in enumerate(competencies_arrays):
		run(
			bandit_true_means,
			changes_at,
			competencies_arr,
			num_timesteps,
			np.copy(adjacency_with_self),
			np.copy(weight_matrix),
			axs=axs[:, competencies_arr_idx],
			experiment_name=np.array2string(competencies_arr),
		)

	plt.suptitle(f'(3, 1) Lollipop, {num_runs} runs, {num_changes - 1} bandit changes, {num_arms} arms')
	plt.xlabel('Timestep')
	plt.ylabel('Regret')
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # legend on the right side
	# plt.save('latest_plt.png')
	logging.info(f'ended script')
	plt.show()


# @njit
@njit(parallel=True)
def coopucb2(bandit_true_means: np.ndarray, competencies: np.ndarray, num_timesteps: int, P: np.ndarray):
	'''
	Plays coopucb2 given the number of runs, number of arms, timesteps, true
	means of arms, and the P matrix of the network. Optimized to work with
	numba.

	## Parameters
	bandit_true_means: (num_runs, num_changes, num_arms) array of true means of arms
	competencies: (num_agents) array of agent competencies
	num_timesteps: number of timesteps to run the algorithm
	P: (num_agents, num_agents) weight matrix of network for information sharing

	# Returns
	regret: (num_runs, num_agents, num_timesteps) array of regret for each agent at each timestep
	percent_optimal_action: (num_runs, num_agents, num_timesteps) array of the percentage of times the optimal action was selected by each agent at each timestep
	'''

	# constants, hyperparameters
	num_runs, num_changes, num_arms = bandit_true_means.shape
	current_change_idx = 0
	num_agents, _ = P.shape

	sigma_g = 1
	eta = 2 # try 2, 2.2, 3.2
	gamma = 2.9 # try 1.9, 2.9
	f = lambda t : np.sqrt(np.log(t))
	min_variance = 1 # minimum variance for the gaussian distribution behind each arm

	# precompute constants
	G_eta = 1 - (eta ** 2) / 16
	c0 = 2 * gamma / G_eta

	# preallocate arrays
	regret = np.zeros((num_runs, num_agents, num_timesteps))
	times_best_arm_selected = np.zeros((num_runs, num_agents), dtype=np.int16)
	percent_optimal_action = np.zeros((num_runs, num_agents, num_timesteps)) # keep track of the percentage of times the optimal action was selected by each agent

	for run in prange(num_runs):
		# estimated reward
		Q = np.zeros((num_changes, num_agents, num_arms))

		# estimated means s/n
		s = np.zeros((num_changes, num_agents, num_arms)) # estimated cumulative expected reward
		n = np.zeros((num_changes, num_agents, num_arms)) # estimated number of times an arm has been selected by each agent
		s_real = np.zeros((num_changes, num_agents, num_arms)) # cumulative reward - actual value
		n_real = np.zeros((num_changes, num_agents, num_arms)) # number of times an arm has been selected by each agent - actual value

		xsi = np.zeros((num_agents, num_arms)) # number of times that arm has been selected in that timestep
		reward = np.zeros((num_agents, num_arms)) # reward vector in that timestep
		total_individual_rewards = np.zeros((num_agents)) # throughout the run

		bandit = bandit_true_means[run, 0, :] # initialize bandit
		best_arm_mean_idx = np.argmax(bandit)
		best_arm_mean = bandit[best_arm_mean_idx]

		current_bandit_t = 0 # timestep local to the current bandit

		for t in range(num_timesteps):
			last_t = current_bandit_t - 1 if current_bandit_t > 0 else 0

			if current_bandit_t < num_arms:
				for k in range(num_agents):
					reward[k] = np.zeros(num_arms)
					xsi[k] = np.zeros(num_arms)
					action = current_bandit_t

					reward[k, action] = np.random.normal(bandit[action], min_variance / competencies[k])
					total_individual_rewards[k] += reward[k, action]
					regret[run, k, t] = best_arm_mean - bandit[action]
					xsi[k, action] += 1
					s_real[current_change_idx, k, action] += reward[k, action]
					n_real[current_change_idx, k, action] += 1

					if action == best_arm_mean_idx:
						times_best_arm_selected[run, k] += 1
			else:
				for k in range(num_agents):
					for i in range(num_arms):
						true_estimate = s[current_change_idx, k, i] / n[current_change_idx, k, i]
						c1 = (n[current_change_idx, k, i] + f(last_t)) / (num_agents * n[current_change_idx, k, i])
						c2 = np.log(last_t) / n[current_change_idx, k, i]
						confidence_bound = sigma_g * np.sqrt(c0 * c1 * c2)
						Q[current_change_idx, k, i] = true_estimate + confidence_bound

					reward[k] = np.zeros(num_arms)
					xsi[k] = np.zeros(num_arms)

					action = np.argmax(Q[current_change_idx, k, :])
					reward[k, action] = np.random.normal(bandit[action], min_variance / competencies[k])
					total_individual_rewards[k] += reward[k, action]
					regret[run, k, t] = best_arm_mean - bandit[action]
					xsi[k, action] += 1
					s_real[current_change_idx, k, action] += reward[k, action]
					n_real[current_change_idx, k, action] += 1

					if action == best_arm_mean_idx:
						times_best_arm_selected[run, k] += 1

			percent_optimal_action[run, :, t] = times_best_arm_selected[run, :] / (t + 1)

			# update estimates using running consensus
			for i in range(num_arms):
				n[current_change_idx, :, i] = P @ (n[current_change_idx, :, i] + xsi[:, i])
				s[current_change_idx, :, i] = P @ (s[current_change_idx, :, i] + reward[:, i])

			current_bandit_t += 1

	return regret, percent_optimal_action

if __name__ == '__main__':
	logging.basicConfig(filename='output_debug.log',filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
	np.set_printoptions(linewidth=9999999)
	main()
