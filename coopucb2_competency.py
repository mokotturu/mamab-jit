import argparse
import logging
from pathlib import Path
from time import ctime, time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
from networkx.drawing.nx_pydot import graphviz_layout
from numba import njit, prange

from graph_optimization import (fastest_averaging_constant_weight,
                                fdla_weights_symmetric, fmmc_weights,
                                lmsc_weights, max_degree_weights,
                                metropolis_hastings_weights)
from old_lmsc import generateP

def main(args):
	logging.info(f'started script')
	network_name = args.network
	adjacency_matrix = np.load(f'data/saved_networks/{network_name}_adj.npy')
	network_name += '_c_randn'
	# adjacency_matrix = nx.to_numpy_array(nx.star_graph(4))
	alg_type = coopucb2_og if args.alg == 'coopucb2_og' else coopucb2_limited_communication
	alg_type_str = args.alg

	incidence_matrix = np.asarray(nx.linalg.graphmatrix.incidence_matrix(nx.from_numpy_array(adjacency_matrix), oriented=True).todense())
	num_agents = adjacency_matrix.shape[0]

	adjacency_with_self = adjacency_matrix + np.eye(num_agents)
	neighbor_count = np.sum(adjacency_with_self, axis=1)

	# weight_matrix = adjacency_with_self / neighbor_count[:, None]
	# weight_matrix, _ = generateP(adjacency_matrix, 0.6)
	_, weight_matrix, _ = fmmc_weights(incidence_matrix)

	logging.info(f'incidence matrix:\n{incidence_matrix}')
	logging.info(f'weight matrix:\n{weight_matrix}')

	num_runs = 10000
	num_arms = 20
	num_timesteps = 500
	changes_at = np.arange(num_timesteps, step=1000)
	num_changes = changes_at.shape[0]
	bandit_true_means = np.random.normal(0, 1, (num_runs, num_changes, num_arms))

	# plt.figure(figsize=(8, 5))

	# competencies_arrays = np.array([
	# 	np.array([1.0, 1.0, 1.0, 1.0]),
	# 	np.array([0.2, 1.0, 1.0, 1.0]),
	# 	np.array([1.0, 0.2, 1.0, 1.0]),
	# 	np.array([1.0, 1.0, 1.0, 0.2]),
	# 	np.array([1.0, 1.0, 0.2, 0.2]),
	# 	np.array([0.2, 0.2, 0.2, 0.2]),
	# ])
	competencies_arrays = np.array([
		np.ones(num_agents),
		np.array([0.2, 1.0, 1.0, 1.0]),
		np.array([1.0, 0.2, 1.0, 1.0]),
		np.array([1.0, 1.0, 0.2, 0.2]),
		np.array([0.2, 0.2, 0.2, 0.2]),
	])

	# linestyles = ['-' if i < 10 else ':' for i in range(num_agents)]
	linestyles = ['-' for i in range(num_agents)]
	linemarkers = list(Line2D.markers)[:-4] # remove last 4 entries as they're invalid markers

	num_experiments = competencies_arrays.shape[0]
	num_experiment_cols = competencies_arrays.shape[0]
	num_experiment_rows = 3

	cmap = plt.get_cmap('tab10')
	COLORS = cmap(np.arange(num_agents))
	# COLORS = cmap(np.linspace(0, 1, num_agents))

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
			experiment_name=np.array2string(competencies_arr),
			alg_type=alg_type,
			axs=axs[:, competencies_arr_idx],
			colors=COLORS,
			linestyles=linestyles,
			linemarkers=linemarkers,
		)

	plt.suptitle(f'{network_name.title()} graph, {num_runs} runs, {num_changes - 1} bandit changes, {num_arms} arms')
	fig.supxlabel('Timestep')
	# fig.supylabel('Regret')
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # legend on the right side
	plt.savefig(f'data/img/png/{network_name.replace(" ", "_")}_{alg_type_str}.png', format='png')
	plt.savefig(f'data/img/svg/{network_name.replace(" ", "_")}_{alg_type_str}.svg', format='svg')
	logging.info(f'ended script')
	plt.show()


def run(bandit_true_means: np.ndarray, changes_at: np.ndarray, competencies: np.ndarray, num_timesteps: int, adjacency_with_self: np.ndarray, weight_matrix: np.ndarray, experiment_name: str, alg_type, axs=None, colors=None, linestyles=None, linemarkers=None):
	if axs is None:
		raise ValueError('ax must be provided')

	# reset plt color cycle
	for _ax in axs:
		_ax.set_prop_cycle(None)

	ax_top, ax_middle, ax_bottom = axs

	if alg_type == None:
		raise ValueError('alg_type must be provided')

	logging.info(f'started {experiment_name} version')
	regret, s_n_error, percent_optimal_action = alg_type(
		bandit_true_means,
		competencies,
		num_timesteps,
		weight_matrix,
	)

	regret = np.cumsum(regret, axis=2) # cumulative regret over time
	regret = np.mean(regret, axis=0) # average over runs
	s_n_error = np.mean(s_n_error, axis=0) # average over runs
	percent_optimal_action = np.mean(percent_optimal_action, axis=0) # average over runs

	# np.save('data/data/changing_P_regret.npy', regret)
	# regret = np.load('data/data/changing_P_regret.npy')

	# plot regret
	for agent_idx, agent_regret in enumerate(regret):
		ax_top.plot(agent_regret, label=f'agent {agent_idx + 1}', linestyle=linestyles[agent_idx], alpha=(0.5 + (agent_idx % 2) / 2), marker=f'${agent_idx + 1}$', markevery=num_timesteps // 6, color=colors[agent_idx])
		logging.info(f'agent {agent_idx + 1} final regret: {agent_regret[-1]}')

	regret = np.mean(regret, axis=0) # average over agents
	ax_top.plot(regret, label=f'team average', color='black', linestyle='--')

	ax_top.title.set_text(experiment_name)
	ax_top.grid()
	ax_top.set_xlabel('Timestep')
	ax_top.set_ylabel('Regret')

	# plot s/n error
	width = 2
	inset = [0.425, 0.4, 0.55, 0.55]
	axins = ax_middle.inset_axes(inset)
	axins.spines['bottom'].set_linewidth(width)
	axins.spines['top'].set_linewidth(width)
	axins.spines['right'].set_linewidth(width)
	axins.spines['left'].set_linewidth(width)
	axins.tick_params(width=width)
	axins.grid(which='both', axis='both')

	individual_agent_max_errors = np.zeros(s_n_error.shape[0])

	for agent_s_n_error in s_n_error:
		individual_agent_max_errors[agent_idx] = np.max(agent_s_n_error)

	for agent_idx, agent_s_n_error in enumerate(s_n_error):
		decreasing_errors = agent_s_n_error[np.argmax(agent_s_n_error):]
		five_percent_line = np.argmax(agent_s_n_error) + np.argmax(decreasing_errors < 0.05 * np.max(individual_agent_max_errors))
		ax_middle.plot(agent_s_n_error, label=f'agent {agent_idx + 1}', linestyle=linestyles[agent_idx], alpha=(0.5 + (agent_idx % 2) / 2), marker=f'${agent_idx + 1}$', markevery=num_timesteps // 6, color=colors[agent_idx])
		axins.plot(agent_s_n_error, label=f'agent {agent_idx + 1}', linestyle=linestyles[agent_idx], alpha=(0.5 + (agent_idx % 2) / 2), marker=f'${agent_idx + 1}$', markevery=num_timesteps // 6, color=colors[agent_idx])
		# ax_middle.axvline(five_percent_line, linestyle=linestyles[agent_idx], color=colors[agent_idx])
		# axins.axvline(five_percent_line, linestyle=linestyles[agent_idx], color=colors[agent_idx])

	s_n_error = np.mean(s_n_error, axis=0) # average over agents
	ax_middle.plot(s_n_error, label=f'team average', color='black', linestyle='--')
	axins.plot(s_n_error, label=f'team average', color='black', linestyle='--')

	ax_middle.title.set_text(experiment_name)
	ax_middle.grid()
	ax_middle.set_xlabel('Timestep')
	ax_middle.set_ylabel('s/n error')

	# plot percent optimal action
	for agent_idx, agent_percent_optimal_action in enumerate(percent_optimal_action):
		ax_bottom.plot(agent_percent_optimal_action, label=f'agent {agent_idx + 1}', linestyle=linestyles[agent_idx], alpha=(0.5 + (agent_idx % 2) / 2), marker=f'${agent_idx + 1}$', markevery=num_timesteps // 6, color=colors[agent_idx])

	percent_optimal_action = np.mean(percent_optimal_action, axis=0) # average over agents
	ax_bottom.plot(percent_optimal_action, label=f'team average', color='black', linestyle='--')
	ax_bottom.title.set_text(experiment_name)
	ax_bottom.grid()
	ax_bottom.set_xlabel('Timestep')
	ax_bottom.set_ylabel('Percent optimal action')

	logging.info(f'ended {experiment_name} version')


@njit(parallel=True)
def coopucb2_limited_communication(bandit_true_means: np.ndarray, competencies: np.ndarray, num_timesteps: int, P: np.ndarray, comm_every_t: int=50):
	'''
	Plays coopucb2 given the number of runs, number of arms, timesteps, true
	means of arms, and the P matrix of the network. Optimized to work with
	numba.

	## Parameters
	bandit_true_means: (num_runs, num_changes, num_arms) array of true means of arms
	competencies: (num_agents) array of agent competencies
	num_timesteps: number of timesteps to run the algorithm
	P: (num_agents, num_agents) weight matrix of network for information sharing
	comms_every: number of timesteps between communications

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
	s_n_error = np.zeros((num_runs, num_agents, num_timesteps))
	times_best_arm_selected = np.zeros((num_runs, num_agents), dtype=np.int16)
	percent_optimal_action = np.zeros((num_runs, num_agents, num_timesteps)) # keep track of the percentage of times the optimal action was selected by each agent

	for run in prange(num_runs):
		# estimated reward
		Q = np.zeros((num_changes, num_agents, num_arms))

		# estimated means s/n
		s = np.zeros((num_changes, num_agents, num_arms)) # estimated cumulative expected reward
		n = np.zeros((num_changes, num_agents, num_arms)) # estimated number of times an arm has been selected by each agent

		reward = np.zeros((num_agents, num_arms)) # reward vector in that timestep
		xsi = np.zeros((num_agents, num_arms)) # number of times that arm has been selected in that timestep
		# cumulated_reward = np.zeros((num_agents, num_arms))
		# cumulated_xsi = np.zeros((num_agents, num_arms))
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

					reward[k, action] = np.random.normal(
						bandit[action] * (competencies[k] + np.random.randn()),
						min_variance / (competencies[k] * np.abs(np.random.randn()))
					) if competencies[k] < 1 else np.random.normal(bandit[action], min_variance / competencies[k])
					total_individual_rewards[k] += reward[k, action]
					regret[run, k, t] = best_arm_mean - bandit[action]
					xsi[k, action] += 1

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
					reward[k, action] = np.random.normal(
						bandit[action] * (competencies[k] + np.random.randn()),
						min_variance / (competencies[k] * np.abs(np.random.randn()))
					) if competencies[k] < 1 else np.random.normal(bandit[action], min_variance / competencies[k])
					total_individual_rewards[k] += reward[k, action]
					regret[run, k, t] = best_arm_mean - bandit[action]
					xsi[k, action] += 1

					s_n_error[run, k, t] = np.abs(best_arm_mean - (s[current_change_idx, k, best_arm_mean_idx] / n[current_change_idx, k, best_arm_mean_idx]))

					if action == best_arm_mean_idx:
						times_best_arm_selected[run, k] += 1

			percent_optimal_action[run, :, t] = times_best_arm_selected[run, :] / (t + 1)

			# update estimates using running consensus
			for i in range(num_arms):
				s[current_change_idx, :, i] += reward[:, i]
				n[current_change_idx, :, i] += xsi[:, i]

			if t % comm_every_t == 0:
				for i in range(num_arms):
					s[current_change_idx, :, i] = P @ s[current_change_idx, :, i]
					n[current_change_idx, :, i] = P @ n[current_change_idx, :, i]

			current_bandit_t += 1

	return regret, s_n_error, percent_optimal_action


@njit(parallel=True)
def coopucb2_og(bandit_true_means: np.ndarray, competencies: np.ndarray, num_timesteps: int, P: np.ndarray):
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
	s_n_error = np.zeros((num_runs, num_agents, num_timesteps))
	times_best_arm_selected = np.zeros((num_runs, num_agents), dtype=np.int16)
	percent_optimal_action = np.zeros((num_runs, num_agents, num_timesteps)) # keep track of the percentage of times the optimal action was selected by each agent

	for run in prange(num_runs):
		# estimated reward
		Q = np.zeros((num_changes, num_agents, num_arms))

		# estimated means s/n
		s = np.zeros((num_changes, num_agents, num_arms)) # estimated cumulative expected reward
		n = np.zeros((num_changes, num_agents, num_arms)) # estimated number of times an arm has been selected by each agent

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

					reward[k, action] = np.random.normal(
						bandit[action] * (competencies[k] + np.random.randn()),
						min_variance / (competencies[k] * np.abs(np.random.randn()))
					) if competencies[k] < 1 else np.random.normal(bandit[action], min_variance / competencies[k])
					total_individual_rewards[k] += reward[k, action]
					regret[run, k, t] = best_arm_mean - bandit[action]
					xsi[k, action] += 1

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
					reward[k, action] = np.random.normal(
						bandit[action] * (competencies[k] + np.random.randn()),
						min_variance / (competencies[k] * np.abs(np.random.randn()))
					) if competencies[k] < 1 else np.random.normal(bandit[action], min_variance / competencies[k])
					total_individual_rewards[k] += reward[k, action]
					regret[run, k, t] = best_arm_mean - bandit[action]
					xsi[k, action] += 1

					s_n_error[run, k, t] = np.abs(best_arm_mean - (s[current_change_idx, k, best_arm_mean_idx] / n[current_change_idx, k, best_arm_mean_idx]))

					if action == best_arm_mean_idx:
						times_best_arm_selected[run, k] += 1

			percent_optimal_action[run, :, t] = times_best_arm_selected[run, :] / (t + 1)

			# update estimates using running consensus
			for i in range(num_arms):
				n[current_change_idx, :, i] = P @ (n[current_change_idx, :, i] + xsi[:, i])
				s[current_change_idx, :, i] = P @ (s[current_change_idx, :, i] + reward[:, i])

			current_bandit_t += 1

	return regret, s_n_error, percent_optimal_action

if __name__ == '__main__':
	logging.basicConfig(filename=f'output_{Path(__file__).stem}.log',filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
	np.set_printoptions(linewidth=9999999)
	# setup argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--network', type=str, default='tadpole')
	parser.add_argument('--alg', type=str, default='coopucb2_og')
	args = parser.parse_args()

	main(args)