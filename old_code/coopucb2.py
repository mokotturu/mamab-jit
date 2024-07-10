import numpy as np
from numba import njit, prange

@njit(parallel=True)
def coopucb2(bandit_true_means: np.ndarray, num_timesteps: int, P: np.ndarray):
	'''
	Plays coopucb2 given the number of runs, number of arms, timesteps, true
	means of arms, and the P matrix of the network. Optimized to work with
	numba.

	## Parameters
	bandit_true_means: (num_runs, num_arms) array of true means of arms
	num_timesteps: number of timesteps to run the algorithm
	P: (num_agents, num_agents) weight matrix of network for information sharing

	# Returns
	regret: (num_runs, num_agents, num_timesteps) array of regret for each agent at each timestep
	s_n_error: (num_runs, num_agents, num_timesteps) array of the s/n error for each agent at each timestep
	percent_optimal_action: (num_runs, num_agents, num_timesteps) array of the percentage of times the optimal action was selected by each agent at each timestep
	arm_pulled: (num_runs, num_agents, num_timesteps) array of the arm pulled by each agent at each timestep
	'''

	# constants, hyperparameters
	num_runs, num_arms = bandit_true_means.shape
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
	arm_pulled = np.zeros((num_runs, num_agents, num_timesteps), dtype=np.int32)

	for run in prange(num_runs):
		# estimated reward
		Q = np.zeros((num_agents, num_arms))

		# estimated means s/n
		s = np.zeros((num_agents, num_arms)) # estimated cumulative expected reward
		n = np.zeros((num_agents, num_arms)) # estimated number of times an arm has been selected by each agent

		xsi = np.zeros((num_agents, num_arms)) # number of times that arm has been selected in that timestep
		reward = np.zeros((num_agents, num_arms)) # reward vector in that timestep
		total_individual_rewards = np.zeros((num_agents)) # throughout the run

		bandit = bandit_true_means[run, :] # initialize bandit
		best_arm_mean_idx = np.argmax(bandit)
		best_arm_mean = bandit[best_arm_mean_idx]

		for t in range(num_timesteps):
			last_t = t - 1 if t > 0 else 0

			if t < num_arms:
				for k in range(num_agents):
					reward[k] = np.zeros(num_arms)
					xsi[k] = np.zeros(num_arms)
					action = t

					arm_pulled[run, k, t] = action
					reward[k, action] = np.random.normal(bandit[action], min_variance)
					total_individual_rewards[k] += reward[k, action]
					regret[run, k, t] = best_arm_mean - bandit[action]
					xsi[k, action] += 1

					if action == best_arm_mean_idx:
						times_best_arm_selected[run, k] += 1
			else:
				for k in range(num_agents):
					for i in range(num_arms):
						true_estimate = s[k, i] / n[k, i]
						c1 = (n[k, i] + f(last_t)) / (num_agents * n[k, i])
						c2 = np.log(last_t) / n[k, i]
						confidence_bound = sigma_g * np.sqrt(c0 * c1 * c2)
						Q[k, i] = true_estimate + confidence_bound

					reward[k] = np.zeros(num_arms)
					xsi[k] = np.zeros(num_arms)

					action = np.argmax(Q[k, :])
					arm_pulled[run, k, t] = action
					reward[k, action] = np.random.normal(bandit[action], min_variance)
					total_individual_rewards[k] += reward[k, action]
					regret[run, k, t] = best_arm_mean - bandit[action]
					xsi[k, action] += 1

					s_n_error[run, k, t] = np.abs(best_arm_mean - (s[k, best_arm_mean_idx] / n[k, best_arm_mean_idx]))

					if action == best_arm_mean_idx:
						times_best_arm_selected[run, k] += 1

			percent_optimal_action[run, :, t] = times_best_arm_selected[run, :] / (t + 1)

			# update estimates using running consensus
			for i in range(num_arms):
				n[:, i] = P @ (n[:, i] + xsi[:, i])
				s[:, i] = P @ (s[:, i] + reward[:, i])

	return regret, s_n_error, percent_optimal_action, arm_pulled
