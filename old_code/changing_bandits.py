from time import ctime, time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numba import njit, prange


def main():
	num_arms = 2
	num_runs = 1000
	num_timesteps = 2000

	SWITCH_TIMESTEPS = np.array([0])
	NUM_SWITCHES = SWITCH_TIMESTEPS.shape[0]

	# label, agent
	agents = [
		[
			'Competency = 1.0 (perfect)',
			DecayingEpsilonGreedy(num_arms, epsilon=0.01, competency=1.0, should_update_competency=False),
		],
		[
			'Competency = 1.0',
			DecayingEpsilonGreedy(num_arms, epsilon=0.01, competency=1.0, should_update_competency=True),
		],
	]

	true_means = np.random.normal(0, 1, size=(num_runs, NUM_SWITCHES + 1, num_arms))
	print(true_means)
	regret = np.zeros((len(agents), num_runs, num_timesteps))
	competencies = np.zeros((len(agents), num_runs, num_timesteps))

	for run in range(num_runs):
		for _, alg in agents:
			alg.reset()

		bandit = Bandit(true_means[run, 0, :])
		next_switch_idx = 0
		next_switch_timestep = SWITCH_TIMESTEPS[next_switch_idx]

		for t in range(num_timesteps):
			if t == next_switch_timestep:
				if next_switch_idx + 1 < NUM_SWITCHES:
					next_switch_idx += 1
				next_switch_timestep = SWITCH_TIMESTEPS[next_switch_idx]
				bandit.reset(true_means[run, next_switch_idx + 1])

			for alg_idx, (_, alg) in enumerate(agents):
				_, regret[alg_idx, run, t] = alg.act(bandit)
				competencies[alg_idx, run, t] = alg.competency

	regret = np.cumsum(regret, axis=2)
	regret = np.mean(regret, axis=1)
	competencies = np.mean(competencies, axis=1)

	fig, ax = plt.subplots(1, 2)

	# individual regrets
	for alg_reg_idx, alg_reg in enumerate(regret):
		ax[0].plot(alg_reg, label=agents[alg_reg_idx][0])

	# team average regret
	ax[0].plot(np.mean(regret, axis=0), label='Team Average', linestyle='--')

	ax[0].legend()
	ax[0].set_xlabel('Timestep')
	ax[0].set_ylabel('Cumulative Regret')

	# plot competencies
	for alg_comp_idx, alg_comp in enumerate(competencies):
		ax[1].plot(alg_comp, label=agents[alg_comp_idx][0])
	ax[1].legend()
	ax[1].set_xlabel('Timestep')
	ax[1].set_ylabel('Competency')

	plt.show()


class Bandit:
	def __init__(self, true_means: np.ndarray):
		self.true_means = true_means
		self.sigma = 1.0
		self.num_arms = true_means.shape[0]
		self.max_true_mean = np.max(true_means)
		self.max_true_mean_idx = np.argmax(true_means)

	def pull(self, arm: int, agent_competency=1.0):
		'''
		Returns a tuple of the reward and the regret after pulling given arm.
		'''

		# make sure agent_competency is greater than zero?
		return np.random.normal(self.true_means[arm], agent_competency), self.max_true_mean - self.true_means[arm]

	def evaluate_competency(self, agent_estimates: np.ndarray):
		# return the distance between the true means and the agent's estimates
		# 1 <= competency < inf
		# try log(competency + e)
		return 1 + np.linalg.norm(self.true_means - agent_estimates)
		# return 1 + np.abs(self.max_true_mean - agent_estimates[self.max_true_mean_idx])

	def reset(self, new_true_means: np.ndarray):
		self.true_means = new_true_means
		self.num_arms = new_true_means.shape[0]
		self.max_true_mean = np.max(new_true_means)


class UCB:
	def __init__(self, num_arms, competency=1.0, should_update_competency=True):
		self.num_arms = num_arms
		self.competency = competency
		self.Q = np.zeros(num_arms)
		self.N = np.zeros(num_arms)
		self.tick = 0
		self.should_update_competency = should_update_competency

	def act(self, bandit: Bandit):
		if self.tick < self.num_arms:
			action = self.tick
		else:
			action = np.random.choice(self.num_arms) if np.random.rand() < self.epsilon else np.argmax(self.Q)

		# to make sure all arms all pulled at least once to avoid division by zero
		reward, regret = bandit.pull(action, self.competency)

		self.N[action] += 1
		self.Q[action] += (reward - self.Q[action]) / self.N[action]
		self.tick += 1

		if self.epsilon > self.min_epsilon:
			self.epsilon = 1 / np.sqrt(self.tick + 1)

		if self.should_update_competency and self.tick > self.num_arms:
			self.competency = bandit.evaluate_competency(self.Q / self.N)

		return reward, regret

	def reset(self):
		self.Q = np.zeros(self.num_arms)
		self.N = np.zeros(self.num_arms)
		self.tick = 0

class DecayingEpsilonGreedy:
	def __init__(self, num_arms, epsilon=1, min_epsilon=0.01, competency=1.0, should_update_competency=True):
		self.num_arms = num_arms
		self.epsilon = epsilon
		self.min_epsilon = min_epsilon
		self.competency = competency
		self.Q = np.zeros(num_arms)
		self.N = np.zeros(num_arms)
		self.tick = 0
		self.should_update_competency = should_update_competency

	def act(self, bandit: Bandit):
		if self.tick < self.num_arms:
			action = self.tick
		else:
			action = np.random.choice(self.num_arms) if np.random.rand() < self.epsilon else np.argmax(self.Q)

		# to make sure all arms all pulled at least once to avoid division by zero
		reward, regret = bandit.pull(action, self.competency)

		self.N[action] += 1
		self.Q[action] += (reward - self.Q[action]) / self.N[action]
		self.tick += 1

		# does not decay epsilon properly when bandit changes
		# if self.epsilon > self.min_epsilon:
		# 	self.epsilon = 1 / np.sqrt(self.tick + 1)

		if self.should_update_competency and self.tick > self.num_arms:
			self.competency = bandit.evaluate_competency(self.Q / self.N)

		return reward, regret

	def reset(self):
		self.Q = np.zeros(self.num_arms)
		self.N = np.zeros(self.num_arms)
		self.tick = 0


if __name__ == '__main__':
	main()
