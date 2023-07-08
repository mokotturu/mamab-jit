import cProfile
import pstats

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import cupy as cp
from numba import njit, prange
from tqdm import tqdm

from graph_optimization import fastest_averaging_constant_weight
from lmsc import coopucb2


@njit
def get_W(adjacency_matrix: np.ndarray) -> np.ndarray:
	'''
	Generate edge weights for LAIE given the adjacency matrix
	'''
	num_nodes = adjacency_matrix.shape[0]
	diagonal_matrix = np.diag(np.sum(adjacency_matrix, axis=0))
	# laplacian = diagonal_matrix - adjacency_matrix
	adjacency_with_self = np.add(adjacency_matrix, np.identity(num_nodes))

	neighbors_with_self = []
	for i in range(num_nodes):
		neighbors_with_self.append(np.nonzero(adjacency_with_self[i])[0])

	neighbors = []
	for i in range(num_nodes):
		neighbors.append(np.nonzero(adjacency_matrix[i])[0])

	W = np.zeros((num_nodes, num_nodes, num_nodes), dtype=np.float32)

	for i in range(num_nodes):
		for j in range(num_nodes):
			for k in range(num_nodes):
				if j not in neighbors_with_self[i] and k == j:
					W[i, j, k] = 1
				elif (j in neighbors_with_self[i] and k in neighbors_with_self[i]) or (i == j and j == k and i == k):
					W[i, j, k] = 1 / len(neighbors_with_self[i])
				else:
					W[i, j, k] = 0

	return W

@njit(parallel=True)
def MAMAB_LAIE(W: np.ndarray, true_means: np.ndarray, timesteps=100, runs=100, epsilon=0.01, decay=0.9):
	num_nodes = W.shape[0]
	num_arms = true_means[0].shape[0]
	sigma = 1

	regret_history = np.zeros((runs, num_nodes, timesteps))

	selected_arms_snapshot = np.zeros(num_nodes)

	for run in prange(runs):
		cumulative_estimated_reward = np.zeros((num_nodes, num_arms), dtype=np.float32)
		cumulative_estimated_pulls = np.zeros((num_nodes, num_arms), dtype=np.float32)
		for t in range(timesteps):
			current_reward_vector = np.zeros((num_nodes, num_arms), dtype=np.float32)
			current_pulls = np.zeros((num_nodes, num_arms), dtype=np.float32)
			if t < num_arms:
				for k in range(num_nodes):
					action = t
					current_reward_vector[k, action] = np.random.normal(true_means[run, action], sigma)
					current_pulls[k, action] += 1
					regret_history[run, k, t] = np.abs(np.max(true_means[run]) - true_means[run, action])
			else:
				for k in range(num_nodes):
					# epsilon greedy
					action = np.random.randint(0, num_arms) if np.random.rand() < epsilon else np.argmax(cumulative_estimated_reward[k] / cumulative_estimated_pulls[k])
					current_reward_vector[k, action] = np.random.normal(true_means[run, action], sigma)
					current_pulls[k, action] += 1
					regret_history[run, k, t] = np.abs(np.max(true_means[run]) - true_means[run, action])

					if t == num_arms:
						selected_arms_snapshot[k] = action

				# epsilon *= decay

			# exchange information using LAIE algorithm
			for i in range(num_arms):
				for k in range(num_nodes):
					cumulative_estimated_reward[:, i] = W[k] @ (cumulative_estimated_reward[:, i] + current_reward_vector[:, i])
					cumulative_estimated_pulls[:, i] = W[k] @ (cumulative_estimated_pulls[:, i] + current_pulls[:, i])


	return regret_history, selected_arms_snapshot

def main():
	# mamab constants
	runs = 1
	timesteps = 1000
	num_arms = 200
	true_means = np.array([np.random.normal(0, 1, num_arms) for _ in range(runs)])

	num_nodes_arr = [80] # , 100, 120, 140, 180]
	degree = 3
	networks = [nx.to_numpy_array(nx.random_regular_graph(degree, num_nodes)) for num_nodes in num_nodes_arr]

	# print current time
	import datetime
	print(datetime.datetime.now())

	for networkIdx, adjacency_matrix in enumerate(networks):
		num_nodes_in_network = adjacency_matrix.shape[0]

		print('running LAIE...')
		print('generating W...')
		W = np.array(get_W(adjacency_matrix))
		print('done.')

		print('running LAIE...')
		regret_history, snap = MAMAB_LAIE(W, true_means, timesteps=timesteps, runs=runs, epsilon=0.01, decay=1)

		print(snap)
		regret_history = np.cumsum(regret_history, axis=2)
		regret = np.mean(regret_history, axis=(0, 1)) # average over runs and nodes
		plt.plot(regret, label=f'LAIE, {num_nodes_in_network} nodes')
		print('done.')

		print('running CoopUCB2...')
		_, _, best_constant_W, _ = fastest_averaging_constant_weight(np.asarray(nx.linalg.graphmatrix.incidence_matrix(nx.from_numpy_array(adjacency_matrix), oriented=True).todense()))
		coopucb2_regret, snap = coopucb2(runs, num_arms, timesteps, true_means, best_constant_W)
		print(snap)
		coopucb2_regret = np.cumsum(coopucb2_regret, axis=2)
		coopucb2_regret = np.mean(coopucb2_regret, axis=(0, 1)) # average over runs and nodes
		plt.plot(coopucb2_regret, label=f'CoopUCB2, {num_nodes_in_network} nodes')
		print('done.')

	plt.xlabel('Timesteps')
	plt.ylabel('Regret')
	# plt.xlim(left=200)
	# plt.ylim(bottom=500)
	plt.grid()
	plt.legend()
	# print time
	print(datetime.datetime.now())

if __name__ == '__main__':
	main()
	# x = np.arange(6).reshape(2, 3).astype('f')
	# print(x)
	# print(x.sum(axis=1))