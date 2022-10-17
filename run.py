from time import ctime, time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from scipy.sparse.csgraph import laplacian

from graph_optimization import fdla_weights_symmetric


def main():
	N = 10
	runs = 10000
	T = 1000

	# adjacency matrices
	As = [
		np.array([
			[0, 1, 1, 1, 1],
			[1, 0, 1, 1, 1],
			[1, 1, 0, 1, 1],
			[1, 1, 1, 0, 1],
			[1, 1, 1, 1, 0],
		]),
		np.array([
			[0, 0, 1, 0, 0],
			[0, 0, 1, 0, 0],
			[1, 1, 0, 1, 1],
			[0, 0, 1, 0, 0],
			[0, 0, 1, 0, 0],
		]),
	]

	# corresponding incidence matrices
	Is = [
		np.array([
			[ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0],
			[-1,  0,  0,  0,  1,  1,  1,  0,  0,  0],
			[ 0, -1,  0,  0, -1,  0,  0,  1,  1,  0],
			[ 0,  0, -1,  0,  0, -1,  0, -1,  0,  1],
			[ 0,  0,  0, -1,  0,  0, -1,  0, -1, -1],
		]),
		np.array([
			[  1,  0,  0,  0],
			[  0,  1,  0,  0],
			[ -1, -1, -1, -1],
			[  0,  0,  1,  0],
			[  0,  0,  0,  1],
		]),
	]

	kappas = [0.02, 0.1, 0.3, 0.6, 0.9]
	networks = [
		'All-to-All',
		'Star',
	]

	trueMeans = np.array([np.random.normal(0, 30, N) for _ in range(runs)])

	# sizing
	SMALL_SIZE = 10
	MEDIUM_SIZE = 14
	LARGE_SIZE = 18

	plt.rcParams["figure.figsize"] = (15, 8)
	plt.rc('font', size=MEDIUM_SIZE)
	plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=MEDIUM_SIZE)
	plt.rc('xtick', labelsize=MEDIUM_SIZE)
	plt.rc('ytick', labelsize=MEDIUM_SIZE)

	fig, axs = plt.subplots(len(networks), 1, sharex=True, sharey=True)

	# useful for agent wise plots
	# colormap = plt.cm.nipy_spectral
	# colors = [colormap(i) for i in np.linspace(0, 1, 16)]
	# axs.set_prop_cycle('color', colors)

	print(f'Simulation started at {ctime(time())}')

	for A, I, network, ax in zip(As, Is, networks, axs):
		PLabels, PMats = [], []

		# kappas
		for kappa in kappas:
			PLabels.append(f'Kappa = {kappa}')
			PMats.append(generateP(A, kappa))

		# FDLA
		PLabels.append('FDLA')
		PMats.append(fdla_weights_symmetric(I)[1])

		for label, P in zip(PLabels, PMats):
			res = run(runs, N, T, trueMeans, P)
			res = np.mean(res, axis=0) # avg over all runs

			# # log all agent wise data
			np.savetxt(f'{network}-{label}.txt', res)
			# res = np.genfromtxt(f'{network}-{label}.txt')

			# for i, agentReg in enumerate(res):
			ax.plot(np.cumsum(np.mean(res, axis=0)), label=f'{label}', lw=2)

		# subplot settings
		ax.grid(True)
		ax.set_title(network)
		ax.legend()
		ax.set_xlabel('Timesteps')
		ax.set_ylabel('Average Cumulative Regret')


	print(f'Simulation ended at {ctime(time())}')
	plt.show()

@njit(parallel=True)
def run(runs: int, N: int, T: int, trueMeans: np.ndarray, P: np.ndarray) -> np.ndarray:
	'''
	Plays coopucb2 given the number of runs, number of arms, timesteps, true
	means of arms, and the P matrix of the network. Optimized to work with
	numba.
	'''
	sigma_g = 1		# try 10
	# eta = 2		# try 2, 2.2, 3.2
	gamma = 2.0 	# try 1.9, 2.9
	f = lambda t : np.sqrt(np.log(t))
	Geta = 2.0		# try 1 - (eta ** 2)/16
	var = 1.0		# variance for the gaussian distribution behind each arm
	M, _ = P.shape

	reg = np.zeros((runs, M, T))

	# run coop-ucb2 "runs" number of times
	for run in prange(runs):
		n = np.zeros((M, N))	# number of times an arm has been selected by each agent
		s = np.zeros((M, N))	# cumulative expected reward
		xsi = np.zeros((M, N))	# number of times that arm has been selected in that timestep
		rew = np.zeros((M, N))	# reward
		Q = np.zeros((M, N))	# estimated reward

		bestArm = np.max(trueMeans[run])

		for t in range(T):
			if t < N:
				for k in range(M):
					action = t
					rew[k, action] = np.random.normal(trueMeans[run, action], var)
					reg[run, k, t] = bestArm - trueMeans[run, action]
					xsi[k, action] += 1
			else:
				for k in range(M):
					for i in range(N):
						Q[k, i] = (s[k, i] / n[k, i]) + sigma_g * (np.sqrt((2 * gamma / Geta) * ((n[k, i] + f(t - 1)) / (M * n[k, i])) * (np.log(t - 1) / n[k, i])))

					rew[k] = np.zeros(N)
					xsi[k] = np.zeros(N)

					action = np.argmax(Q[k])
					rew[k, action] = np.random.normal(trueMeans[run, action], var)
					reg[run, k, t] = bestArm - trueMeans[run, action]
					xsi[k, action] += 1

			# update estimates using running consensus
			for i in range(N):
				n[:, i] = P @ (n[:, i] + xsi[:, i])
				s[:, i] = P @ (s[:, i] + rew[:, i])

	return reg

def generateP(A, kappa):
	dmax = np.max(np.sum(A, axis=0))
	L = laplacian(A, normed=False)
	M, _ = np.shape(A)
	I = np.eye(M)

	P = I - (kappa/dmax) * L

	# print rho
	l = np.absolute(np.linalg.eigvals(P))
	print(f'all abs eigvals: {l}')
	l = l[1 - l > 1e-5]
	print(f'kappa: {kappa}, rho: {np.max(l)}')

	return P


if __name__ == '__main__':
	main()
