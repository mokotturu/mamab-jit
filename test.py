from time import ctime, time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from scipy.sparse.csgraph import laplacian

from graph_optimization import fdla_weights_symmetric, fmmc_weights, fastest_averaging_constant_weight, max_degree_weights, metropolis_hastings_weights


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
		np.array([
			[0, 1, 1, 1, 0, 0, 0, 0],
			[1, 0, 1, 0, 1, 0, 0, 0],
			[1, 1, 0, 1, 1, 0, 0, 0],
			[1, 0, 1, 0, 1, 1, 1, 1],
			[0, 1, 1, 1, 0, 1, 1, 1],
			[0, 0, 0, 1, 1, 0, 1, 1],
			[0, 0, 0, 1, 1, 1, 0, 1],
			[0, 0, 0, 1, 1, 1, 1, 0],
		])
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
		np.array([
			[  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
			[ -1,  0,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
			[  0, -1,  0, -1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0],
			[  0,  0, -1,  0,  0,  0,  0,  0, -1,  0,  1,  0,  0,  0,  0,  0,  0],
			[  0,  0,  0,  0, -1,  0,  0,  0,  0, -1, -1,  1,  1,  1,  0,  0,  0],
			[  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1,  0,  0,  1,  1,  0],
			[  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1,  0, -1,  0,  1],
			[  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1,  0, -1, -1],
		])
	]

	kappas = [0.02, 0.3, 0.9]
	networks = [
		'All-to-All',
		'Star',
		'8-agents'
	]


	# cm = plt.get_cmap('gist_rainbow')
	colors = [
		'tab:blue',
		'tab:orange',
		'tab:green',
		'tab:red',
		'tab:purple',
		'tab:brown',
		'tab:pink',
		'tab:gray',
		'tab:olive',
		'tab:cyan',
	]

	labels = [
		'κ = 0.02',
		'κ = 0.3',
		'κ = 0.9',
		'Constant-edge',
		'Maximum-degree',
		'Local-degree (MH)',
		'FMMC',
		'FDLA',
		'LMSC',
	]

	markers = [
		'o',
		'^',
		's',
		'x',
		'v',
		'*',
		'1',
		'D',
		'P',
	]

	ylim_bottoms = [22, 22, 20]

	trueMeans = np.array([np.random.normal(0, 1, N) for _ in range(runs)])

	# trueMeans = np.array([[-0.61868816,  1.94749521,  0.45181122,  2.00299915, -0.52358688, -1.56532959, -0.65631623, -0.32224913,  0.59380186,  0.62113954] for _ in range(runs)])
	np.save('trueMeans.npy', trueMeans)

	# sizing
	SMALL_SIZE = 10
	MEDIUM_SIZE = 14
	LARGE_SIZE = 18

	# plt.rcParams["figure.figsize"] = (15, 8)
	plt.rc('font', size=SMALL_SIZE)
	plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=MEDIUM_SIZE)
	plt.rc('xtick', labelsize=MEDIUM_SIZE)
	plt.rc('ytick', labelsize=MEDIUM_SIZE)

	fig, ax = plt.subplots()

	# useful for agent wise plots
	# colormap = plt.cm.nipy_spectral
	# colors = [colormap(i) for i in np.linspace(0, 1, 16)]
	# axs.set_prop_cycle('color', colors)

	print(f'Simulation started at {ctime(time())}')

	for networkIdx, (A, I, network) in enumerate(zip(As, Is, networks)):
		PLabels, PMats = [], []

		# kappas
		for kappa in kappas:
			PLabels.append(f'κ = {kappa}')
			PMats.append(generateP(A, kappa))

		# FMMC
		PLabels.append('FMMC')
		PMats.append(fmmc_weights(I)[1])

		# constant-degree
		PLabels.append('Constant-degree')
		PMats.append(fastest_averaging_constant_weight(I)[1])

		# max-degree
		PLabels.append('Maximum-degree')
		PMats.append(max_degree_weights(I)[1])

		# local-degree (MH)
		PLabels.append('Local-degree (MH)')
		PMats.append(metropolis_hastings_weights(I)[1])

		# FDLA
		PLabels.append('FDLA')
		PMats.append(fdla_weights_symmetric(I)[1])

		for idx, (label, P) in enumerate(zip(PLabels, PMats)):
			# print(label)
			# print(P)
			res = run(runs, N, T, trueMeans, P)
			# resMod = run(runs, N, T, trueMeans, P, sigma_g=0)

			# log all agent wise data
			np.save(f'testdata/{network}-{label}.npy', res)
			# np.save(f'testdata/{network}-{label}-{N}-{runs}-MOD.npy', resMod)

			# res = np.load(f'data/{network}-{label}.npy')

			res = np.mean(np.cumsum(res, axis=2), axis=1) # cumulative sum over T and avg over agents
			# resMod = np.mean(np.cumsum(resMod, axis=2), axis=1) # cumulative sum over T and avg over agents

			# for meanresidx, r in enumerate(res):
			# 	sortedMeans = sorted(trueMeans[meanresidx])
			# 	print(f'{sortedMeans}\n\t\tdifference = {sortedMeans[-1] - sortedMeans[-2]}\t\tres[-1] = {r[-1]}')

			# std = np.std(res, axis=0)
			# std *= 2
			mean = np.mean(res, axis=0) # avg over all runs
			# meanMod = np.mean(resMod, axis=0) # avg over all runs
			# lower_bound = mean - std
			# upper_bound = mean + std

			# for i, agentReg in enumerate(res):
			# for r in res:
			# 	ax.plot(r, lw=1, color='gray', alpha=0.2)

			# print(f'high_count: {high_count}')

			ax.plot(mean, color=colors[idx], label=f'{label}', lw=2)
			# ax.plot(meanMod, color=colors[idx], alpha=0.5, marker='x', markevery=50, label=f'{label}-MOD', lw=2)
			# ax.fill_between(np.arange(T), upper_bound, lower_bound, alpha=0.4)
			# ax2.errorbar(np.arange(T), mean, yerr=std, errorevery=50, marker=markers[idx], markevery=50, label=f'{label}', lw=2)


		# subplot settings
		# ax.set_ylim(bottom=ylim_bottoms[networkIdx])
		ax.grid(True)
		# ax2.grid(True)
		ax.set_title(network)
		ax.legend()
		ax.set_xlabel('Timesteps')
		ax.set_ylabel('Average Cumulative Regret')
		plt.savefig(f'testimg/{network}-avg-cum-reg.svg', format='svg')
		plt.savefig(f'testimg/{network}-avg-cum-reg.png', format='png')
		ax.clear()


	print(f'Simulation ended at {ctime(time())}')
	fig.tight_layout()
	# plt.show()

@njit(parallel=True)
def run(runs: int, N: int, T: int, trueMeans: np.ndarray, P: np.ndarray, sigma_g=1) -> np.ndarray:
	'''
	Plays coopucb2 given the number of runs, number of arms, timesteps, true
	means of arms, and the P matrix of the network. Optimized to work with
	numba.
	'''
	# sigma_g = 1		# try 10
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
	# print(f'all abs eigvals: {l}')
	l = l[1 - l > 1e-5]
	print(f'kappa: {kappa}, rho: {np.max(l)}')

	return P


if __name__ == '__main__':
	main()
