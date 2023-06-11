from time import ctime, time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
from numba import njit, prange
from scipy.sparse.csgraph import laplacian
from scipy import stats
from scipy.linalg import svdvals
import networkx as nx

from graph_optimization import fdla_weights_symmetric, fmmc_weights, lmsc_weights, fastest_averaging_constant_weight, max_degree_weights, metropolis_hastings_weights


def main():
	num_arms = 4
	num_runs = 100
	num_timesteps = 3000
	num_agents = 50

	# adjacency matrices
	As = [
		# all-to-all auto
		nx.to_numpy_array(nx.complete_graph(num_agents)),
		# # house
		# np.array([
		# 	[0, 1, 1, 0, 0],
		# 	[1, 0, 1, 1, 0],
		# 	[1, 1, 0, 0, 1],
		# 	[0, 1, 0, 0, 1],
		# 	[0, 0, 1, 1, 0],
		# ]),
		# ring
		nx.to_numpy_array(nx.cycle_graph(num_agents)),
		# # line
		# np.array([
		# 	[0, 1, 0, 0, 0],
		# 	[1, 0, 1, 0, 0],
		# 	[0, 1, 0, 1, 0],
		# 	[0, 0, 1, 0, 1],
		# 	[0, 0, 0, 1, 0],
		# ]),
		# # star
		# np.array([
		# 	[0, 1, 1, 1, 1],
		# 	[1, 0, 0, 0, 0],
		# 	[1, 0, 0, 0, 0],
		# 	[1, 0, 0, 0, 0],
		# 	[1, 0, 0, 0, 0],
		# ]),
		# # 8-agents
		# np.array([
		# 	[0, 1, 1, 1, 0, 0, 0, 0],
		# 	[1, 0, 1, 0, 1, 0, 0, 0],
		# 	[1, 1, 0, 1, 1, 0, 0, 0],
		# 	[1, 0, 1, 0, 1, 1, 1, 1],
		# 	[0, 1, 1, 1, 0, 1, 1, 1],
		# 	[0, 0, 0, 1, 1, 0, 1, 1],
		# 	[0, 0, 0, 1, 1, 1, 0, 1],
		# 	[0, 0, 0, 1, 1, 1, 1, 0],
		# ]),
		# # large 50
		# np.load('data/saved_networks/large_50_adj.npy'),
	]

	# corresponding incidence matrices
	Is = [
		# all-to-all auto
		np.asarray(nx.linalg.graphmatrix.incidence_matrix(nx.complete_graph(num_agents), oriented=True).todense()),
		# # house
		# np.array([
		# 	[ 1,  0,  0,  0, -1,  0],
		# 	[ 0,  0,  0, -1,  1,  1],
		# 	[-1,  1,  0,  0,  0, -1],
		# 	[ 0,  0, -1,  1,  0,  0],
		# 	[ 0, -1,  1,  0,  0,  0],
		# ]),
		# ring
		np.asarray(nx.linalg.graphmatrix.incidence_matrix(nx.cycle_graph(num_agents), oriented=True).todense()),
		# # line
		# np.array([
		# 	[ 1,  0,  0,  0],
		# 	[-1,  1,  0,  0],
		# 	[ 0, -1,  1,  0],
		# 	[ 0,  0, -1,  1],
		# 	[ 0,  0,  0, -1],
		# ]),
		# # star
		# np.array([
		# 	[ 1,  1,  1,  1],
		# 	[-1,  0,  0,  0],
		# 	[ 0, -1,  0,  0],
		# 	[ 0,  0, -1,  0],
		# 	[ 0,  0,  0, -1],
		# ]),
		# # 8-agents
		# np.array([
		# 	[  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		# 	[ -1,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		# 	[  0, -1,  0, -1,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		# 	[  0,  0, -1,  0,  0, -1,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0],
		# 	[  0,  0,  0,  0, -1,  0, -1, -1,  0,  0,  0,  1,  1,  1,  0,  0,  0],
		# 	[  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0, -1,  0,  0,  1,  1,  0],
		# 	[  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0, -1,  0, -1,  0,  1],
		# 	[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0, -1,  0, -1, -1],
		# ]),
		# # large 50
		# np.load('data/saved_networks/large_50_inc.npy'),
	]

	networks = [
		'all-to-all',
		# 'house',
		'ring',
		# 'line',
		# 'star',
		# '8-agents',
		# 'large_50',
	]

	# trueMeans = np.array([np.random.normal(0, 1, num_arms) for _ in range(num_runs)])
	trueMeans = np.array([[0.2646,  0.6135, 0.8950, 0.5764] for _ in range(num_runs)])
	trueMeansGroups = np.array([[[0.0149,  0.7161, 0.7944, 0.6749], [0.5144,  0.5108, 0.9955, 0.4778]] for _ in range(num_runs)])
	# trueMeans = np.sort(trueMeans, axis=1)[:, ::-1]

	# sizing
	SMALL_SIZE = 10
	MEDIUM_SIZE = 14
	LARGE_SIZE = 18

	# plt.rcParams["figure.figsize"] = (15, 8)
	matplotlib.use('Agg')
	plt.rc('font', size=SMALL_SIZE)
	plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=MEDIUM_SIZE)
	plt.rc('xtick', labelsize=MEDIUM_SIZE)
	plt.rc('ytick', labelsize=MEDIUM_SIZE)

	np.set_printoptions(threshold=99999999999999999, linewidth=99999999999999999)

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

	print(f'Simulation started at {ctime(time())}')

	for mat_idx in range(len(As)):
		print(f'Network: {networks[mat_idx]}')
		Ps, rhos, labels = [], [], []

		# for k in [0.02]:
		# 	P, rho = generateP(As[mat_idx], kappa=k)
		# 	Ps.append(P)
		# 	rhos.append(rho)
		# 	print(f'{"kappa " + str(k):<20s}: {rho} {(1 / np.log(1 / rho))}')
		# 	labels.append(fr'$\kappa$ = {k}')
		# 	print(P)

		# constant edge
		alpha, _, P, rho = fastest_averaging_constant_weight(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		print(f'{"Constant-edge":<20s}: {rho} {(1 / np.log(1 / rho))}')
		labels.append(fr'Constant-edge ($\alpha$ = {alpha})')
		# print(P)

		# maximum degree
		alpha, _, P, rho = max_degree_weights(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		print(f'{"Max-degree":<20s}: {rho} {(1 / np.log(1 / rho))}')
		labels.append(fr'Maximum-degree ($\alpha$ = {alpha})')
		# print(P)

		# local degree (MH)
		_, P, rho = metropolis_hastings_weights(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		print(f'{"Local-degree":<20s}: {rho} {(1 / np.log(1 / rho))}')
		labels.append(fr'Local-degree')
		# print(P)

		# # fmmc
		# _, P, rho = fmmc_weights(Is[mat_idx])
		# rho = get_rho(P)
		# Ps.append(P)
		# rhos.append(rho)
		# print(f'{"FMMC":<20s}: {rho} {(1 / np.log(1 / rho))}')
		# labels.append('FMMC')
		# print(P)

		# fdla
		_, P, rho = fdla_weights_symmetric(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		print(f'{"FDLA":<20s}: {rho} {(1 / np.log(1 / rho))}')
		labels.append('FDLA')
		# print(P)

		# # lmsc
		# _, P = lmsc_weights(Is[mat_idx])
		# rho = get_rho(P)
		# Ps.append(P)
		# rhos.append(rho)
		# print(f'{"LMSC":<20s}: {rho} {(1 / np.log(1 / rho))}')
		# labels.append('LMSC')
		# print(P)

		# average weights
		P = generateAvgW(As[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		print(f'{"Avg":<20s}: {rho} {(1 / np.log(1 / rho))}')
		labels.append('Avg')
		print(spectralGap(P))
		print('\n\n')

		fig, ax = plt.subplots()

		for idx, P in enumerate(Ps):
			# coopucb2
			# reg = coopucb2(num_runs, num_arms, num_timesteps, trueMeans, P)

			# d-UER
			reg = dUER(num_runs, num_arms, num_timesteps, trueMeansGroups, P)

			# save regret
			reg = np.mean(reg, axis=0)	# mean over runs
			np.save(f'data/data/new_{networks[mat_idx].replace(" ", "-")}_reg_{labels[idx]}.npy', reg)

			# comment previous lines and uncomment the following line to read regret values from file
			# reg = np.load(f'data/data/new_{networks[mat_idx].replace(" ", "-")}_reg_{labels[idx]}.npy')

			# plot regret
			fig.suptitle(f'{networks[mat_idx].title()} network')
			fig.supxlabel('Timesteps')
			fig.supylabel('Mean Cumulative Regret')
			ax.plot(np.cumsum(np.mean(reg, axis=0)), marker=markers[idx], markevery=200, lw=2, linestyle='solid', color=colors[idx], label=labels[idx])

		ax.grid(True)
		ax.legend()
		plt.savefig(f'data/img/new_{networks[mat_idx].replace(" ", "-")}-reg.svg', format='svg', bbox_inches='tight')
		plt.savefig(f'data/img/new_{networks[mat_idx].replace(" ", "-")}-reg.png', format='png', bbox_inches='tight')
		fig, ax = plt.subplots()

		fig.suptitle(f'{networks[mat_idx].title()} network')
		fig.supxlabel('Timesteps')
		fig.supylabel('Mean estimate error for the best arm')
		# plt.show()


@njit(parallel=True)
def laie(runs: int, num_arms: int, num_timesteps: int, trueMeans: np.ndarray, P: np.ndarray, noisy_agents=np.array([1])) -> tuple:
	'''
	Plays LAIE given the number of runs, number of arms, timesteps, true
	means of arms, and the P matrix of the network. Optimized to work with
	numba.
	'''

	num_agents, _ = P.shape
	regret = np.zeros((runs, num_agents, num_timesteps))

	for run in prange(runs):
		# one experiment of LAIE
		pass

	return regret


@njit(parallel=True)
def dUER(runs: int, num_arms: int, num_timesteps: int, trueMeans: np.ndarray, P: np.ndarray, noisy_agents=np.array([1])) -> tuple:
	'''
	Plays d-UER given the number of runs, number of arms, timesteps, true
	means of arms, and the P matrix of the network. Optimized to work with
	numba.
	'''

	num_agents, _ = P.shape
	regret = np.zeros((runs, num_agents, num_timesteps))
	d = 2

	for run in prange(runs):
		# initialize
		# each action is played once
		estimates = np.zeros((num_agents, num_arms)) # phi
		private_observations = np.zeros((num_agents, num_arms))
		for i in range(num_agents):
			for k in range(num_arms):
				# reward
				# "int(i / num_agents >= 0.5)" puts the first half of the agents in group 0 and the rest in group 1
				private_observations[i, k] = np.random.normal(trueMeans[run, int(i / num_agents >= 0.5), k], 1)
		num_pulls = np.ones(num_arms)
		I = np.zeros(num_agents, dtype=np.int64)

		for t in range(num_timesteps):
			for i in range(num_agents):
				temp_sum = np.zeros((num_agents, num_arms))
				for j in range(num_agents):
					temp_sum[j] = P[i, j] * estimates[j]
				estimates[i] = np.sum(temp_sum, axis=0) + private_observations[i]
				C = np.sqrt(2 * np.log(t) * ((1 / (num_agents * num_pulls)) + (2 * d / np.power(num_pulls, 2))))
				I[i] = np.argmax((estimates[i] / num_pulls) + C)

			# most_freq_arm = stats.mode(I)[0] # not compatible with numba
			# most_freq_arm = I[np.argmax(np.unique(I, return_counts=True)[1])] # not compatible with numba
			most_freq_arm = np.argmax(np.bincount(I)) # compatible with numba
			num_pulls[most_freq_arm] += 1
			private_observations = np.zeros((num_agents, num_arms))
			for i in range(num_agents):
				# reward
				private_observations[i, most_freq_arm] = np.random.normal(trueMeans[run, int(i / num_agents >= 0.5), most_freq_arm], 1)
				regret[run, i, t] = np.abs(trueMeans[run, int(i / num_agents >= 0.5), most_freq_arm] - np.max(trueMeans[run, int(i / num_agents >= 0.5), :]))

	return regret


@njit(parallel=True)
def coopucb2(runs: int, num_arms: int, num_timesteps: int, trueMeans: np.ndarray, P: np.ndarray, noisy_agents=np.array([1])) -> tuple:
	'''
	Plays coopucb2 given the number of runs, number of arms, timesteps, true
	means of arms, and the P matrix of the network. Optimized to work with
	numba.
	'''
	sigma_g = 1		# try 10
	eta = 2		# try 2, 2.2, 3.2
	gamma = 2.9 	# try 1.9, 2.9
	f = lambda t : np.sqrt(np.log(t))
	Geta = 1 - (eta ** 2)/16
	var = 1		# variance for the gaussian distribution behind each arm
	num_agents, _ = P.shape
	x1 = 2 * gamma / Geta

	regret = np.zeros((runs, num_agents, num_timesteps))

	# run coop-ucb2 "runs" number of times
	for run in prange(runs):
		Q = np.zeros((num_agents, num_arms))	# estimated reward
		n = np.zeros((num_agents, num_arms))	# number of times an arm has been selected by each agent
		s = np.zeros((num_agents, num_arms))	# cumulative expected reward
		xsi = np.zeros((num_agents, num_arms))	# number of times that arm has been selected in that timestep
		reward = np.zeros((num_agents, num_arms))	# reward
		bestArmMean = np.max(trueMeans[run])
		bestArmIdx = np.argmax(trueMeans[run])

		for t in range(num_timesteps):
			_t = t - 1 if t > 0 else 0
			if t < num_arms:
				for k in range(num_agents):
					reward[k] = np.zeros(num_arms)
					xsi[k] = np.zeros(num_arms)
					action = t

					reward[k, action] = np.random.normal(trueMeans[run, action], var)
					regret[run, k, t] = bestArmMean - trueMeans[run, action]
					xsi[k, action] += 1
			else:
				for k in range(num_agents):
					for i in range(num_arms):
						x0 = s[k, i] / n[k, i]
						x2 = (n[k, i] + f(_t)) / (num_agents * n[k, i])
						x3 = np.log(_t) / n[k, i]
						C = sigma_g * np.sqrt(x1 * x2 * x3)
						Q[k, i] = x0 + C

					reward[k] = np.zeros(num_arms)
					xsi[k] = np.zeros(num_arms)

					action = np.argmax(Q[k, :])
					reward[k, action] = np.random.normal(trueMeans[run, action], var)
					regret[run, k, t] = bestArmMean - trueMeans[run, action]
					xsi[k, action] += 1

			# # add noise
			# for k in noisy_agents:
			# 	rew[k, :] *= -1 * np.abs(np.random.normal(0, 1))

			# update estimates using running consensus
			for i in range(num_arms):
				n[:, i] = P @ (n[:, i] + xsi[:, i])
				s[:, i] = P @ (s[:, i] + reward[:, i])

	return regret


def spectralGap(P):
	r'''
	spectral gap of P = 1 - second largest SINGULAR VALUE of P
	'''
	singular_values = np.unique(np.linalg.svd(P, compute_uv=False))
	singular_values.sort()
	print(singular_values)
	return 1 - singular_values[-2]

def generateAvgW(A):
	return (1 / np.sum(A, axis=0)) * A

def generateP(A, kappa):
	dmax = np.max(np.sum(A, axis=0))
	L = laplacian(A, normed=False)
	M, _ = np.shape(A)
	I = np.eye(M)

	P = I - (kappa/dmax) * L

	return P, get_rho(P)

def get_rho(P):
	n = P.shape[0]
	_P = P - np.ones((n, n)) * (1/n)
	l = np.abs(np.linalg.eigvals(_P))
	l = l[1 - l > 1e-5]
	return np.max(l)

if __name__ == '__main__':
	main()
