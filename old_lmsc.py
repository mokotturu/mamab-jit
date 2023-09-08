import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import ConnectionPatch
from numba import njit, prange
from scipy.sparse.csgraph import laplacian

from graph_optimization import (fastest_averaging_constant_weight,
                                fdla_weights_symmetric, fmmc_weights,
                                lmsc_weights, max_degree_weights,
                                metropolis_hastings_weights)


def main():
	logging.info(f'started old_lmsc.py')
	N = 100
	runs = 2000
	T = 1000

	# adjacency matrices
	As = [
		# all-to-all
		np.array([
			[0, 1, 1, 1, 1],
			[1, 0, 1, 1, 1],
			[1, 1, 0, 1, 1],
			[1, 1, 1, 0, 1],
			[1, 1, 1, 1, 0],
		]),
		# # house
		# np.array([
		# 	[0, 1, 1, 0, 0],
		# 	[1, 0, 1, 1, 0],
		# 	[1, 1, 0, 0, 1],
		# 	[0, 1, 0, 0, 1],
		# 	[0, 0, 1, 1, 0],
		# ]),
		# star
		np.array([
			[0, 1, 1, 1, 1],
			[1, 0, 0, 0, 0],
			[1, 0, 0, 0, 0],
			[1, 0, 0, 0, 0],
			[1, 0, 0, 0, 0],
		]),
		# 8-agent
		np.array([
			[0, 1, 1, 1, 0, 0, 0, 0],
			[1, 0, 1, 0, 1, 0, 0, 0],
			[1, 1, 0, 1, 1, 0, 0, 0],
			[1, 0, 1, 0, 1, 1, 1, 1],
			[0, 1, 1, 1, 0, 1, 1, 1],
			[0, 0, 0, 1, 1, 0, 1, 1],
			[0, 0, 0, 1, 1, 1, 0, 1],
			[0, 0, 0, 1, 1, 1, 1, 0],
		]),
		np.load('data/saved_networks/2_clusters_adj.npy'),
		np.load('data/saved_networks/3_clusters_adj.npy'),
		np.load('data/saved_networks/4_clusters_adj.npy'),
		np.load('data/saved_networks/5_clusters_adj.npy'),
	]

	# corresponding incidence matrices
	Is = [
		# all-to-all
		np.array([
			[ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0],
			[-1,  0,  0,  0,  1,  1,  1,  0,  0,  0],
			[ 0, -1,  0,  0, -1,  0,  0,  1,  1,  0],
			[ 0,  0, -1,  0,  0, -1,  0, -1,  0,  1],
			[ 0,  0,  0, -1,  0,  0, -1,  0, -1, -1],
		]),
		# # house
		# np.array([
		# 	[ 1,  0,  0,  0, -1,  0],
		# 	[ 0,  0,  0, -1,  1,  1],
		# 	[-1,  1,  0,  0,  0, -1],
		# 	[ 0,  0, -1,  1,  0,  0],
		# 	[ 0, -1,  1,  0,  0,  0],
		# ]),
		# star
		np.array([
			[ 1,  1,  1,  1],
			[-1,  0,  0,  0],
			[ 0, -1,  0,  0],
			[ 0,  0, -1,  0],
			[ 0,  0,  0, -1],
		]),
		# 8-agent
		np.array([
			[  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
			[ -1,  0,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
			[  0, -1,  0, -1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0],
			[  0,  0, -1,  0,  0,  0,  0,  0, -1,  0,  1,  0,  0,  0,  0,  0,  0],
			[  0,  0,  0,  0, -1,  0,  0,  0,  0, -1, -1,  1,  1,  1,  0,  0,  0],
			[  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1,  0,  0,  1,  1,  0],
			[  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1,  0, -1,  0,  1],
			[  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1,  0, -1, -1],
		]),
		np.load('data/saved_networks/2_clusters_inc.npy'),
		np.load('data/saved_networks/3_clusters_inc.npy'),
		np.load('data/saved_networks/4_clusters_inc.npy'),
		np.load('data/saved_networks/5_clusters_inc.npy'),
	]

	networks = [
		'all-to-all',
		# 'house',
		'star',
		'8 agents',
		'2 clusters',
		'3 clusters',
		'4 clusters',
		'5 clusters',
	]

	trueMeans = np.array([np.random.normal(0, 1, N) for _ in range(runs)])
	# trueMeans = np.array([temp for _ in range(runs)])

	# sizing
	SMALL_SIZE = 10
	MEDIUM_SIZE = 14
	LARGE_SIZE = 18

	# plt.rcParams["figure.figsize"] = (15, 8)
	plt.rc('font', size=SMALL_SIZE)
	plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=MEDIUM_SIZE)
	plt.rc('xtick', labelsize=MEDIUM_SIZE)
	plt.rc('ytick', labelsize=MEDIUM_SIZE)

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
		'd',
		'1',
		'D',
		'P',
	]

	inset = [0.425, 0.4, 0.55, 0.55]

	for mat_idx in range(len(As)):
		Ps, rhos, labels = [], [], []

		# uncomment to save an image of the networks
		# G = nx.from_numpy_array(As[mat_idx])
		# pos = nx.spring_layout(G)
		# nx.draw(G, pos)
		# plt.savefig(f'data/img/networks/{networks[mat_idx].replace(" ", "-")}_network.png', format='png')
		# plt.savefig(f'data/img/networks/{networks[mat_idx].replace(" ", "-")}_network.svg', format='svg')
		# plt.clf()

		for k in [0.02]:
			P, rho = generateP(As[mat_idx], kappa=k)
			Ps.append(P)
			rhos.append(rho)
			logging.info(f'{"kappa " + str(k):<20s}: {rho:.3f} {(1 / np.log(1 / rho)):.3f}')
			labels.append(fr'$\kappa$ = {k}')
			logging.info(f'\n{P}')

		# constant edge
		alpha, _, P, rho = fastest_averaging_constant_weight(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		logging.info(f'{"Constant-edge":<20s}: {rho:.3f} {(1 / np.log(1 / rho)):.3f}')
		labels.append(fr'Constant-edge ($\alpha$ = {alpha:.3f})')
		logging.info(f'\n{P}')

		# maximum degree
		alpha, _, P, rho = max_degree_weights(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		logging.info(f'{"Max-degree":<20s}: {rho:.3f} {(1 / np.log(1 / rho)):.3f}')
		labels.append(fr'Maximum-degree ($\alpha$ = {alpha:.3f})')
		logging.info(f'\n{P}')

		# local degree (MH)
		_, P, rho = metropolis_hastings_weights(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		logging.info(f'{"Local-degree":<20s}: {rho:.3f} {(1 / np.log(1 / rho)):.3f}')
		labels.append(fr'Local-degree')
		logging.info(f'\n{P}')

		# fmmc
		_, P, rho = fmmc_weights(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		logging.info(f'{"FMMC":<20s}: {rho:.3f} {(1 / np.log(1 / rho)):.3f}')
		labels.append('FMMC')
		logging.info(f'\n{P}')

		# fdla
		_, P, rho = fdla_weights_symmetric(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		logging.info(f'{"FDLA":<20s}: {rho:.3f} {(1 / np.log(1 / rho)):.3f}')
		labels.append('FDLA')
		# logging.info('\n\n')

		# in_xl, in_xr, in_yb, in_yu = axins_limits[mat_idx]
		fig, ax = plt.subplots()

		means = np.zeros((len(Ps), T))
		means_maxs = np.zeros(len(Ps))

		logging.info(f'Network: {networks[mat_idx]}')
		for idx, P in enumerate(Ps):
			reg, snerr = run(runs, N, T, trueMeans, P)

			reg = np.mean(reg, axis=0)	# mean over runs
			snerr = np.mean(snerr, axis=0)	# mean over runs

			np.save(f'data/data/new_{networks[mat_idx].replace(" ", "-")}_snerr_{labels[idx]}.npy', snerr)
			np.save(f'data/data/new_{networks[mat_idx].replace(" ", "-")}_reg_{labels[idx]}.npy', reg)
			# reg = np.load(f'data/data/new_{networks[mat_idx].replace(" ", "-")}_reg_{labels[idx]}.npy')
			# snerr = np.load(f'data/data/new_{networks[mat_idx].replace(" ", "-")}_snerr_{labels[idx]}.npy')

			means[idx] = np.mean(snerr, axis=0)	# mean over agents
			means_maxs[idx] = np.max(means[idx])

			# plot regret
			fig.suptitle(f'{networks[mat_idx].title()} network')
			fig.supxlabel('Timesteps')
			fig.supylabel('Exploration portion of Q for the best arm')
			ax.plot(np.cumsum(np.mean(reg, axis=0)), marker=markers[idx], markevery=200, lw=2, linestyle='solid', color=colors[idx], label=labels[idx])

		ax.grid(True)
		ax.legend()
		plt.savefig(f'data/img/{networks[mat_idx].replace(" ", "-")}_regret.svg', format='svg', bbox_inches='tight')
		plt.savefig(f'data/img/{networks[mat_idx].replace(" ", "-")}_regret.png', format='png', bbox_inches='tight')
		fig, ax = plt.subplots()

		# fig.suptitle(f'{networks[mat_idx].title()} network')
		fig.supxlabel('Timesteps')
		fig.supylabel('Mean estimate error for the best arm')


		width = 2
		axins = ax.inset_axes(inset)
		axins.spines['bottom'].set_linewidth(width)
		axins.spines['top'].set_linewidth(width)
		axins.spines['right'].set_linewidth(width)
		axins.spines['left'].set_linewidth(width)
		axins.tick_params(width=width)
		axins.grid(which='both', axis='both')

		ax.grid(True)
		vert_lines = np.zeros(len(Ps))
		yb, yu = 1, -1

		for idx, P in enumerate(Ps):
			decreasing_arr = means[idx, np.argmax(means[idx]):]
			vert_lines[idx] = np.argmax(means[idx]) + np.argmax(decreasing_arr < 0.05 * np.max(means_maxs))

			ax.plot(means[idx], marker=markers[idx], markevery=200, lw=2, linestyle='solid', color=colors[idx], label=labels[idx])
			ax.axvline(vert_lines[idx], color=colors[idx], marker=markers[idx], markevery=0.05, linestyle='dashed', lw=2)
			ax.set_ylim(-0.01, 0.25)

			axins.plot(means[idx], marker=markers[idx], markevery=5, lw=2, linestyle='solid', color=colors[idx], label=labels[idx])
			axins.axvline(vert_lines[idx], color=colors[idx], marker=markers[idx], markevery=0.001, linestyle='dashed', lw=2)
			
			if yb > means[idx, int(np.floor(vert_lines[idx]))]:
				yb = means[idx, int(np.floor(vert_lines[idx]))]
			if yu < means[idx, int(np.floor(vert_lines[idx]))]:
				yu = means[idx, int(np.floor(vert_lines[idx]))]

		logging.info(f'\n{np.array(list(zip(labels, vert_lines)))}')
		vert_lines = np.sort(vert_lines)
		in_xl, in_xr, in_yb, in_yu = vert_lines[0] - 2, vert_lines[-2] + 2, yb - 0.00125, yu + 0.00125

		ax.fill_between((in_xl, in_xr), in_yb, in_yu, facecolor='black', alpha=0.2)
		axins.set_xlim((in_xl, in_xr))
		axins.set_ylim((in_yb, in_yu))

		con = ConnectionPatch(xyA=(in_xr, in_yu), coordsA=ax.transData, xyB=(in_xl, in_yb), coordsB=axins.transData, color='black')
		fig.add_artist(con)

		# ax.legend(bbox_to_anchor=(0.92, 0.5), loc='center left', bbox_transform=fig.transFigure)
		plt.savefig(f'data/img/{networks[mat_idx].replace(" ", "-")}.svg', format='svg', bbox_inches='tight')
		plt.savefig(f'data/img/{networks[mat_idx].replace(" ", "-")}.png', format='png', bbox_inches='tight')
		fig.clear()
		logging.info(f'finished experiments for {mat_idx}')

	logging.info('finished experiments')


@njit(parallel=True)
def run(runs: int, N: int, T: int, trueMeans: np.ndarray, P: np.ndarray) -> tuple:
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
	var = 1		# variance for the gaussian distribution behind each arm
	M, _ = P.shape

	omega = 2

	reg = np.zeros((runs, M, T))
	snerr = np.zeros((runs, M, T))

	# run coop-ucb2 "runs" number of times
	for run in prange(runs):
		Q = np.zeros((M, N))	# estimated reward
		n = np.zeros((M, N))	# number of times an arm has been selected by each agent
		s = np.zeros((M, N))	# cumulative expected reward
		xsi = np.zeros((M, N))	# number of times that arm has been selected in that timestep
		rew = np.zeros((M, N))	# reward
		bestArm = np.max(trueMeans[run])
		bestArmIdx = np.argmax(trueMeans[run])

		for t in range(T):
			_t = t - 1 if t > 0 else 0
			if t < N:
				for k in range(M):
					rew[k] = np.zeros(N)
					xsi[k] = np.zeros(N)
					action = t

					rew[k, action] = np.random.normal(trueMeans[run, action], var)
					# rew[run, k, action] = np.random.normal(trueMeans[run, action], var) if k != 1 else np.random.normal(trueMeans[run, action], var) + np.random.normal(0, 1.0)
					reg[run, k, t] = bestArm - trueMeans[run, action]
					xsi[k, action] += 1
			else:
				for k in range(M):
					for i in range(N):
						# Q[k, i, t] = (s[k, i] / n[k, i]) + sigma_g * (np.sqrt((2 * gamma / Geta) * ((n[k, i] + f(t - 1)) / (M * n[k, i])) * (np.log(t - 1) / n[k, i])))
						x0 = s[k, i] / n[k, i]
						# x1 = 2 * gamma / Geta
						x2 = (n[k, i] + f(_t)) / (M * n[k, i])
						x3 = np.log(_t) / n[k, i]
						_explr = sigma_g * np.sqrt(omega * x2 * x3)
						Q[k, i] = x0 + _explr

					rew[k] = np.zeros(N)
					xsi[k] = np.zeros(N)

					action = np.argmax(Q[k, :])
					rew[k, action] = np.random.normal(trueMeans[run, action], var)
					reg[run, k, t] = bestArm - trueMeans[run, action]
					xsi[k, action] += 1

					snerr[run, k, t] = np.abs(trueMeans[run, bestArmIdx] - (s[k, bestArmIdx] / n[k, bestArmIdx]))

			# update estimates using running consensus
			for i in range(N):
				n[:, i] = P @ (n[:, i] + xsi[:, i])
				s[:, i] = P @ (s[:, i] + rew[:, i])

	return reg, snerr

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

def max_degree(A):
	n, _ = A.shape
	I = np.eye(n)
	L = A @ A.T
	dmax = np.max(np.sum(np.abs(A), axis=1))
	alpha = 1 / dmax
	W = I - alpha * L
	rho = get_rho(W)
	return W, rho

if __name__ == '__main__':
	logging.basicConfig(filename='output_lmsc.log',filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
	np.set_printoptions(linewidth=9999999)
	main()
