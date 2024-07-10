import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import ConnectionPatch
from numba import njit, prange
from scipy.optimize import curve_fit
from scipy.sparse.csgraph import laplacian

from graph_optimization import (fastest_averaging_constant_weight,
                                fdla_weights_symmetric, fmmc_weights,
                                lmsc_weights, max_degree_weights,
                                metropolis_hastings_weights)


def main():
	logging.info(f'started old_lmsc.py')
	N = 10
	runs = 10000
	T = 500

	# adjacency matrices
	As = [
		np.load('data/saved_networks/all_to_all_adj.npy'),
		np.load('data/saved_networks/star_adj.npy'),
		np.load('data/saved_networks/house_adj.npy'),
		np.load('data/saved_networks/ring_adj.npy'),
		np.load('data/saved_networks/line_adj.npy'),
		np.load('data/saved_networks/eight_agents_adj.npy'),
		# np.load('data/saved_networks/2_clusters_adj.npy'),
		# np.load('data/saved_networks/3_clusters_adj.npy'),
		# np.load('data/saved_networks/4_clusters_adj.npy'),
		# np.load('data/saved_networks/5_clusters_adj.npy'),
		# np.load('data/saved_networks/large_50_adj.npy'),
		# np.array([
		# 	[0.0, 1.0, 1.0],
		# 	[1.0, 0.0, 1.0],
		# 	[1.0, 1.0, 0.0],
		# ])
		# np.load('data/saved_networks/tadpole_adj.npy')
	]

	# corresponding incidence matrices
	Is = [
		np.load('data/saved_networks/all_to_all_inc.npy'),
		np.load('data/saved_networks/star_inc.npy'),
		np.load('data/saved_networks/house_inc.npy'),
		np.load('data/saved_networks/ring_inc.npy'),
		np.load('data/saved_networks/line_inc.npy'),
		np.load('data/saved_networks/eight_agents_inc.npy'),
		# np.load('data/saved_networks/2_clusters_inc.npy'),
		# np.load('data/saved_networks/3_clusters_inc.npy'),
		# np.load('data/saved_networks/4_clusters_inc.npy'),
		# np.load('data/saved_networks/5_clusters_inc.npy'),
		# np.load('data/saved_networks/large_50_inc.npy'),
		# np.array([
		# 	[1.0, -1.0, 0.0],
		# 	[0.0, 1.0, -1.0],
		# 	[-1.0, 0.0, 1.0],
		# ])
		# np.load('data/saved_networks/tadpole_inc.npy')
	]

	networks = [
		'All-to-all',
		'Star',
		'House',
		'Ring',
		'Path',
		'8-agents',
		# '2 clusters',
		# '3 clusters',
		# '4 clusters',
		# '5 clusters',
		# 'Large 50'
		# 'Triangle'
		# 'Tadpole'
	]

	competencies = [
		np.array([0.8 if i in [0] else 1.0 for i in range(5)]),
		np.array([0.8 if i in [2, 12, 22, 32] else 1.0 for i in range(41)]),
		np.array([0.8 if i in [0] else 1.0 for i in range(5)]),
		np.array([0.8 if i in [2, 12, 22, 32] else 1.0 for i in range(41)]),
		np.array([0.8 if i in [2, 12, 22, 32] else 1.0 for i in range(41)]),
		np.array([0.8 if i in [2, 12, 22] else 1.0 for i in range(31)]),
		np.array([0.5 if i in [40] else 1.0 for i in range(41)]),
		np.array([0.5 if i in [0] else 1.0 for i in range(40)]),
		np.array([0.5, 1.0, 1.0]),
		np.array([0.5, 1.0, 1.0, 1.0])
	]

	bandit_variance = 1
	real_bandit_stage = 2

	trueMeans = np.array([np.random.normal(0, bandit_variance, N) for _ in range(runs)])

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

	for mat_idx, A in enumerate(As):
		Ps, rhos, labels, normal_labels = [], [], [], []
		# print the network name
		logging.info(f'Network: {networks[mat_idx]}')

		# uncomment to save an image of the networks
		# G = nx.from_numpy_array(A)
		# pos = nx.spring_layout(G)
		# nx.draw(G, pos)
		# plt.savefig(f'data/img/networks/png/{networks[mat_idx].replace(" ", "-")}_network.png', format='png')
		# plt.savefig(f'data/img/networks/svg/{networks[mat_idx].replace(" ", "-")}_network.svg', format='svg')
		# plt.clf()

		for k in [0.02]:
			P, rho = generateP(A, kappa=k)
			Ps.append(P)
			rhos.append(rho)
			logging.info(f'{"kappa " + str(k):<20s}: {rho:.3f} {(1 / np.log(1 / rho)):.3f}')
			labels.append(fr'$\kappa$ = {k}')
			normal_labels.append(fr'$\kappa$ = {k}')
			logging.info(f'\n{P}')

		# constant edge
		alpha, _, P, rho = fastest_averaging_constant_weight(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		best_constant_mat = np.array(np.round(P, decimals=3), dtype=np.float64)
		rhos.append(rho)
		logging.info(f'{"Constant-edge":<20s}: {rho:.3f} {(1 / np.log(1 / rho)):.3f}')
		labels.append(fr'Constant-edge ($\alpha$ = {alpha:.3f})')
		normal_labels.append(fr'Constant-edge')
		logging.info(f'\n{P}')

		# maximum degree
		alpha, _, P, rho = max_degree_weights(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		logging.info(f'{"Max-degree":<20s}: {rho:.3f} {(1 / np.log(1 / rho)):.3f}')
		labels.append(fr'Maximum-degree ($\alpha$ = {alpha:.3f})')
		normal_labels.append(fr'Maximum-degree')
		logging.info(f'\n{P}')

		# local degree (MH)
		_, P, rho = metropolis_hastings_weights(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		logging.info(f'{"Local-degree":<20s}: {rho:.3f} {(1 / np.log(1 / rho)):.3f}')
		labels.append(fr'Local-degree')
		normal_labels.append(fr'Local-degree')
		logging.info(f'\n{P}')

		# fmmc
		_, P, rho = fmmc_weights(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		fmmc_weights_mat = np.array(np.round(P, decimals=3), dtype=np.float64)
		rhos.append(rho)
		logging.info(f'{"FMMC":<20s}: {rho:.3f} {(1 / np.log(1 / rho)):.3f}')
		labels.append('FMMC')
		normal_labels.append('FMMC')
		logging.info(f'\n{P}')

		# fdla
		_, P, rho = fdla_weights_symmetric(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		fdla_weights_mat = np.array(np.round(P, decimals=3), dtype=np.float64)
		rhos.append(rho)
		logging.info(f'{"FDLA":<20s}: {rho:.3f} {(1 / np.log(1 / rho)):.3f}')
		labels.append('FDLA')
		normal_labels.append('FDLA')
		logging.info(f'\n{P}')

		# # my weights
		# P = np.array([
		# 	[0.28108112, 0.71891888, 0.        , 0.        ],
		# 	[0.00985469, 0.50163124, 0.18622186, 0.30229222],
		# 	[0.        , 0.33272083, 0.33424009, 0.33303908],
		# 	[0.        , 0.28212891, 0.19196279, 0.52590829],
		# ])
		# rho = get_rho(P)
		# Ps.append(P)
		# rhos.append(rho)
		# logging.info(f'{"OPT":<20s}: {rho} {(1 / np.log(1 / rho))}')
		# labels.append('OPT')
		# normal_labels.append('OPT')
		# logging.info(f'\n{P}')

		# lmsc
		# _, P = lmsc_weights(Is[mat_idx])
		P = np.eye(A.shape[0]) - (2 / (A.shape[0] + 1)) * laplacian(A, normed=False)
		lmsc_weight_mat = np.array(np.round(P, decimals=3), dtype=np.float64)
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		logging.info(f'{"LMSC":<20s}: {rho} {(1 / np.log(1 / rho))}')
		labels.append('LMSC')
		normal_labels.append('LMSC')
		logging.info(f'\n{P}')

		# in_xl, in_xr, in_yb, in_yu = axins_limits[mat_idx]
		fig, ax = plt.subplots()

		means = np.zeros((len(Ps), T))
		means_maxs = np.zeros(len(Ps))
		asymptotes = np.zeros(len(Ps))
		team_percent_optimal_actions = np.zeros((len(Ps), T))

		logging.info(f'Network: {networks[mat_idx]}')
		for P_idx, P in enumerate(Ps):
			reg, snerr, percent_optimal_action = run(runs, N, T, trueMeans, P, competencies[mat_idx])

			reg = np.mean(reg, axis=0) # mean over runs
			snerr = np.mean(snerr, axis=0) # mean over runs
			percent_optimal_action = np.mean(percent_optimal_action, axis=0) # mean over runs
			team_percent_optimal_actions[P_idx] = np.mean(percent_optimal_action, axis=0) # mean over agents

			np.save(f'data/data/LMSC_TEST_{networks[mat_idx].replace(" ", "-")}_reg_{labels[P_idx]}_stage_{real_bandit_stage}.npy', reg)
			np.save(f'data/data/LMSC_TEST_{networks[mat_idx].replace(" ", "-")}_snerr_{labels[P_idx]}_stage_{real_bandit_stage}.npy', snerr)
			np.save(f'data/data/LMSC_TEST_{networks[mat_idx].replace(" ", "-")}_percent_optimal_action_{labels[P_idx]}_stage_{real_bandit_stage}.npy', team_percent_optimal_actions[P_idx])

			# reg = np.load(f'data/data/{networks[mat_idx].replace(" ", "-")}_reg_{labels[P_idx]}.npy')
			# snerr = np.load(f'data/data/{networks[mat_idx].replace(" ", "-")}_snerr_{labels[P_idx]}.npy')
			# team_percent_optimal_actions[P_idx] = np.load(f'data/data/{networks[mat_idx].replace(" ", "-")}_percent_optimal_action_{labels[P_idx]}.npy')


			means[P_idx] = np.mean(snerr, axis=0)	# mean over agents
			means_maxs[P_idx] = np.max(means[P_idx])
			_data = means[P_idx, N:]
			popt, pcov = curve_fit(exp_decay, np.arange(len(_data)), _data, p0=(1, 1e-2, 1), maxfev=10000)
			asymptotes[P_idx] = popt[2]


			# plot regret
			fig.suptitle(f'{networks[mat_idx].title()} network, {runs} runs, {T} timesteps, {N} arms, {A.shape[0]} agents')
			fig.supxlabel('Timesteps')
			fig.supylabel('Regret')
			ax.plot(np.cumsum(np.mean(reg, axis=0)), marker=markers[P_idx], markevery=200, lw=2, linestyle='solid', color=colors[P_idx], label=labels[P_idx])

		ax.grid(True)
		ax.legend()
		plt.savefig(f'data/img/svg/LMSC_TEST_{networks[mat_idx].replace(" ", "-")}_regret_stage_{real_bandit_stage}.svg', format='svg', bbox_inches='tight')
		plt.savefig(f'data/img/png/LMSC_TEST_{networks[mat_idx].replace(" ", "-")}_regret_stage_{real_bandit_stage}.png', format='png', bbox_inches='tight')
		fig, ax = plt.subplots()

		# fig.suptitle(f'{networks[mat_idx].title()} network, {runs} runs, {T} timesteps, {N} arms, {A.shape[0]} agents')
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

		logging.info(f'asymptotes: {asymptotes}')

		for P_idx, P in enumerate(Ps):
			# vert_lines[P_idx] = np.argmax(means[P_idx]) + np.argmax(decreasing_arr < 0.5 * np.max(means_maxs))
			# vert_line is where the curve is within 5% of the asymptote
			vert_lines[P_idx] = np.argmax(np.abs(means[P_idx, N:] - np.min(asymptotes)) < 0.05 * np.min(asymptotes)) + N
			if vert_lines[P_idx] == N:
				# if the curve never reaches 5% of the asymptote, then take the last value
				vert_lines[P_idx] = T - 1
			logging.info(f'vertical line: {vert_lines[P_idx]}')

			ax.plot(means[P_idx], marker=markers[P_idx], markevery=200, lw=2, linestyle='solid', color=colors[P_idx], label=labels[P_idx])
			ax.axvline(vert_lines[P_idx], color=colors[P_idx], marker=markers[P_idx], markevery=0.05, linestyle='dashed', lw=2)
			# ax.set_ylim(-0.01, 0.25)

			axins.plot(means[P_idx], linestyle='solid', color=colors[P_idx], label=labels[P_idx])
			axins.axvline(vert_lines[P_idx], color=colors[P_idx], linestyle='dashed')

			if yb > means[P_idx, int(np.floor(vert_lines[P_idx]))]:
				yb = means[P_idx, int(np.floor(vert_lines[P_idx]))]
			if yu < means[P_idx, int(np.floor(vert_lines[P_idx]))]:
				yu = means[P_idx, int(np.floor(vert_lines[P_idx]))]

		# logging.info(f'\n{np.array(list(zip(labels, vert_lines)))}')
		vert_lines = np.sort(vert_lines)
		in_xl, in_xr, in_yb, in_yu = vert_lines[0] - 1, vert_lines[-2] + 1, yb - 0.00125, yu + 0.00125

		ax.fill_between((in_xl, in_xr), in_yb, in_yu, facecolor='black', alpha=0.2)
		axins.set_xlim((in_xl, in_xr))
		axins.set_ylim((in_yb, in_yu))

		con = ConnectionPatch(xyA=(in_xr, in_yu), coordsA=ax.transData, xyB=(in_xl, in_yb), coordsB=axins.transData, color='black')
		fig.add_artist(con)

		ax.legend(bbox_to_anchor=(0.5, 1.05), loc='upper center', bbox_transform=fig.transFigure, ncols=3)
		plt.savefig(f'data/img/svg/LMSC_TEST_{networks[mat_idx].replace(" ", "-")}_snerr_stage_{real_bandit_stage}.svg', format='svg', bbox_inches='tight')
		plt.savefig(f'data/img/png/LMSC_TEST_{networks[mat_idx].replace(" ", "-")}_snerr_stage_{real_bandit_stage}.png', format='png', bbox_inches='tight')
		plt.savefig(f'data/img/pdf/LMSC_TEST_{networks[mat_idx].replace(" ", "-")}_snerr_stage_{real_bandit_stage}.pdf', format='pdf', bbox_inches='tight')


		fig, ax = plt.subplots()
		for P_idx, P in enumerate(Ps):
			ax.plot(team_percent_optimal_actions[P_idx], lw=2, linestyle='solid', label=labels[P_idx], color=colors[P_idx], marker=markers[P_idx], markevery=200)
		ax.grid(True)
		# ax.legend()
		fig.supxlabel('Timesteps')
		fig.supylabel('Percent optimal action')
		# fig.suptitle(f'{networks[mat_idx].title()} network, {runs} runs, {T} timesteps, {N} arms, {A.shape[0]} agents')

		# ax.set_xlim(4800, 5000)
		# ax.set_ylim(0.9525, 0.9575)

		ax.legend(bbox_to_anchor=(0.5, 1.05), loc='upper center', bbox_transform=fig.transFigure, ncols=3)
		plt.savefig(f'data/img/png/LMSC_TEST_{networks[mat_idx].replace(" ", "-")}_percent_optimal_action_stage_{real_bandit_stage}.png', format='png', bbox_inches='tight')
		plt.savefig(f'data/img/pdf/LMSC_TEST_{networks[mat_idx].replace(" ", "-")}_percent_optimal_action_stage_{real_bandit_stage}.pdf', format='pdf', bbox_inches='tight')

		fig.clear()
		logging.info(f'finished experiments for {mat_idx}')

	logging.info('finished experiments')


@njit(parallel=True)
def run(runs: int, N: int, T: int, trueMeans: np.ndarray, P: np.ndarray, competencies: np.ndarray) -> tuple:
	'''
	Plays coopucb2 given the number of runs, number of arms, timesteps, true
	means of arms, and the P matrix of the network. Optimized to work with
	numba.
	'''
	sigma_g = 1		# try 10
	eta = 2		# try 2, 2.2, 3.2
	gamma = 2.0 	# try 1.9, 2.9
	f = lambda t : np.sqrt(np.log(t))
	var = 1		# variance for the gaussian distribution behind each arm
	M, _ = P.shape
	competencies = np.ones(M)
	# get the agent with the highest degree
	degrees = np.array([np.count_nonzero(row) for row in P])
	max_degree_agent = np.argmin(degrees)

	G_eta = 1 - (eta ** 2) / 16
	c0 = 2 * gamma / G_eta

	reg = np.zeros((runs, M, T))
	snerr = np.zeros((runs, M, T))

	times_best_arm_selected = np.zeros((runs, M), dtype=np.int16)
	percent_optimal_action = np.zeros((runs, M, T))

	# run coop-ucb2 "runs" number of times
	for run in prange(runs):
		Q = np.zeros((M, N))	# estimated reward
		n = np.zeros((M, N))	# number of times an arm has been selected by each agent
		s = np.zeros((M, N))	# cumulative expected reward
		xsi = np.zeros((M, N))	# number of times that arm has been selected in that timestep
		rew = np.zeros((M, N))	# reward
		best_arm = np.max(trueMeans[run])
		best_arm_idx = np.argmax(trueMeans[run])

		for t in range(T):
			_t = t - 1 if t > 0 else 0
			if t < N:
				for k in range(M):
					rew[k] = np.zeros(N)
					xsi[k] = np.zeros(N)
					action = t

					# rew[k, action] = np.random.normal(trueMeans[run, action], var)
					rew[k, action] = np.random.normal(
						trueMeans[run, action] * (competencies[k] + np.random.randn()),
						var / (competencies[k] * np.abs(np.random.randn()))
					) if competencies[k] < 1 else np.random.normal(trueMeans[run, action], var / competencies[k])
					rew[k, action] += np.random.randn()
					reg[run, k, t] = best_arm - trueMeans[run, action]
					xsi[k, action] += 1
					if action == best_arm_idx:
						times_best_arm_selected[run, k] += 1
			else:
				for k in range(M):
					for i in range(N):
						# Q[k, i, t] = (s[k, i] / n[k, i]) + sigma_g * (np.sqrt((2 * gamma / Geta) * ((n[k, i] + f(t - 1)) / (M * n[k, i])) * (np.log(t - 1) / n[k, i])))
						x0 = s[k, i] / n[k, i]
						# x1 = 2 * gamma / Geta
						c1 = (n[k, i] + f(_t)) / (M * n[k, i])
						c2 = np.log(_t) / n[k, i]
						confidence_bound = sigma_g * np.sqrt(c0 * c1 * c2)
						Q[k, i] = x0 + confidence_bound

					rew[k] = np.zeros(N)
					xsi[k] = np.zeros(N)

					action = np.argmax(Q[k, :])
					# rew[k, action] = np.random.normal(trueMeans[run, action], var)
					rew[k, action] = np.random.normal(
						trueMeans[run, action] * (competencies[k] + np.random.randn()),
						var / (competencies[k] * np.abs(np.random.randn()))
					) if competencies[k] < 1 else np.random.normal(trueMeans[run, action], var / competencies[k])
					# add noise to max degree agent
					# if k == max_degree_agent:
					rew[k, action] += np.random.randn()
					reg[run, k, t] = best_arm - trueMeans[run, action]
					xsi[k, action] += 1

					snerr[run, k, t] = np.abs(trueMeans[run, best_arm_idx] - (s[k, best_arm_idx] / n[k, best_arm_idx]))

					if action == best_arm_idx:
						times_best_arm_selected[run, k] += 1


			percent_optimal_action[run, :, t] = times_best_arm_selected[run, :] / (t + 1)

			# update estimates using running consensus
			for i in range(N):
				n[:, i] = P @ (n[:, i] + xsi[:, i])
				s[:, i] = P @ (s[:, i] + rew[:, i])

	return reg, snerr, percent_optimal_action

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

def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

if __name__ == '__main__':
	logging.basicConfig(filename='output_lmsc.log',filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
	np.set_printoptions(linewidth=np.inf, threshold=np.inf)
	main()
