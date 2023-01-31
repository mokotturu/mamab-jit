from time import ctime, time

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
from numba import njit, prange
from scipy.sparse.csgraph import laplacian

from graph_optimization import fdla_weights_symmetric, fmmc_weights


def main():
	N = 2
	runs = 10000
	T = 200

	# adjacency matrices
	As = [
		# all-to-all
		# np.array([
		# 	[0, 1, 1, 1, 1],
		# 	[1, 0, 1, 1, 1],
		# 	[1, 1, 0, 1, 1],
		# 	[1, 1, 1, 0, 1],
		# 	[1, 1, 1, 1, 0],
		# ])
		# star
		np.array([
			[0, 1, 1, 1, 1],
			[1, 0, 0, 0, 0],
			[1, 0, 0, 0, 0],
			[1, 0, 0, 0, 0],
			[1, 0, 0, 0, 0],
		])
		# 8-agent
		# np.array([
		# 	[0, 1, 1, 1, 0, 0, 0, 0],
		# 	[1, 0, 1, 0, 1, 0, 0, 0],
		# 	[1, 1, 0, 1, 1, 0, 0, 0],
		# 	[1, 0, 1, 0, 1, 1, 1, 1],
		# 	[0, 1, 1, 1, 0, 1, 1, 1],
		# 	[0, 0, 0, 1, 1, 0, 1, 1],
		# 	[0, 0, 0, 1, 1, 1, 0, 1],
		# 	[0, 0, 0, 1, 1, 1, 1, 0],
		# ])
	]

	# corresponding incidence matrices
	Is = [
		# all-to-all
		# np.array([
		# 	[ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0],
		# 	[-1,  0,  0,  0,  1,  1,  1,  0,  0,  0],
		# 	[ 0, -1,  0,  0, -1,  0,  0,  1,  1,  0],
		# 	[ 0,  0, -1,  0,  0, -1,  0, -1,  0,  1],
		# 	[ 0,  0,  0, -1,  0,  0, -1,  0, -1, -1],
		# ])
		# star
		np.array([
			[ 1,  1,  1,  1],
			[-1,  0,  0,  0],
			[ 0, -1,  0,  0],
			[ 0,  0, -1,  0],
			[ 0,  0,  0, -1],
		])
		# 8-agent
		# np.array([
		# 	[  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		# 	[ -1,  0,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		# 	[  0, -1,  0, -1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0],
		# 	[  0,  0, -1,  0,  0,  0,  0,  0, -1,  0,  1,  0,  0,  0,  0,  0,  0],
		# 	[  0,  0,  0,  0, -1,  0,  0,  0,  0, -1, -1,  1,  1,  1,  0,  0,  0],
		# 	[  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1,  0,  0,  1,  1,  0],
		# 	[  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1,  0, -1,  0,  1],
		# 	[  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1,  0, -1, -1],
		# ])
	]

	M, _ = As[0].shape

	networks = [
		'all-to-all',
		'star',
		'8-agents',
	]

	# temp = np.array([-0.5, 0.5])
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

	np.set_printoptions(threshold=99999999999999999)

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

	# fig, axss = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(16, 6), constrained_layout=True)
	# plt.title('All-to-All network')
	# for ax in fig.get_axes():
	# 	ax.label_outer()
	# axs0 = axss
	# axs1 = axss[1].flatten()

	print(f'Simulation started at {ctime(time())}')

	labels = [
		'κ = 0.02',
		'κ = 0.3',
		'κ = 0.9',
		'FMMC',
		'FDLA',
	]

	markers = [
		'o',
		'^',
		's',
		'x',
		'v',
	]

	# P0, rho0 = generateP(As[0], kappa=0.02)
	# _, P1, rho1 = fdla_weights_symmetric(Is[0])
	# _, P2, rho2 = fmmc_weights(Is[0])

	Ps, rhos = [], []

	for k in [0.02, 0.3, 0.9]:
		P, rho = generateP(As[0], kappa=k)
		Ps.append(P)
		rhos.append(rho)
		print(f'MYOUT: kappa = {k}, rho = {rho}, tau = {1 / np.log(1 / rho)}')

	# fmmc
	_, P, rho = fmmc_weights(Is[0])
	Ps.append(P)
	rhos.append(rho)
	print(f'MYOUT: rho = {rho}, tau = {1 / np.log(1 / rho)}')

	# fdla
	_, P, rho = fdla_weights_symmetric(Is[0])
	Ps.append(P)
	rhos.append(rho)
	print(f'MYOUT: rho = {rho}, tau = {1 / np.log(1 / rho)}')

	### ZOOM IN PLOT
	fig, ax = plt.subplots()

	# axins plot limits
	# in_xl, in_xr, in_yb, in_yu = 15, 53, 0, 0.02
	in_xl, in_xr, in_yb, in_yu = 20, 102, 0, 0.01
	# in_xl, in_xr, in_yb, in_yu = 13, 60, 0, 0.01

	# insets
	inset = [0.425, 0.4, 0.55, 0.55]
	# inset = [0.425, 0.4, 0.55, 0.55]
	# inset = [0.425, 0.4, 0.55, 0.55]

	fig.suptitle('Star network')
	fig.supxlabel('Timesteps')
	fig.supylabel('Mean estimate error for the best arm')

	# ax.arrow(in_xr, in_yu, dx, dy, width=0.0005, head_width=0.003, head_length=3, color='black')

	axins = ax.inset_axes(inset)

	width = 2
	axins.spines['bottom'].set_linewidth(width)
	axins.spines['top'].set_linewidth(width) 
	axins.spines['right'].set_linewidth(width)
	axins.spines['left'].set_linewidth(width)
	axins.tick_params(width=width)

	ax.grid(True)
	axins.grid(which='both', axis='both')
	# ax.indicate_inset_zoom(axins, edgecolor='black')
	ax.fill_between((in_xl, in_xr), in_yb, in_yu, facecolor='black', alpha=0.2)


	### SUBPLOTS
	# fig = plt.figure	(figsize=(6.4, 9.6))
	# fig.suptitle('8-agent network')
	# fig.supxlabel('Timesteps')
	# fig.supylabel('Mean estimation error for the best arm')

	# sub1 = fig.add_subplot(211)
	# sub1.set_ylim(-0.01, 0.12)
	# sub1.grid()
	
	# sub2 = fig.add_subplot(212)
	# sub2.set_ylim(-0.01, 0.12)
	# sub2.grid()

	# sub2.fill_between((0, 30), -0.01, 0.12, facecolor='black', alpha=0.2)
	# con1 = ConnectionPatch(xyA=(0, 0.12), coordsA=ax.transData, xyB=(0, 0.12), coordsB=axins.transData, color='black')
	# con2 = ConnectionPatch(xyA=(30, -0.01), coordsA=ax.transData, xyB=(30, -0.01), coordsB=axins.transData, color='black')
	# fig.add_artist(con1)
	# fig.add_artist(con2)

	means = np.zeros((len(Ps), T))
	means_maxs = np.zeros(len(Ps))

	for idx, (P, rho) in enumerate(zip(Ps, rhos)):
		_, Q, snerr, s, n = run(runs, N, T, trueMeans, P)

		# Q = np.mean(Q, axis=0)
		snerr = np.mean(snerr, axis=0)	# mean over runs

		np.save(f'data/star_snerr_{labels[idx]}.npy', snerr)
		# snerr = np.load(f'data/star_snerr_{labels[idx]}.npy')
		means[idx] = np.mean(snerr, axis=0)	# mean over agents
		means_maxs[idx] = np.max(means[idx])

	for idx, (P, rho) in enumerate(zip(Ps, rhos)):
		# tau = 1 / (np.log(1 / rho))
		decreasing_arr = means[idx, np.argmax(means[idx]):]
		vert_line = np.argmax(means[idx]) + np.argmax(decreasing_arr < 0.05 * np.max(means_maxs))

		ax.plot(means[idx], marker=markers[idx], markevery=5, lw=2, linestyle='solid', color=colors[idx], label=labels[idx])
		ax.axvline(vert_line, color=colors[idx], linestyle='dashed', lw=2, alpha=0.7)
		ax.set_ylim(-0.01, 0.12)

		axins.plot(means[idx], marker=markers[idx], markevery=5, lw=2, linestyle='solid', color=colors[idx], label=labels[idx])
		axins.axvline(vert_line, color=colors[idx], linestyle='dashed', lw=2, alpha=0.7)
		axins.set_ylim(-0.01, 0.12)
		print(P)

		### SUBPLOT 1
		# sub1.plot(mean, marker=markers[idx], markevery=5, lw=2, linestyle='solid', color=colors[idx], label=labels[idx])
		# sub1.set_xlim(0, 30)


		# ### SUBPLOT 2
		# sub2.plot(mean, marker=markers[idx], markevery=20, lw=2, linestyle='solid', color=colors[idx], label=labels[idx])
		# # sub2.set_xlim(0, 200)

		# if tau <= 30:
		# 	sub1.axvline(tau, color=colors[idx], linestyle='dashed', lw=2, alpha=0.7)

		# sub2.axvline(tau, color=colors[idx], linestyle='dashed', lw=2, alpha=0.7)

	axins.set_xlim((in_xl, in_xr))
	axins.set_ylim((in_yb, in_yu))

	con = ConnectionPatch(xyA=(in_xr, in_yu), coordsA=ax.transData, xyB=(in_xl, in_yb), coordsB=axins.transData, color='black')
	fig.add_artist(con)

	print(f'Simulation ended at {ctime(time())}')
	plt.tight_layout()
	# sub1.legend()
	axins.legend()
	plt.savefig('img/star-errors.svg', format='svg')
	plt.savefig('img/star-errors.png', format='png')
	# plt.show()


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
	var = 1.0		# variance for the gaussian distribution behind each arm
	M, _ = P.shape

	omega = 2

	reg = np.zeros((runs, M, T))
	Q = np.zeros((runs, M, N, T))	# estimated reward
	snerr = np.zeros((runs, M, T))
	# plotQ = np.zeros((runs, M, N, T))
	# plotExplore = np.zeros((runs, M, N, T))

	n = np.zeros((runs, M, N))	# number of times an arm has been selected by each agent
	s = np.zeros((runs, M, N))	# cumulative expected reward
	xsi = np.zeros((runs, M, N))	# number of times that arm has been selected in that timestep
	rew = np.zeros((runs, M, N))	# reward

	# run coop-ucb2 "runs" number of times
	for run in prange(runs):
		# print(f'run {run}')
		bestArm = np.max(trueMeans[run])
		bestArmIdx = np.argmax(trueMeans[run])

		for t in range(T):
			if t < N:
				for k in range(M):
					rew[run, k] = np.zeros(N)
					xsi[run, k] = np.zeros(N)
					action = t
					rew[run, k, action] = np.random.normal(trueMeans[run, action], var)
					reg[run, k, t] = bestArm - trueMeans[run, action]
					xsi[run, k, action] += 1
			else:
				for k in range(M):
					for i in range(N):
						# Q[k, i, t] = (s[k, i] / n[k, i]) + sigma_g * (np.sqrt((2 * gamma / Geta) * ((n[k, i] + f(t - 1)) / (M * n[k, i])) * (np.log(t - 1) / n[k, i])))
						x0 = s[run, k, i] / n[run, k, i]
						# x1 = 2 * gamma / Geta
						x2 = (n[run, k, i] + f(t - 1)) / (M * n[run, k, i])
						x3 = np.log(t - 1) / n[run, k, i]
						Q[run, k, i, t] = x0 + sigma_g * np.sqrt(omega * x2 * x3)
						# plotQ[run, k, i, t] = x0
						# plotExplore[run, k, i, t] = sigma_g * np.sqrt(omega * x2 * x3)

					rew[run, k] = np.zeros(N)
					xsi[run, k] = np.zeros(N)

					action = np.argmax(Q[run, k, :, t])
					rew[run, k, action] = np.random.normal(trueMeans[run, action], var)
					reg[run, k, t] = bestArm - trueMeans[run, action]
					xsi[run, k, action] += 1

					# for i in range(N):
					# print(trueMeans[run, action], s[run, k, i] / n[run, k, i])
					snerr[run, k, t] = trueMeans[run, bestArmIdx] - (s[run, k, bestArmIdx] / n[run, k, bestArmIdx])

			# update estimates using running consensus
			for i in range(N):
				n[run, :, i] = P @ (n[run, :, i] + xsi[run, :, i])
				s[run, :, i] = P @ (s[run, :, i] + rew[run, :, i])

			# print(f'n{t}', n)
			# print(f's{t}', s)
			# print(f'xsi{t}', xsi)
			# print(f'exp{t}', plotExplore[:, :, t])
			# print(f'Q{t}', Q[:, :, t])

	return reg, Q, snerr, s, n

def generateP(A, kappa):
	dmax = np.max(np.sum(A, axis=0))
	L = laplacian(A, normed=False)
	M, _ = np.shape(A)
	I = np.eye(M)

	P = I - (kappa/dmax) * L

	# print rho
	l = np.absolute(np.linalg.eigvals(P))
	l = l[1 - l > 1e-5]
	return P, np.max(l)


if __name__ == '__main__':
	main()
