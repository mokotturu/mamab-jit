from time import ctime, time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from scipy.sparse.csgraph import laplacian

from graph_optimization import fdla_weights_symmetric, fmmc_weights


def main():
	N = 2
	runs = 10000
	T = 100

	# adjacency matrices
	As = [
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
		# np.array([
		# 	[0, 1, 1, 1, 1],
		# 	[1, 0, 0, 0, 0],
		# 	[1, 0, 0, 0, 0],
		# 	[1, 0, 0, 0, 0],
		# 	[1, 0, 0, 0, 0],
		# ])
		np.array([
			[0, 1, 1, 1, 1],
			[1, 0, 1, 1, 1],
			[1, 1, 0, 1, 1],
			[1, 1, 1, 0, 1],
			[1, 1, 1, 1, 0],
		])
	]

	# corresponding incidence matrices
	Is = [
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
		# np.array([
		# 	[ 1,  1,  1,  1],
		# 	[-1,  0,  0,  0],
		# 	[ 0, -1,  0,  0],
		# 	[ 0,  0, -1,  0],
		# 	[ 0,  0,  0, -1],
		# ])
		np.array([
			[ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0],
			[-1,  0,  0,  0,  1,  1,  1,  0,  0,  0],
			[ 0, -1,  0,  0, -1,  0,  0,  1,  1,  0],
			[ 0,  0, -1,  0,  0, -1,  0, -1,  0,  1],
			[ 0,  0,  0, -1,  0,  0, -1,  0, -1, -1],
		])
	]

	M, _ = As[0].shape

	kappas = [0.02]
	networks = [
		'8-agent network',
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
	plt.title('All-to-All network')
	# for ax in fig.get_axes():
	# 	ax.label_outer()
	# axs0 = axss
	# axs1 = axss[1].flatten()

	print(f'Simulation started at {ctime(time())}')

	labels = [
		'κ = 0.02',
		'FDLA',
		'FMMC',
	]

	markers = ['o', '^', 's']

	P0 = generateP(As[0], kappa=0.02)
	_, P1, _ = fdla_weights_symmetric(Is[0])
	_, P2, _ = fmmc_weights(Is[0])

	# P0, P1, P2 = [], [], []

	for idx, P in enumerate([P0, P1, P2]):
		_, Q, snerr, s, n = run(runs, N, T, trueMeans, P)

		
		# Q = np.mean(Q, axis=0)
		snerr = np.mean(snerr, axis=0)

		np.save(f'snerr_{idx}.npy', snerr)
		# snerr = np.load(f'snerr_{idx}.npy')

		# s = np.mean(s, axis=0)
		# n = np.mean(n, axis=0)

		# print(snerr)

		# print(f's: {s}')
		# print(f'n: {n}')

		# for i, axUp in enumerate(axs0):
		# ax.set_prop_cycle(color=[cm(1.*j/10) for j in range(10)])
		plt.grid(True)
		# plt.title(f'Agent {i}')
		plt.xlabel('Timesteps')
		plt.ylabel('Mean estimate error for the best arm')
		# if i == 0:
			# axUp.set_ylabel('s / n error')

		# axDown.grid(True)
		# axDown.set_title(f'Agent {i}')
		# axDown.set_xlabel('Timesteps')
		# axDown.set_ylabel('s / n error')
		# if i == 0:
		# 	axDown.set_ylabel('s / n error')

		# for n in range(N):
			# axUp.plot(sn[i, n, :], label=f'Arm {n}', lw=1, linestyle='solid', color=colors[n])
			# axUp.plot([trueMeans[0][n] for _ in range(1000)], label=f'Arm {n} true mean', lw=1, linestyle='dashed', color=colors[n])
		# for i in range(M):
		# if i == 0:
		# plt.plot(np.mean(snerr, axis=0), lw=1, linestyle='solid', color=colors[idx])
		# else:
		plt.plot(np.mean(snerr, axis=0), marker=markers[idx], markevery=5, lw=2, linestyle='solid', color=colors[idx], label=labels[idx])
		snerr = np.mean(snerr, axis=0)
		print(snerr)
		argmaxidx = np.argmax(snerr)
		sanitized_arr = snerr[argmaxidx:]
		drop_amt = snerr[argmaxidx] / np.exp(1)
		print(snerr[argmaxidx], drop_amt)
		drop_amt_idx = np.argwhere(sanitized_arr < drop_amt)[0][0]
		print("SETTLING TIME: ", drop_amt_idx, snerr[drop_amt_idx])
		plt.axvline(argmaxidx + drop_amt_idx, color=colors[idx], linestyle='dashed', lw=2)

			# axDown.plot([trueMeans[0][n] for _ in range(1000)], label=f'Arm {n} true mean', lw=1, linestyle='dashed', color=colors[n])
		plt.legend()
		# ax.legend()

	# handles, labels = axs0[0].get_legend_handles_labels()
	# fig.legend(handles, labels, loc='right')

	# P = fdla_weights_symmetric(Is[0])[1]
	# _, Q = run(runs, N, T, trueMeans, P)
	# print(Q.shape)


	# log all agent wise data
	# res = np.genfromtxt(f'{network}-{label}.txt')

	# for i, agentReg in enumerate(res):
	# ax.plot(np.cumsum(np.mean(res, axis=0)), label=f'{label}', lw=2)
	# for i, arr in enumerate(errors):
	# 	ax.plot(arr, label=f'{label} arm {i}', lw=1, linestyle='dashed' if label == 'FDLA' else 'solid')


	print(f'Simulation ended at {ctime(time())}')
	plt.savefig('ata-param.svg', format='svg')
	plt.savefig('ata-param.png', format='png')
	plt.tight_layout()
	plt.show()

@njit(parallel=True)
# @njit
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
	print(f'kappa: {kappa}, rho: {np.max(l)}')

	print(P)

	return P


if __name__ == '__main__':
	main()
