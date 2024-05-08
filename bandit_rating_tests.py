import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from numba import njit, prange
from tqdm import tqdm


@njit(parallel=True)
def run_ucb(bandit, runs, timesteps, c=2):
	regret = np.zeros((runs, timesteps))
	best_arm_selected_percent = np.zeros((runs, timesteps))
	for run in prange(runs):
		Q = np.zeros(bandit.shape)
		N = np.zeros(bandit.shape)
		optimal_action_picked = 0
		for t in range(timesteps):
			action = np.argmax(Q + c * np.sqrt(np.log(t + 1) / (N + 1e-5)))
			reward = np.random.normal(bandit[action], 1)
			N[action] += 1
			Q[action] += (reward - Q[action]) / N[action]
			regret[run, t] = np.max(bandit) - bandit[action]
			if regret[run, t] == 0:
				optimal_action_picked += 1
				best_arm_selected_percent[run, t] = optimal_action_picked / (t + 1)

	return regret, best_arm_selected_percent

def tvd(mu_1, sigma_1, mu_2, sigma_2):
	'''
	calculate total variation distance given the means and standard deviations
	of two normal distributions
	'''
	x = np.linspace(min(mu_1 - 3 * sigma_1, mu_2 - 3 * sigma_2), max(mu_1 + 3 * sigma_1, mu_2 + 3 * sigma_2), 1000)
	pdf_1 = stats.norm.pdf(x, mu_1, sigma_1)
	pdf_2 = stats.norm.pdf(x, mu_2, sigma_2)
	return 0.5 * np.trapz(np.abs(pdf_1 - pdf_2), x)

def bandit_difficulties(bandits):
	difficulties = np.zeros(len(bandits))
	NUM_BANDITS, NUM_ARMS = bandits.shape
	for idx, bandit in enumerate(bandits):
		best_arm = np.max(bandit)
		for arm in bandit:
			difficulties[idx] += tvd(arm, 1, best_arm, 1)
		difficulties[idx] /= (NUM_ARMS)
	return difficulties

def test_bandit_ratings():
	bandits = np.array([
		[1, 2, 5],
		[1, 4, 5],
	])

	print(bandits)
	print(bandit_difficulties(bandits))

	runs = 1_000_000
	timesteps = 1000

	for bandit in bandits:
		# regret, percent_optimal = run_ucb(bandit, runs, timesteps)
		# regret = np.mean(np.cumsum(regret, axis=1), axis=0)
		# percent_optimal = np.mean(percent_optimal, axis=0)
		# np.save(f'data/data/ucb_{bandit}_regret.npy', regret)
		# np.save(f'data/data/ucb_{bandit}_percent.npy', percent_optimal)
		regret = np.load(f'data/data/ucb_{bandit}_regret.npy')
		percent_optimal = np.load(f'data/data/ucb_{bandit}_percent.npy')
		plt.plot(percent_optimal, label=bandit)
	plt.grid()
	plt.xlabel('Timesteps')
	plt.ylabel('Percent Optimal Action')
	plt.title('% Optimal Action by UCB Agent')
	plt.legend()
	plt.savefig(f'data/img/png/bandit_rating_ucb.png')
	plt.savefig(f'data/img/pdf/bandit_rating_ucb.pdf', format='pdf')
	plt.show()

def generate_bandits():
	NUM_BANDITS = 1_000_000
	NUM_ARMS = 3
	dists = [
		(0, 0.01),
		(0, 0.1),
		(0, 1),
		(0, 10),
	]

	for mean, std in tqdm(dists):
		bandits = np.random.normal(mean, std, (NUM_BANDITS, NUM_ARMS))
		np.save(f'data/data/bandits_{mean}_{std}.npy', bandits)
		print(f'Bandits with N({mean}, {std}) saved')
		ratings = bandit_difficulties(bandits)
		print(f'Bandits with N({mean}, {std}) difficulties calculated')
		plt.hist(ratings, label=f'N({mean}, {std})', bins=30, alpha=0.5)

	plt.xlabel('Bandit Rating')
	plt.ylabel('Frequency')
	plt.title('Histogram of Bandit Ratings')
	plt.legend()
	plt.savefig(f'data/img/png/bandit_ratings_histogram.pdf', format='pdf')
	# plt.show()

if __name__ == '__main__':
	np.set_printoptions(suppress=True)

	test_bandit_ratings()
	# generate_bandits()