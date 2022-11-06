import numpy as np

M = 5
N = 2

n = np.zeros((M, N))	# number of times an arm has been selected by each agent
s = np.zeros((M, N))	# cumulative expected reward
xsi = np.zeros((M, N))	# number of times that arm has been selected in that timestep
rew = np.zeros((M, N))	# reward

P = np.array([
	[0.980, 0.005, 0.005, 0.005, 0.005],
	[0.005, 0.995, 0.000, 0.000, 0.000],
	[0.005, 0.000, 0.995, 0.000, 0.000],
	[0.005, 0.000, 0.000, 0.995, 0.000],
	[0.005, 0.000, 0.000, 0.000, 0.995],
])


print(n)
print(s)

for i in range(N):
	n[:, i] = P @ (n[:, i] + xsi[:, i])
	s[:, i] = P @ (s[:, i] + rew[:, i])

print(n)
print(s)
