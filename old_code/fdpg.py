import numpy as np

def signal_smoothness(X: np.ndarray, W: np.ndarray):
	'''
	E = ||x_i - x_j||^2
	'''
	E = nodal_distance(X, W)
	tv = 0.5 * np.sum(np.abs(W * E))
	return tv

def nodal_distance(X: np.ndarray, W: np.ndarray) -> np.ndarray:
	'''
	Compute the distance between each node in X
	'''
	E = np.zeros_like(W)
	for i in range(X.shape[0]):
		for j in range(X.shape[0]):
			E[i, j] = np.linalg.norm(X[i] - X[j]) ** 2
	return E

def calculate_S(n):
	iu = np.triu_indices(n, k=1)
	S = np.zeros((n, len(iu[0])))
	S[iu[0], range(len(iu[0]))] = 1
	S[iu[1], range(len(iu[0]))] = 1
	return S

def fdpg(e, K, alpha, beta, N, epsilon) -> np.ndarray:
	'''
	Fast Dual-based Proximal Gradient Method (FDPG)

	e: vec(E), where E is the TVD matrix of the signals
	K: number of iterations
	alpha: tuning parameter (penalizes the possibility of isolated nodes)
	beta: tuning parameter (encourages graph sparsity)
	N: number of nodes
	epsilon: convergence threshold
	'''
	L = (N - 1) / beta # constant
	_rand = np.random.uniform(0, 1, (N, 1))
	omega_k = _rand.copy()
	lambda_k_m_1 = _rand.copy()
	lambda_k = _rand.copy()
	t_k = 1

	iu = np.triu_indices(N, k=1)
	S = np.zeros((N, len(iu[0])))
	S[iu[0], range(len(iu[0]))] = 1
	S[iu[1], range(len(iu[0]))] = 1

	for k in range(K):
		w_k = np.clip((S.T @ omega_k - 2 * e) / (2 * beta), 0, None)
		u_k = 0.5 * (S @ w_k - L * omega_k + np.sqrt((S @ w_k - L * omega_k) ** 2 + 4 * alpha * L * np.ones(N)))
		lambda_k = omega_k - (1 / L) * ((S @ w_k) - u_k)
		tau_k_p_1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k ** 2))
		omega_k = lambda_k + ((t_k - 1) / tau_k_p_1) * (lambda_k - lambda_k_m_1)
		if np.linalg.norm(lambda_k - lambda_k_m_1) / np.linalg.norm(lambda_k_m_1) < epsilon:
			print(f'converged at iteration {k}')
			break
	return w_k
