""" This file contains utils functions for the different classes """

import numpy as np
from qutip import sigmax, sigmay, sigmaz, qeye, bell_state, ket2dm, tensor, Qobj, basis, expect, gates, rand_dm

def int_or_float(value):
    try:
        return int(value)
    except ValueError:
        return float(value)

def load_observables_data(L, Js, W, h, dt, Vmp, Dmp, N_rep, task_name, it, 
						  inp_type='qubit', back_action=False, monitor_axis='x',
						  meas_strength=None):

	""" 
	This function load the obsevable dynamics for a given configuration
	"""

	path = f"results/data/"
	if back_action:
		path += f"back_action/{monitor_axis}/"
		pathend = f"_MeasStr_{meas_strength}"
	
	path += f"{task_name}/QRC/{inp_type}/L{L}_Js{Js}_h{h}_W{W}_dt{dt}_V{Vmp}_D{Dmp}_Nrep{N_rep}"
	if back_action:
		path += pathend
	
	path += f"/Iter_{it}.npz"
	obs = np.load(path, allow_pickle=True)

	return obs

def get_obs_idx(L, axis, caxis, ccaxis, Vmp, Dmp):

	"""
	Returns the indices of observables for linear regression.
	"""
	Lc = np.sum(np.arange(L))
	Lc2 = 2*Lc
	idx = []
	q1_idx = [] # indeces of qubit 1

	base = 3 * (L + Lc + Lc2) * Vmp

	for D in range(Dmp):
		offset_D = D * base

		# Local observables
		for ax_index, ax in enumerate(['z', 'x', 'y']):
			if ax in axis:
				for i in range(L):
					for V in range(Vmp):
						val = i + L*V + ax_index * L * Vmp + offset_D
						idx.append(val)
						if i == 0:
							q1_idx.append(val)

		# Correlation observables
		for cax_index, cax in enumerate(['z', 'x', 'y']):
			if cax in caxis:
				for i in range(Lc):
					for V in range(Vmp):
						val = i + Lc*V + (3*L + cax_index*Lc)*Vmp + offset_D
						idx.append(val)
						if i == 0:
							q1_idx.append(val)
		
		# Cross axis correlations observables
		for ccax_index, ccax in enumerate(['zx', 'xy', 'zy']):
			if ccax in ccaxis:
				for i in range(Lc2):
					for V in range(Vmp):
						val = i + Lc2*V + (3*(L+Lc) + ccax_index*Lc2)*Vmp + offset_D
						idx.append(val)
						if i == 0:
							q1_idx.append(val)
						
	return idx, q1_idx

def random_qubit_generator(seed):

	# Computational basis
	ket0 = basis(2,0)
	ket1 = basis(2,1)

	rng = np.random.default_rng(seed)

	# To generate random eigenvectors we generate a radnom unitary matrix:
	alpha = rng.uniform(0,2*np.pi)
	psi = rng.uniform(0,2*np.pi)
	xi = rng.uniform(0,2*np.pi)
	zeta = rng.uniform(0,1)
	phi = np.arcsin(np.sqrt(zeta))

	u11 = np.cos(phi) * np.exp(complex(0,psi))
	u22 = np.conj(u11)
	u12 = np.sin(phi) * np.exp(complex(0,xi))
	u21 = -np.conj(u12)

	U = np.exp(complex(0,alpha))* Qobj([[u11,u12],[u21,u22]])

	# Generate random eigen vectors
	v0 = U * ket0
	v1 = U * ket1

	# Generate random eigenvalues (probabilities)
	p0 = rng.uniform(0,1)
	p1 = 1 - p0

	rho = p0 * ket2dm(v0) + p1 *ket2dm(v1)
	return rho

def uniform_putiry_state_generator(seed):

	rng = np.random.default_rng(seed)
	purity = rng.uniform(0.5, 1)
	r = np.sqrt(2*purity-1)

	phi = rng.uniform(0, 2*np.pi)
	theta = rng.uniform(0, np.pi)

	sintheta = np.sin(theta)

	x = r * sintheta * np.cos(phi)
	y = r * sintheta * np.sin(phi)
	z = r * np.cos(theta)

	rho = 0.5 * (qeye(2) + x * sigmax() + y * sigmay() + z * sigmaz())
	return rho

def random_werner_state(p):

	rho = p * ket2dm(bell_state('11')) + (1-p)/4 * qeye([[2,2]])
	return rho

def random_x_state(seed):

	# Real diagonal elements, sampled randomly and normalized (trace 1)
	rng = np.random.default_rng(seed=seed)

	diag = rng.random(4)
	diag /= np.sum(diag)
	a, b, c, d = diag 

	assert (np.sum(diag) - 1 < 1e-10)

	# Random complex off-diagonal terms
	max_w = np.sqrt(a * d)  # constraint from positivity
	max_z = np.sqrt(b * c)

	min_w = 0 ; min_z = 0

	abs_w = rng.uniform(min_w, max_w)
	abs_z = rng.uniform(min_z, max_z)
	phase_w = rng.uniform(0, 2 * np.pi)
	phase_z = rng.uniform(0, 2 * np.pi)

	w = abs_w * np.exp(1j * phase_w)
	z = abs_z * np.exp(1j * phase_z)

	# Construct the X-state matrix
	rho = np.array([
		[a, 0,   0,   w],
		[0, b,   z,   0],
		[0, z.conjugate(), c, 0],
		[w.conjugate(), 0, 0, d]
	], dtype=complex)

	rho = Qobj(rho, dims=[[2,2],[2,2]])

	return rho

def load_readout_layer(L, Js, W, h, dt, Vmp, Dmp, N_rep, N_esn, g, l, task_name, it):

	""" Load the qrc observable and esn states dynamics computed with the satticmethod parallel_hybrid_worker"""

	path = f'results/data/{task_name}/HYB/L{L}_Js{Js}_h{h}_W{W}_dt{dt}_Vmp{Vmp}_Dmp{Dmp}_Nrep{N_rep}_Nesn{N_esn}_g{g}_l{l}/Iter_{it}.npz'
	state = np.load(path)

	return state

def ax_to_str(axis,caxis):

	""" Return the lsit of axis and correlation axis as a string sequence """
	ax_str = ''
	cax_str = ''
	
	for i in axis:
		ax_str += i
	for j in caxis:
		cax_str += j

	return ax_str, cax_str

def reconstruct_rho(sz,sx,sy):

	""" Computes the density from the spin expected values """

	rho = 0.5 * (qeye(2) + sz * sigmaz() + sx * sigmax() + sy *sigmay())
	
	return rho

def reconstruct_2qubit(row):
	"""
	Reconstruct 2-qubit density matrix from one row of sigma_zxy_qinput 
	(output of input_full_esn with axis=['x','y','z']).

	Args:
		row (array): 1D array with 15 expectation values 
					 [xI, Ix, yI, Iy, zI, Iz, xx, xy, xz, yx, yy, yz, zx, zy, zz]

	Returns:
		Qobj: 4x4 density matrix
	"""
	# Define single-qubit Pauli operators
	paulis = [qeye(2), sigmax(), sigmay(), sigmaz()]
	ops = []

	for i, A in enumerate(paulis):
		for j, B in enumerate(paulis):
			if not (i == 0 and j == 0):  # skip I\otimesI term
				ops.append(tensor(A, B))

	# Start with identity contribution
	rho = tensor(qeye(2), qeye(2))

	# Add weighted Pauli terms
	for val, op in zip(row, ops):
		rho += val * op

	rho = rho / 4
	return rho

def input_full_esn(inp, inp_type, axis=None):

	if axis is None:
		axis = []

	axis_op = {
		'z': sigmaz(),
		'x': sigmax(),
		'y': sigmay()
	}

	n = len(inp)

	if inp_type == 'qubit':
		a = np.zeros((n, len(axis)))
		for k, ax in enumerate(axis):
			a[:, k] = np.array([expect(axis_op[ax], i) for i in inp])

	elif inp_type in ['werner', 'x_state', '2qubit', 'rand_bell_mix']:

		# validate axes (silently ignore unknown)
		axis = [ax for ax in axis if ax in axis_op]
		if len(axis) == 0:
			raise ValueError("No valid axes provided (choose from 'x','y','z').")

		ops = []
		labels = []

		# local operators: sigma_a ⊗ I  and I ⊗ sigma_a
		for ax in axis:
			ops.append(tensor(axis_op[ax], qeye(2)))
			labels.append(f"{ax}I")
			ops.append(tensor(qeye(2), axis_op[ax]))
			labels.append(f"I{ax}")

		# correlators: sigma_a ⊗ sigma_b for all pairs (a,b) in axes x axes
		for ax1 in axis:
			for ax2 in axis:
				ops.append(tensor(axis_op[ax1], axis_op[ax2]))
				labels.append(f"{ax1}{ax2}")

		n_ops = len(ops)

		a = np.full((n, n_ops), np.nan)

		# Use qutip.expect with list of states for each operator (fast)
		for i, op in enumerate(ops):
			a[:, i] = np.real(expect(op, list(inp)))

	else:
		raise ValueError(f"Unknown inp_type: {inp_type}")

	return a

def load_state_data(N_esn, g, l, task_name, it, ax_str='', inp_type='qubit'):

	""" Load the state dynamics data computed with the staticmethod esn_worker """
	if task_name == 'Qinp':
		path = f'results/data/{task_name}/ESN/ESN_Nesn{N_esn}_g{g}_l{l}_ax_{ax_str}_{inp_type}/Iter_{it}.npz'
	else:
		path = f'results/data/{task_name}/ESN/ESN_Nesn{N_esn}_g{g}_l{l}/Iter_{it}.npz'

	state = np.load(path)

	return state

def ensure_physical(rho, tol=1e-12):
	"""
	Returns rho if physical (PSD and trace 1), otherwise None.
	"""
	evals = rho.eigenenergies()  # cheaper than eigenstates
	if (np.min(evals) >= -tol) and (abs(rho.tr() - 1.0) < tol):
		return rho
	else:
		return None

def get_M_Had_HS_operators(monitor_axis, meas_strength, L):

	"""
	Returns the back actions operators (M, Had, HS)
	"""

	sup = np.exp(-meas_strength**2/2)
	M_qubit = np.array([[1, sup], [sup, 1]])
	QM_qubit = Qobj(M_qubit)
	QM = tensor(QM_qubit for _ in range(L))
	M = QM.full()

	if monitor_axis in ['x', 'y']:
		Had_qubit = gates.snot()
		Had = tensor(Had_qubit for _ in range(L))
	else:
		Had = None

	if monitor_axis == 'y':
		S_qubit = gates.phasegate(np.pi / 2)
		S = tensor(S_qubit for _ in range(L))
		Ry = Had * S.dag()
	else:
		Ry = None

	return M, Had, Ry


def monitor_rho_transform(rho, monitor_axis, M, Had=None, Ry=None):

	"""
	Returns densitry matrix under the effect of continious monitoring (back-action) under weak-measurements.
	The transformation depends on the axis in which the state is monitored. 
	Arguments:
	- rho: State after CPTP map (can be either unitary evolution or FN map) (Qobj)
	- monitor_axis: Direction in which the state is monitored ('x', 'y', 'z') (str)
	- M: Back-action matrix (array)
	- Had: Hadamard gate (Qobj)
	- Ry: Hadamard * Phase shift.dag() gate (Qobj)
	"""

	if monitor_axis not in ['z', 'x', 'y']:
		raise ValueError("Monitor axis must be one of the following strings: 'x', 'y', 'z'.")
	elif monitor_axis == 'z':
		rho_ba = Qobj(np.multiply(M,rho.full()), dims=rho.dims)
	elif monitor_axis == 'x':
		rho_rotx = Had * rho * Had
		rho_ba = Had * Qobj(np.multiply(M, rho_rotx.full()), dims=rho.dims) * Had
	elif monitor_axis == 'y':
		rho_roty = Ry * rho * Ry.dag()
		rho_ba = Ry.dag() * Qobj(np.multiply(M, rho_roty.full()), dims=rho.dims) * Ry

	return rho_ba

def statistical_noise(obs, N_meas, meas_strength, L, V, back_action, seed = 34):

	"""	Add statistical noise to the ideal (infinite ensemble and no back-action) value of the observables	
	Arguments:
	- obs: Measured observables (ndarray--dimensions [time steps, number of obs] )
	- N_meas: Number of experimental measurements.
	- meas_strength: value of strength back-action.
	- L: number of qubits
	"""

	T = obs.shape[0]
	noisy_obs = obs.copy()
	LV = L*V

	rng = np.random.default_rng(seed)

	if not meas_strength:
		raise ValueError("Error diverges to infinite, choose a nonzero g value")

	if back_action:

		g4 = meas_strength ** 4
		g2 = meas_strength ** 2

		one_obs_noise = np.sqrt((g2 + 1)/(g2 * N_meas))
		two_corr_noise = np.sqrt((g4 + 2*g2 + 1)/(g4 * N_meas))

		# print('local_noise ', one_obs_noise)
		# print('corr_noise ', two_corr_noise)
		# print(noisy_obs.shape, LV)

		noisy_obs[:, :LV] += rng.normal(0, one_obs_noise, size=(T, LV))
		noisy_obs[:, LV:] += rng.normal(0, two_corr_noise, size=(T, obs.shape[1]-LV))

	else:

		all_obs_noise = 1/np.sqrt(N_meas)
		noisy_obs += rng.normal(0, all_obs_noise, size=obs.shape)

	# print('local ', obs[2000:2003,0], noisy_obs[2000:2003,0])
	# print('corr ', obs[2000:2003,91], noisy_obs[2000:2003,91])

	return noisy_obs

def concurrence_not_max(rho):
    sy = sigmay()
    Y = tensor(sy, sy)

    # spin-flipped state
    rho_tilde = Y * rho.conj() * Y

    # R = sqrt( sqrt(rho) * rho_tilde * sqrt(rho) )
    sqrt_rho = rho.sqrtm()
    R = (sqrt_rho * rho_tilde * sqrt_rho).sqrtm()

    # eigenvalues of R in decreasing order
    evals = np.sort(np.real(R.eigenenergies()))[::-1]

    # Wootters concurrence
    C = evals[0] - evals[1] - evals[2] - evals[3]
    return C

def rand_bell_mixture(p, seed):

	bell = bell_state('11')
	Bell = bell * bell.dag()
	rho = p * rand_dm(2*[2],seed=seed) + (1-p) * Bell
	return rho