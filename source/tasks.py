""" This file contains the Tasks class. """

from .base import * # imports BaseSeedClass, numpy, matplotlib, joblib (delayed and Parallel)
import qutip
from sklearn import linear_model
from .utils import uniform_putiry_state_generator, random_werner_state, reconstruct_rho, random_x_state, random_qubit_generator, reconstruct_2qubit, ensure_physical, concurrence_not_max, rand_bell_mixture
from qutip import sigmaz, sigmay, sigmax, partial_transpose, fidelity, entropy_vn, concurrence, qeye, tensor
from .hamiltonian import Hamiltonian

class Tasks(BaseSeededClass):

	""" Class to define the tasks for the input signals. """

	def __init__(self, n_max_delay, task_name, n_steps=4000, n_wo=1000, n_train=2000, n_test=1000, bias=True, max_bound_input=None, pm='Capacity', qtasks=[], inp_type='qubit', **kwargs):

		"""
		Initialize the Tasks class.
		Parameters:
		n_max_delay (int): delay memory of the given Task.
		task_name (str): name of the task (NARMAn, STM)
		n_steps (int): number of time steps for the time evolution
		n_wo (int): number of time steps for the warm-up period
		n_train (int): number of time steps for the training period
		n_test (int): number of time steps for the testing period
		bias (bool): If adding bias term or not in training
		max_bound_input (float): Maximum value of input values generator (Only for NARMA task)
		pm (str): Performance metric. Possible values: 'Capacity' (Pearson correlation²) or 'NMSE (Normalised Mean Squared Error)'
		qtasks (list): List with quantum tasks to analyze if input is quantum.
		inp_type (str): Type of input. Possible values: 'qubit', '2qubit'.
	
		Raises:
		x_out is not initialized.
		"""
		super().__init__(**kwargs) # initialize the parent class

		self.n_max_delay = n_max_delay # maximum delay for the input signal

		if n_wo + n_train + n_test != n_steps:
			raise ValueError("n_wo + n_train + n_test must be equal to n_steps.")

		if task_name not in [f"NARMA", "STM", "PC", "Qinp"]:
			raise ValueError("task_name must be 'NARMA', 'STM', 'PC', 'Qinp''.")
		
		if pm not in ["Capacity", "NMSE", "Fidelity", "FidelityPurity"]:
			raise ValueError("Performance metric (pm) mus be 'Capacity', 'NMSE' or 'Fidelity.")

		self.n_steps = n_steps # number of time steps for the time evolution
		self.n_wo = n_wo # number of time steps for the warm-up period
		self.n_train = n_train # number of time steps for the training period
		self.n_test = n_test # number of time steps for the testing period
		
		self.task_name = task_name # name of the task
		self.bias = bias # add bias term to the input signal
		self.max_bound_input = max_bound_input

		if not qtasks:
			self.qtasks = ["Tracerho2", "Tracerho3", "Tracerho4", "Sigman", "Entanglement", "Fidelity"]
		else:
			self.qtasks = qtasks
		self.nqtasks = len(self.qtasks)
		self.strqtasks = self.get_str_qtasks()

		self.pm = pm
		self.inp_type = inp_type

		self.x_out = None # output signal

	def __repr__(self):

		""" Return a representation of the Tasks class. """

		return f"Tasks(n_max_delay={self.n_max_delay}, n_steps={self.n_steps}, task_name={self.task_name})"
	
	def __str__(self):
			
		""" Return a string description of the Tasks class. """
	
		return f"Tasks class with maximum delay {self.n_max_delay}, number of time steps {self.n_steps}, and task name {self.task_name}."
	
	def get_str_qtasks(self):

		"""
		Generates the strings of quantum tasks
		"""
		self.strqtasks = ''

		for qt in self.qtasks:
			self.strqtasks += qt

		return self.strqtasks

	def get_input_signal(self):

		"""
		Generates the input signal for the task.
		The input signal depends on the Task name:
		- NARMAn: Nonlinear auto-regressive moving average n task
		- STM: Short Term Memory task
		- PC: Parity Check task
		- Qinp: Quantum input
		"""

		if self.task_name == "NARMA":

			if self.max_bound_input == None:
				if self.n_max_delay <= 10:
					self.max_bound_input = 0.2
				if self.n_max_delay > 10 and self.n_max_delay <= 20:
					self.max_bound_input = 0.02
				if self.n_max_delay > 20:
					self.max_bound_input = 0.001

			self.input_signals = self.rng.uniform(0, self.max_bound_input, size=(self.n_steps))
			
		if self.task_name == "STM":
			self.max_bound_input = 1
			self.input_signals = self.rng.uniform(0, self.max_bound_input, size=(self.n_steps)) # input signal

		if self.task_name == "PC":
			self.max_bound_input = 1
			self.input_signals = self.rng.integers(0, self.max_bound_input, size=(self.n_steps), endpoint=True)

		if self.task_name == "Qinp":
			
			seeds = self.rng.integers(0, 1e9, size=self.n_steps)

			if self.inp_type == '2qubit':
				self.input_signals = np.array([qutip.rand_dm(2*[2], distribution='ginibre', seed=s) for s in seeds]) # input signal of density matrices
			elif self.inp_type == 'qubit':
				if 'Entanglement' in self.qtasks:
					raise ValueError('For Entanglement task it is required a minimum of 2 qubit state. Change inp_type')
				self.input_signals = np.array([qutip.rand_dm([2], distribution='ginibre', seed=s) for s in seeds]) # input signal of density matrices
			elif self.inp_type == 'werner':
				p_val = self.rng.uniform(0,1, size=(self.n_steps))
				self.input_signals = np.array([qutip.Qobj(random_werner_state(p)) for p in p_val ])
			elif self.inp_type == 'x_state':
				self.input_signals = np.array([random_x_state(s) for s in seeds])
			elif self.inp_type == 'rand_bell_mix':
				p_val = self.rng.uniform(0,1, size=(self.n_steps))
				self.input_signals = np.array([rand_bell_mixture(p, s) for p, s in zip(p_val, seeds)])
				print(self.input_signals.shape)

		return self.input_signals
	
	def get_quantum_input_features(self, qtask, sigmas = np.array([sigmax(), sigmay(), sigmaz()])):

		"""
		Generates the quantity of interest for a given quantum task
		"""

		if not hasattr(self, 'input_signals') or self.input_signals is None:
			print('There is no input signal, generating ...')
			self.get_input_signal() # generate the input signal

		if qtask == 'Tracerho2':
			self.tr_dm2 = np.array([i.purity() for i in self.input_signals])
		elif qtask == 'Tracerho3':
			dm3 = np.array([i * i * i for i in self.input_signals])
			self.tr_dm3 = np.array([m.tr() for m in dm3])
		elif qtask == 'Tracerho4':
			dm4 = np.array([i * i * i * i for i in self.input_signals])
			self.tr_dm4 = np.array([m.tr() for m in dm4])
		elif qtask == 'Sigman':
			n = np.array([1,1,1]); n_norm = n / np.linalg.norm(n)
			obs_sn = np.dot(n_norm, sigmas)
			self.tr_ob_sn = qutip.expect(obs_sn, list(self.input_signals))
		elif qtask == 'Entanglement':
			# eigv = [i.eigenenergies(sort='high') for i in self.input_signals]
			# self.conc = np.array([eig[0] - np.sum(eig[1:]) for eig in eigv])
			self.conc = np.array([concurrence_not_max(i) for i in self.input_signals])
		elif qtask == 'Fidelity':
			self.get_inputstate_tomography(inp_type=self.inp_type)
		elif qtask == 'Entropy':
			self.entropy_vn = np.array([qutip.entropy_vn(i, base=2) for i in self.input_signals])

		return
	
	def get_inputstate_tomography(self, inp_type='qubit'):

		"""
		Computes the all axis projections of the input state
		"""

		if inp_type == 'qubit':

			sz = sigmaz()
			sx = sigmax()
			sy = sigmay()

			self.sigma_zxy_qinput = np.full((self.n_steps, 3), fill_value=np.nan)
			self.sigma_zxy_qinput[:, 0] = np.array(qutip.expect(sz, list(self.input_signals))) # Sigma z
			self.sigma_zxy_qinput[:, 1] = np.array(qutip.expect(sx, list(self.input_signals))) # Sigma x
			self.sigma_zxy_qinput[:, 2] = np.array(qutip.expect(sy, list(self.input_signals))) # Sigma y

		elif inp_type in ['werner', 'x_state', '2qubit', '2qubit1', 'rand_bell_mix']:

			# Define the single-qubit Pauli basis
			paulis = [qeye(2), sigmax(), sigmay(), sigmaz()]
			labels = ['I', 'X', 'Y', 'Z']

			all_ops = []
			all_labels = []

			for i, A in enumerate(paulis):
				for j, B in enumerate(paulis):
					if not (i == 0 and j == 0):  # skip I\otimesI term
						all_ops.append(tensor(A, B))
						all_labels.append(labels[i] + labels[j])

			for i, rho in enumerate(self.input_signals):
				if not isinstance(rho, qutip.Qobj):
					print(f"Non-Qobj at index {i}: {type(rho)}")

			for i, op in enumerate(all_ops):
				if not isinstance(op, qutip.Qobj):
					print(f"Non-Qobj at index {i}: {type(op)}")

			self.sigma_zxy_qinput = np.full((self.n_steps, 15), np.nan)

			for i, op in enumerate(all_ops):
				self.sigma_zxy_qinput[:, i] = qutip.expect(op, list(self.input_signals))

		return self.sigma_zxy_qinput

	
	def reset_quantum_input_features(self):

		"""
		Set to None the quantum features
		"""

		self.tr_dm2 = None
		self.tr_dm3 = None
		self.tr_dm4 = None
		self.tr_ob_sn = None
		self.conc = None
		self.sigma_zxy_qinput = None
		self.entropy_vn = None

		return

	def get_output_signal(self, n_delay=None, qtasks=None, reshapeflag=False):

		"""
		Generates the output signal for the task.
		The output signal depends on the Task name.
		For Classical inputs:
		- NARMA: Nonlinear auto-regressive moving average n task
		- STM: Short Term Memory task
		- PC: Parity Check task
		
		For Quantum inputs:
		- Tr: Trace of the square of the density state
		- Det: Determinant of the density state
		- LogNeg: Logarithmic Negativity
		"""

		if not hasattr(self, 'input_signals') or self.input_signals is None:
			print('There is no input signal, generating ...')
			self.get_input_signal() # generate the input signal

		if n_delay is None:
			n_delay = self.n_max_delay # If delay is nos specified, set the total delay

		if self.task_name == "NARMA":
			alpha, beta, gamma, delta = 0.3, 0.05, 1.5, 0.1 # NARMA-10 parameters
			self.output_signals = np.zeros(self.n_steps, dtype=float) # output signal

			for k in range(n_delay, self.n_steps):  
				y_prev = self.output_signals[k-1]
				y_sum = np.sum(self.output_signals[k - n_delay : k])
				u_k_prev = self.input_signals[k-1]
				u_k_delay = self.input_signals[k - n_delay]
				self.output_signals[k] = (
					alpha * y_prev +
					beta * y_prev * y_sum +
					gamma * u_k_prev * u_k_delay +
					delta
				)
		
		if self.task_name == "STM":
			# Generate output signal with delay
			self.output_signals = np.zeros_like(self.input_signals, dtype=float)
			if n_delay == 0:
				self.output_signals = self.input_signals
			else:
				self.output_signals[n_delay:] = self.input_signals[:-n_delay]

		if self.task_name == "PC":
			self.output_signals = np.zeros_like(self.input_signals, dtype=int)

			for k in range(n_delay, self.n_steps):
				self.output_signals[k] = np.sum(self.input_signals[k-n_delay : k]) % 2

		if self.task_name == "Qinp":

			if qtasks is None:
				qtasks = self.qtasks

			is_single_qtask = len(qtasks) == 1
			output_shape = (self.n_steps,) if is_single_qtask and not reshapeflag else (len(qtasks), self.n_steps)
			self.output_signals = np.zeros(output_shape)

			for idx, qtask in enumerate(qtasks):
				if qtask == 'Tracerho2':
					if not hasattr(self, 'tr_dm2') or self.tr_dm2 is None:
						self.get_quantum_input_features(qtask)
					
					signal = self.tr_dm2[:-n_delay] if n_delay else self.tr_dm2

				if qtask == 'Tracerho3':
					if not hasattr(self, 'tr_dm3') or self.tr_dm3 is None:
						self.get_quantum_input_features(qtask)
					
					signal = self.tr_dm3[:-n_delay] if n_delay else self.tr_dm3

				if qtask == 'Tracerho4':
					if not hasattr(self, 'tr_dm4') or self.tr_dm4 is None:
						self.get_quantum_input_features(qtask)
					
					signal = self.tr_dm4[:-n_delay] if n_delay else self.tr_dm4

				elif qtask == 'Exp':
					exp = np.array([i.expm() for i in self.input_signals])
					tr_exp = np.array([e.tr() for e in exp])
					signal = tr_exp[:-n_delay] if n_delay else tr_exp

				elif qtask == 'Entropy':
					if not hasattr(self, 'entropy_vn') or self.entropy_vn is None:
						self.get_quantum_input_features(qtask)

					signal = self.entropy_vn[:-n_delay] if n_delay else self.entropy_vn

				elif qtask == 'Sigman':
					if not hasattr(self, 'tr_ob_sn') or self.tr_ob_sn is None:
						self.get_quantum_input_features(qtask)
					signal = self.tr_ob_sn[:-n_delay] if n_delay else self.tr_ob_sn

				elif qtask == "Entanglement":

					if not hasattr(self, 'conc') or self.conc is None:
						self.get_quantum_input_features(qtask)

					signal = self.conc[:-n_delay] if n_delay else self.conc

				elif qtask == "Fidelity":
					
					tom_shape = 3 if self.inp_type == 'qubit' else 15
					self.output_signals = np.zeros((self.n_steps, tom_shape)) if is_single_qtask and not reshapeflag else np.zeros((len(qtasks),self.n_steps, tom_shape))

					if not hasattr(self, 'sigma_zxy_qinput') or self.sigma_zxy_qinput is None:
						self.get_quantum_input_features(qtask)
					signal = self.sigma_zxy_qinput[:-n_delay] if n_delay else self.sigma_zxy_qinput

				# Store in the output
				if is_single_qtask and not reshapeflag:
					self.output_signals[n_delay:] = signal
				else:
					self.output_signals[idx, n_delay:] = signal

			self.qoutput_signals = self.output_signals  # For performance metrics

		return self.output_signals

	def split_data(self):

		"""
		Function to split the data into training and testing sets.
		"""

		if not hasattr(self, 'x_out') or self.x_out is None:
			raise ValueError("x_out is not initialized.")
		if not hasattr(self, 'output_signals') or self.output_signals is None:
			raise ValueError("target is not initialized.")

		# Wasshing out intial values
		target = self.output_signals[self.n_wo:] # remove the warm-up period
		x_out = self.x_out[self.n_wo:] # remove the warm-up period

		if self.bias:
			# Add bias term
			x_out = np.concatenate((np.ones((self.n_steps-self.n_wo, 1)), x_out), axis=1)

		# Split the data into training and testing sets
		self.x_train = x_out[:self.n_train] # training set
		self.x_test = x_out[self.n_train:self.n_train+self.n_test]
		self.y_train = target[:self.n_train]
		self.y_test = target[self.n_train:self.n_train+self.n_test]

		return self.x_train, self.x_test, self.y_train, self.y_test
	
	def linear_regression(self, ridge=False):

		"""
		Training the output weigths using linear regression.
		The output weights are computed using the ordinary least squares method:
		W = (X^T X)^-1 X^T y
		where X is the input matrix and y is the output vector.
		"""

		self.split_data() # split the data into training and testing sets
		print(ridge)
		
		if ridge:
			clf = linear_model.Ridge(fit_intercept=False, alpha=1e-5) # we do not intercept value (b=0), then y = Wx.
		else:
			clf = linear_model.LinearRegression(fit_intercept=False) # we do not intercept value (b=0), then y = Wx.

		clf.fit(self.x_train, self.y_train)
		self.y_pred = clf.predict(self.x_test); self.y_pred_debugg = clf.predict(self.x_train)

		return self.y_pred
	
	def get_pm(self, y_test, y_pred):
		
		""" This method returns the performance metric """

		if 'Entanglement' in self.qtasks and self.nqtasks == 1:
			y_pred = np.maximum(y_pred,0)
			y_test = np.maximum(y_test,0)

		if self.pm == 'Capacity':
			C = np.corrcoef(y_test, y_pred, rowvar=False)[0, 1]**2 # Square pearson corr
		elif self.pm == 'NMSE':
			C = np.mean((y_pred - y_test)**2) / np.var(y_test)       
		elif self.pm == 'Fidelity':

			tau = self.n_max_delay
			t0 = self.n_wo + self.n_train        # test starts here
			t1 = t0 + self.n_test                # test ends here (exclusive)

			# Reconstruct predicted states
			if self.inp_type == 'qubit':
				new_rho = np.array([reconstruct_rho(sz, sx, sy) for sz, sx, sy in y_pred])
			else:
				new_rho = np.array([reconstruct_2qubit(a) for a in y_pred])

			# Step 1: Mark nonphysical states as None
			new_rho_checked = [rho if ensure_physical(rho) else None for rho in new_rho]

			# Step 2: Get the delayed targets
			self.delayed_input_state = self.input_signals[t0 - tau : t1 - tau]

			# Step 3: Sanity check
			assert len(new_rho_checked) == len(self.delayed_input_state), "Mismatch in lengths"

			# Step 4: Compute fidelity^2 only for valid states
			F_list = []
			for rho, idm in zip(new_rho_checked, self.delayed_input_state):
				if rho is not None:
					F_list.append(fidelity(idm, rho)**2)

			# Step 5: Average
			C = np.mean(F_list)
			
			# j=0
			# for i,f in enumerate(F_list):
			#     if f>1:
			#         print(i, f, new_rho[i].eigenenergies())
			#         j+=1

			# print(np.max(C))
			# print(len(F_list))

		elif self.pm == 'FidelityPurity':

			tau = self.n_max_delay
			t0 = self.n_wo + self.n_train        # test starts here
			t1 = t0 + self.n_test                # test ends here (exclusive)

			# Reconstruct predicted states
			if self.inp_type == 'qubit':
				new_rho = np.array([reconstruct_rho(sz, sx, sy) for sz, sx, sy in y_pred])
			else:
				new_rho = np.array([reconstruct_2qubit(a) for a in y_pred])

			# Step 1: Mark nonphysical states as None
			new_rho_checked = [rho if ensure_physical(rho) else None for rho in new_rho]

			# Step 2: Get the delayed targets
			self.delayed_input_state = self.input_signals[t0 - tau : t1 - tau]

			# Step 3: Sanity check — are we really comparing shifted values?
			assert len(new_rho) == len(self.delayed_input_state), "Mismatch in predicted and reference lengths"

			# Step 4: Compute fidelity^2 only for valid states
			delay_purity = []
			reconstruct_purity = []
			for rho, idm in zip(new_rho_checked, self.delayed_input_state):
				if rho is not None:
					reconstruct_purity.append(rho.purity())
					delay_purity.append(idm.purity())

			delay_purity = np.array(delay_purity)
			reconstruct_purity = np.array(reconstruct_purity)

			C = np.mean((reconstruct_purity - delay_purity)**2) / np.var(delay_purity)

		return C

	def performance(self, qflag=False, ridge=False):

		"""
		This function computes the performance metric for the model.
		The performance metric is defiend as the square of the Pearson correlation coefficient between the predicted and target values:
		C = (cov(X, Y) / (std(X) * std(Y)))^2
		where X is the predicted values and Y is the target values.
		"""

		if self.task_name == "Qinp" and qflag:

			self.C = np.full((self.nqtasks), fill_value=np.nan)
			for i in range(self.nqtasks):
				self.output_signals = self.qoutput_signals[i,:]
				self.linear_regression(ridge=ridge)
				self.C[i] = self.get_pm(self.y_test, self.y_pred)

		else:
			self.linear_regression(ridge=ridge)
			self.C = self.get_pm(self.y_test, self.y_pred)

		# if self.n_max_delay == 1:
		# 	plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})
		# 	plt.rcParams['axes.titlesize'] = 20
		# 	plt.rcParams['lines.linewidth'] = 3
		# 	plt.rcParams['axes.linewidth'] = 2
		# 	fig, ax = plt.subplots(figsize=(12,5), ncols=1, nrows=1)
		# 	ax.plot(range(3180,3250),np.maximum(0,self.y_test[180:250]), marker='D', color='k', markeredgecolor='k', markeredgewidth=1.5,
		# 			label='Target')
		# 	ax.plot(range(3180,3250),np.maximum(0,self.y_pred[180:250]), linewidth=2, marker='o', color='darkorange', markeredgecolor='k',
		# 			markeredgewidth=1.5, label='Prediction')
		# 	ax.set_xlabel('k')
		# 	ax.set_ylabel(r'C($\rho_{1,2}$)')
		# 	ax.legend(loc='best', frameon=False)
		# 	plt.show()
		print(f'Check iteration delay {self.n_max_delay}----------------------------')
		print('test_window: ', self.C)
		print('task.window: ', self.get_pm(self.y_train, self.y_pred_debugg))
		
		return self.C