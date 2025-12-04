""" This file contains HybridDynamics class. """

from .quantum_reservoir import QuantumReservoirDynamics, load_observables_data, pathlib, Parallel, delayed, get_obs_idx, os
from .esn import Esn, EsnDynamics
from .tasks import linear_model, Tasks
from .hamiltonian import Hamiltonian
from .utils import load_readout_layer, np, ax_to_str, input_full_esn, statistical_noise, Nmeas_RegPrepParam
import matplotlib.pyplot as plt

class HybridDynamics(QuantumReservoirDynamics, EsnDynamics):


	""" Class to compute the dynamics og the hybrid architecture (QRC + ESN) """
		
	def __init__(self, L, J, h, W, dt, N_esn, g, l, task_name, seed=None, n_max_delay=10, ratio_QRC_delay=0.5, **kwargs):
				
		"""
		Initialize the HybridDynamics class.
		Additional Parameters:
		ratio_QRC_delay (float): ratio of the delay of the QRC (delay_QRC/total_delay) it must be within [0,1]
		"""

		super().__init__(L=L, J=J, h=h, W=W, dt=dt, N_esn=N_esn, g=g, l=l, n_max_delay=n_max_delay, seed=seed, task_name=task_name, **kwargs) # initialize the parent classes

		self.N_obs = len(self.axis)*self.L + len(self.caxis)*self.Lc + len(self.ccaxis)*self.Lcc # number of observables (QRC outputs) to be used in the output signal

		if ratio_QRC_delay < 0 or ratio_QRC_delay > 1:
			raise ValueError("ratio_QRC_delay must be inside [0,1]")
			
		self.n_delay_QRC = int(self.n_max_delay * ratio_QRC_delay)
		self.esn = EsnDynamics(N_esn, g, l)
		self.qrc = QuantumReservoirDynamics(L, J, h, W,)

	def __repr__(self):

		""" Return a representation of the HybridDynamics class. """

		return Hamiltonian.__repr__(self) + "\n" + Esn.__repr__(self)
		
	def __str__(self):

		""" Return a string description of the HybridDynamics class. """

		return Hamiltonian.__str__(self) + "\n" + Esn.__str__(self)
	
	@staticmethod
	def serie_hybrid_performance(
		L, Js, h, W, dt, N_esn, g, l, Vmp=1, Dmp=1, task_name='Qinp', N_iter=1, seed=None,
		load_obs = False, axis=['z'], caxis=['z'], ccaxis=[], n_min_delay = 0, n_max_delay= 10, ratio_delay_qrc=0.5,
		pm = 'NMSE', store=True, N_rep=1, qtasks=[], onlyesn=False, N_esn_solely=25,
		all_info=False, qrc_prep_perf=False, inp_type='qubit',
		back_action=False, monitor_axis='x', meas_strength=0.2,
		noise=False, N_meas=1e5, reg_prep_param=0, noise_ridge_corr=False, **kwargs):

		""" This function computes the performance of the serie hybrid configuration.
		In this case input signal is preprocessed via QRC by performing lienar regression for a time delay QRC smaller than the origianl target.
		Then, this prediction acts as an input of the ESN that, within the observable dynamics of the QRC, applies linear regssion
		for a final target delay.
		"""

		if task_name == 'PC' and n_min_delay < 1:
			n_min_delay = 1
			print('For PC task the minimum delay is 1: n_min_delay set to 1')

		if ratio_delay_qrc < 0 or ratio_delay_qrc > 1:
			raise ValueError("ratio_qrc_delay must be within 0 and 1")
		
		if back_action:
			axis = [monitor_axis]; caxis = [monitor_axis]

		delays = list(range(n_min_delay, n_max_delay))
		task_qrc = Tasks(n_max_delay=0, task_name=task_name, pm=pm, qtasks=qtasks, seed=seed, inp_type=inp_type)
		task_hyb = Tasks(n_max_delay=0, task_name=task_name, pm=pm, qtasks=qtasks, seed=seed, inp_type=inp_type)
		strqtasks = task_qrc.strqtasks
		obs_idx, _ = get_obs_idx(L, axis, caxis, ccaxis, Vmp, Dmp)
		nqtasks = task_qrc.nqtasks

		q_preproces_shape = 3 if inp_type == 'qubit' else 15

		ax_str, cax_str = ax_to_str(axis=axis, caxis=caxis)

		C_shape = (nqtasks, len(delays), N_iter)
		C = np.full(C_shape, fill_value=np.nan)
		C_preprocess = np.full((q_preproces_shape, len(delays), N_iter),fill_value=np.nan)

		if onlyesn:
			C_esn = np.full_like(C, fill_value=np.nan)

		if not load_obs:

			data = QuantumReservoirDynamics.qrc_obs(
					L=L, Js=Js, N_iter=N_iter, task_name=task_name, dt=dt, Vmp=Vmp, seed=seed,
					store=False, sweep_param=None, fixed_h=h, fixed_W=W, 
					rewrite=False, Dmp=Dmp, N_rep=N_rep, qtasks=qtasks,
					back_action=back_action, monitor_axis=monitor_axis,
					meas_strength=meas_strength, inp_type=inp_type)
			
		rng = np.random.default_rng(seed=seed)
		seeds = rng.integers(1, 1e9, size=(N_iter))
		
		for it in range(N_iter):

			if not load_obs:

				sample = data[it]
				obs = sample[0]
				inp = sample[1]
				
			else:

				data = load_observables_data(L, Js, W, h, dt, Vmp, Dmp, N_rep, task_name, it, inp_type=inp_type,
								 back_action=back_action, monitor_axis=monitor_axis, meas_strength=meas_strength)
				inp = data['inp']
				obs = data['obs']
			
			if noise:
				obs, noise_ofm = statistical_noise(obs.copy(), N_meas, meas_strength, L, Vmp, back_action, seeds[it])
				if noise_ridge_corr:
					reg_prep_param = Nmeas_RegPrepParam(Vmp, noise_ofm)
					print(reg_prep_param)

			task_qrc.x_out = obs[:, obs_idx] if not back_action else obs
			task_qrc.input_signals = inp
			task_hyb.input_signals = inp
			task_qrc.reset_quantum_input_features()
			task_hyb.reset_quantum_input_features()
			task_qrc.get_inputstate_tomography(inp_type=inp_type) # get spin projections of the input state

			# Feed ESN with this output signal
			esn = EsnDynamics(N_esn=N_esn, g=g, l=l, task_name=task_name, n_max_delay=0, seed=seeds[it])

			if onlyesn:
				solely_esn = EsnDynamics(N_esn=N_esn_solely, g=g, l=l, task_name=task_name, qtasks=qtasks, n_max_delay=0, seed=seeds[it], pm=pm, inp_type=inp_type)
				# Insert directly the spin projections of the input quantum state
				solely_esn.u = task_qrc.sigma_zxy_qinput if all_info else input_full_esn(inp, inp_type, axis=axis)
				solely_esn.input_signals = inp
				solely_esn.echo_states() # Get readout layer
				solely_esn.reset_quantum_input_features() # Reset taregt values

			for j, n_delay in enumerate(delays):

				task_hyb.n_max_delay = n_delay # Set the delaty of the final target

				qrc_delay = int(n_delay*ratio_delay_qrc) # Get delay of QRC
				task_qrc.n_max_delay = qrc_delay
				# task_qrc.get_output_signal(reshapeflag=True)

				if onlyesn:
					solely_esn.n_max_delay = n_delay # Set delay in the full ESN

				for i, qtask in enumerate(task_qrc.qtasks):

					q_preproces = np.full((task_qrc.n_steps, q_preproces_shape), fill_value=np.nan) # Array to store the spin projection predictions at tau_qrc of the input
					task_qrc.output_signals = np.zeros_like(q_preproces)
					
					task_qrc.output_signals[qrc_delay:] = task_qrc.sigma_zxy_qinput[:-qrc_delay] if qrc_delay else task_qrc.sigma_zxy_qinput
					task_qrc.split_data() # Get train and test data

					if not reg_prep_param:
						clf = linear_model.LinearRegression(fit_intercept=False)
					else:
						clf = linear_model.Ridge(fit_intercept=False, alpha=reg_prep_param)
					
					clf.fit(task_qrc.x_train, task_qrc.y_train)

					if task_qrc.bias:
						x_qrc = np.concatenate((np.ones((task_qrc.n_steps, 1)), task_qrc.x_out), axis=1)
					else:
						x_qrc = task_qrc.x_out

					q_preproces = clf.predict(x_qrc)

					# C of preprocessing layer
					if qrc_prep_perf:
						for p in range(q_preproces.shape[1]):
							C_preprocess[p,qrc_delay,it] = np.corrcoef(task_qrc.output_signals[3000:,p], q_preproces[3000:,p], rowvar=False)[0, 1]**2 # Square pearson corr

					esn.u = q_preproces
					esn.echo_states()

					task_hyb.x_out = esn.x_out
					qout_total = task_hyb.get_output_signal(qtasks=[qtask])
					task_hyb.output_signals = qout_total
					C[i,j,it] = task_hyb.performance()

					if onlyesn:
						solely_esn.get_output_signal(qtasks=[task_hyb.qtasks[i]])
						C_esn[i,j,it] = solely_esn.performance()
						# if task_hyb.n_max_delay == 15 and solely_esn.n_max_delay == 15:
						# 	plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})
						# 	plt.rcParams['axes.titlesize'] = 20
						# 	plt.rcParams['lines.linewidth'] = 3
						# 	plt.rcParams['axes.linewidth'] = 2
						# 	fig, ax = plt.subplots(figsize=(12,5), ncols=1, nrows=1)
						# 	ax.plot(range(3180,3250),np.maximum(0,task_hyb.y_test[180:250]), marker='D', color='k', markeredgecolor='k', markeredgewidth=1.5,
						# 			label='Target')
						# 	ax.plot(range(3180,3250),np.maximum(0,task_hyb.y_pred[180:250]), linewidth=2, marker='o', color='lime', markeredgecolor='k',
						# 			markeredgewidth=1.5, label='Prediction')
						# 	ax.plot(range(3180,3250),np.maximum(0,solely_esn.y_pred[180:250]), linewidth=2, marker='o', color='dodgerblue', markeredgecolor='k',
						# 			markeredgewidth=1.5, label='Prediction')
						# 	ax.set_xlabel('k')
						# 	ax.set_ylabel(r'C($\rho_{1,2}$)')
						# 	ax.legend(loc='best', frameon=False)
						# 	plt.show()
			
		C_mean = np.mean(C, axis=2) 
		C_std = np.std(C, axis=2)

		q_task_dict = {}
		for i, q_task in enumerate(task_qrc.qtasks):
			q_task_dict['C_mean '+q_task] = C_mean[i]
			q_task_dict['C_std '+q_task] = C_std[i]

		if store:
			fname = 'results/data/'
			if back_action:
				fname += f'back_action/{monitor_axis}/'
				fend = f'_MeasStr{meas_strength}'

			fname += f'{task_name}/{strqtasks}/HYB/{inp_type}/'
			fname += f'N_meas{N_meas}/' if noise else 'N_measInf/'
			pathlib.Path(fname).mkdir(parents=True, exist_ok=True)
			fname += f'{pm}_L{L}_Js{Js}_V{Vmp}_D{Dmp}_Nrep{N_rep}_h{h}_W{W}_dt{dt}_ax_{ax_str}_cax_{cax_str}_Nesn{N_esn}_g{g}_l{l}_delayqrc{ratio_delay_qrc}_sweep_delay'

			if back_action:
				fname += fend

			np.savez_compressed(fname, delays=delays, **q_task_dict)

		if qrc_prep_perf:

			C_prep_mean = np.mean(C_preprocess, axis=2)
			C_prep_std = np.std(C_preprocess, axis=2)

			prep_dict = {}
			for i, a in enumerate(['z', 'x', 'y']):
				prep_dict['C_mean '+a] = C_prep_mean[i]
				prep_dict['C_std '+a] = C_prep_std[i]

			if store:
				fprepname = 'results/data/'
				if back_action:
					fprepname += f'back_action/{monitor_axis}/'
					fprepend = f'_MeasStr{meas_strength}'

				fprepname += f'{task_name}/QPreprocess/{inp_type}/'
				fprepname += f'N_meas{N_meas}/' if noise else 'N_measInf/'
				pathlib.Path(fprepname).mkdir(parents=True, exist_ok=True)
				fprepname += f'Capacity_L{L}_Js{Js}_V{Vmp}_D{Dmp}_Nrep{N_rep}_h{h}_W{W}_dt{dt}_ax_{ax_str}_sweep_delay'

				if back_action:
					fprepname += fprepend

				np.savez_compressed(fprepname, delays=delays, **prep_dict)

		if onlyesn:

			C_mean_esn = np.mean(C_esn, axis=2) 
			C_std_esn = np.std(C_esn, axis=2) 
			
			q_task_dict = {}
			for i, q_task in enumerate(task_qrc.qtasks):
				q_task_dict['C_mean '+q_task] = C_mean_esn[i]
				q_task_dict['C_std '+q_task] = C_std_esn[i]

			if store:
				fname = f'results/data/{task_name}/{strqtasks}/ESN/{inp_type}/{pm}_L{L}_Js{Js}_V{Vmp}_D{Dmp}_Nrep{N_rep}_h{h}_W{W}_dt{dt}_ax_{ax_str}_cax_{cax_str}_Nesn{N_esn_solely}_g{g}_l{l}_sweep_delay'
				pathlib.Path(f'results/data/{task_name}/{strqtasks}/ESN/{inp_type}/').mkdir(parents=True, exist_ok=True)
				np.savez_compressed(fname, delays=delays, **q_task_dict)

		return delays, C_mean, C_std




				




		





		
		
