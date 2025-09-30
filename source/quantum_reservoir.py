""" This file contains QuantumReservoirDynamics class. """

from .hamiltonian import *
from .tasks import Tasks
from cycler import cycler
import pathlib
import os
from .utils import load_observables_data, get_obs_idx, ax_to_str

class QuantumReservoirDynamics(Hamiltonian, Tasks):

    """ Class to compute the quantum reservoir dynamics. """

    def __init__(
        self, L, J, h, W, dt=10, sx=None, sy=None, sz=None, correlations_x=None, correlations_y=None, correlations_z=None,
        correlations_xy=None, correlations_zx=None, correlations_zy=None, 
        random_rho_0=False, axis=['z','x','y'], caxis=['z','x','y'], ccaxis=['zx', 'xy', 'zy'], Vmp=1, N_rep=1, **kwargs):

        """
        Initialize the QuantumReservoirDynamics class.
        Parameters:
        L (int): number of qubits
        J (float): coupling constant
        h (float): transverse field
        W (float): local disorder strength
        dt (float): time step for the time evolution (time between input signals)
        s_i (list): list with sigma_i oeprators (i\in{x,y,z}) for each spin site.
        correlations_i (list): spin correlation operators in the axis i.
        random_rho_0 (bool): if the initial matrix is random or maximal coherent.
        axis (list): list with the strings of the axis observables to consider.
        caxis (list): list with the strings of the axis correlations to consider.
        Vmp (int): subdivided time steps for time multiplexing.
        N_rep (int): number of repetitions of the input injection
        """

        super().__init__(L=L, J=J, h=h, W=W, **kwargs) # initialize the parent classes

        self.Vmp = Vmp
        self.dt = dt/Vmp # time step between signal injection

        # Initalize the Observable operators
        self.axis = axis
        self.caxis = caxis
        self.ccaxis = ccaxis

        # Local spin operators for each axis in self.axis
        self.s_ops = {}
        for ax in self.axis:
            self.s_ops[ax] = locals().get(f"s{ax}") if locals().get(f"s{ax}") is not None else self.local_spin_operators(axis=ax)

        # Two-spin correlation operators for each axis in self.caxis
        self.corr_ops = {}
        for cax in self.caxis:
            corr_var = locals().get(f"correlations_{cax}")
            self.corr_ops[cax] = corr_var if corr_var is not None else self.two_spins_correlations(axis=cax)

        # Two-spin correlation operators for each axis in self.ccaxis
        self.ccorr_ops = {}
        for ccax in self.ccaxis:
            ccorr_var = locals().get(f"correlations_{ccax}")
            self.ccorr_ops[ccax] = ccorr_var if ccorr_var is not None else self.two_spins_correlations(axis=ccax)

        self.random_rho_0 = random_rho_0 # Random initial matrix
        self.trace_indices = list(range(2,self.L)) if self.task_name == 'Qinp' and self.inp_type in ['2qubit', 'werner', 'x_state'] else list(range(1, self.L))
        self.N_rep = N_rep

    def __repr__(self):

        """ Return a representation of the QuantumReservoirDynamics class. """

        return f"QuantumReservoirDynamics(L={self.L}, J={self.Js}, h={self.h}, W={self.W}, n_steps={self.n_steps}, dt={self.dt}, local_observables={self.axis}, correlations={self.caxis}, axis correlations={self.ccaxis})"
    
    def __str__(self):

        """ Return a string description of the QuantumReservoirDynamics class. """

        return f"QuantumReservoirDynamics class with {self.L} qubits, coupling constant J={self.Js}, transverse field h={self.h}, disorder strength W={self.W}, number of time steps {self.n_steps}, and time step {self.dt}."
    
    def get_time_evol_op(self):

        """
        Returns the time evolution operator for the system.
        """

        E, V = self.get_E_V()
        U = V @ np.diag(np.exp(-1j*E*self.dt)) @ V.conj().T
        self.U = Qobj(U, dims=[[2]*self.L]*2)
        self.Udag = self.U.dag()
        return self.U, self.Udag

    def get_initial_density_matrix(self):

        """
        Returns the initial density matrix for the system.
        """
        if self.random_rho_0:
            self.rho_0 = rand_dm([2]*self.L, seed=self.seed)

        else: 
            state = (basis(2,0) + basis(2,1)).unit()
            psi_coh = tensor([state] * self.L)
            self.rho_0 = ket2dm(psi_coh)

        return self.rho_0
    
    def get_initial_obs_storage(self):

        """
        Returns the initial storage for the observables.
        """

        self.L_corr = np.sum(range(self.L))
        self.L_ccorr = 2 * self.L_corr
        self.store_local_obs = np.full((len(self.s_ops), self.n_steps, self.L*self.Vmp), np.nan)
        self.store_corr_obs = np.full((len(self.corr_ops), self.n_steps, self.L_corr*self.Vmp), np.nan)
        self.store_ccorr_obs = np.full((len(self.ccorr_ops), self.n_steps, self.L_ccorr*self.Vmp), np.nan)


        return self.store_local_obs, self.store_corr_obs, self.store_ccorr_obs
    
    def get_initialization(self):

        """
        Returns the initial density matrix, the time evolution operator, and the initial storage for the observables.
        """
        
        self.get_initial_density_matrix()
        self.get_time_evol_op()
        self.get_initial_obs_storage()

        return self.rho_0, self.U, self.Udag, self.store_local_obs, self.store_corr_obs, self.store_ccorr_obs

    def input_state(self):

        """
        Returns the input state for the system.
        The input state is a superposition of the |0> and |1> states.
        """

        # Qubits states
        ket0 = basis(2,0)
        ket1 = basis(2,1)
        self.state_sk = [np.sqrt(1-self.sk_array[i])*ket0 + np.sqrt(self.sk_array[i])*ket1 for i in range(len(self.sk_array))] # input signal injected in the first qubit
        # 1st qubit density matrix
        self.rho_1 = [self.state_sk[i]*self.state_sk[i].dag() for i in range(len(self.state_sk))] # input signal injected in the first qubit

        return self.rho_1
    
    def get_unitary_evolve_density(self, rho):

        """
        Return the density matrixafter unitary evolution.
        """

        rhot = self.U @ rho @ self.Udag
        return rhot
    
    @staticmethod
    def input_update_qubit_RC(U, Udag, rho_0, rho_1, trace_indices):

        # Partial trace over the first qubit (remove the first qubit from the system)
        Tr1 = rho_0.ptrace(trace_indices) # Keep the reservoir spins except the first qubit
        # Inject the input state to the full system
        M = tensor(rho_1, Tr1)

        rho_t = U @ M @ Udag # time evolution of the density matrix

        return rho_t
    
    def quantum_outputs_time_evolution(self):

        """
        Returns the expectation value of the observables at each time step.
        """

        # Get the initial density matrix, time evolution operator, and observable storage
        self.get_initialization()

        if not hasattr(self, 'input_signals') or self.input_signals is None:
            print('generating input signal')
            input_signal = self.get_input_signal() # generate the input signal
        else:
            input_signal = self.input_signals

        if self.task_name == "Qinp":
            self.rho_1 = input_signal # If the input is quantum, we replace a random qubit as input signal.
        else:
            reescale = 1 / self.max_bound_input
            self.sk_array =  reescale * input_signal
            self.input_state() # generate the input states

        for k in range(self.n_steps):

            for _ in range(self.N_rep):
                self.rho_0 = self.input_update_qubit_RC(self.U, self.Udag, self.rho_0, self.rho_1[k], self.trace_indices)

            for i, ax in enumerate(self.axis):
                self.store_local_obs[i, k, :self.L] = self.average_values(self.rho_0, self.s_ops[ax])
            for j, ax in enumerate(self.caxis):
                self.store_corr_obs[j, k, :self.L_corr] = self.average_values(self.rho_0, self.corr_ops[ax])
            for p, ax in enumerate(self.ccaxis):
                self.store_ccorr_obs[p, k, :self.L_ccorr] = self.average_values(self.rho_0, self.ccorr_ops[ax])

            for d in range(1,self.Vmp):
                self.rho_0 = self.get_unitary_evolve_density(self.rho_0)

                for i, ax in enumerate(self.axis):
                    self.store_local_obs[i, k, d*self.L:(d+1)*self.L] = self.average_values(self.rho_0, self.s_ops[ax])
                for j, ax in enumerate(self.caxis):
                    self.store_corr_obs[j, k, d*self.L_corr:(d+1)*self.L_corr] = self.average_values(self.rho_0, self.corr_ops[ax])
                for j, ax in enumerate(self.ccaxis):
                    self.store_ccorr_obs[j, k, d*self.L_ccorr:(d+1)*self.L_ccorr] = self.average_values(self.rho_0, self.ccorr_ops[ax])
                

        local_obs_list = [self.store_local_obs[i] for i in range(len(self.axis))]
        corr_obs_list = [self.store_corr_obs[j] for j in range(len(self.caxis))]
        ccorr_obs_list = [self.store_ccorr_obs[j] for j in range(len(self.ccaxis))]
        self.x_out = np.concatenate(local_obs_list + corr_obs_list + ccorr_obs_list, axis=1)

        return self.x_out
    
    @staticmethod
    def qrc_worker(
        L, Js, h, W, task_name="NARMA", dt=10, idx_iter=0, 
        seed=None, sx=None, sy=None, sz=None, correlations_x=None, correlations_y=None, correlations_z=None,
        correlations_xy=None, correlations_zx=None, correlations_zy=None,
        max_bound_input=None, axis=['z','x','y'], caxis=['z'], ccaxis=[], Vmp=1, store=True, Dmp=1, N_rep=1, qtasks=[], inp_type='qubit', **kwargs):

        """
        Worker function to compute the prediction performance of the reservoir dynamics for a single combination of h, W, and seed.
        This function is used for parallel computation.
        """

        obs = []
        task = Tasks(0, task_name=task_name, qtasks=qtasks, seed=seed, inp_type=inp_type)
        task.get_input_signal()

        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 1e9, size=(Dmp))

        for i in range(Dmp):

            # Create an instance of the TimeEvolution class
            res = QuantumReservoirDynamics(
                L, Js, h, W, task_name=task_name, dt=dt, seed=seeds[i], sx=sx, sy=sy, sz=sz, 
                correlations_x=correlations_x, correlations_y=correlations_y, correlations_z=correlations_z,
                correlations_xy=correlations_xy, correlations_zx=correlations_zx, correlations_zy=correlations_zy,
                max_bound_input=max_bound_input, axis=axis, caxis=caxis, ccaxis=ccaxis, Vmp=Vmp, n_max_delay=0,
                inp_type=inp_type, N_rep=N_rep, qtasks=qtasks, **kwargs)
            
            res.input_signals = task.input_signals
            res.max_bound_input = task.max_bound_input
            res.quantum_outputs_time_evolution() # computes the observables time evolution
            obs.append(res.x_out)

        total_obs = np.concatenate(obs, axis=1)    

        if store:
            if 'Entanglement' in qtasks:
                path = f'results/data/{task_name}/QRC/{inp_type}/L{L}_Js{Js}_h{h}_W{W}_dt{dt}_V{Vmp}_D{Dmp}_Nrep{N_rep}'
            else:
                if task_name == 'Qinp':
                    path = f'results/data/{task_name}/QRC/{inp_type}/L{L}_Js{Js}_h{h}_W{W}_dt{dt}_V{Vmp}_D{Dmp}_Nrep{N_rep}'
                else:
                    path = f'results/data/{task_name}/QRC/L{L}_Js{Js}_h{h}_W{W}_dt{dt}_V{Vmp}_D{Dmp}_Nrep{N_rep}'

            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f'{path}/Iter_{idx_iter}', obs=total_obs, inp=task.input_signals)

        return total_obs, task.input_signals
    
    @staticmethod
    def qrc_obs(
        L, Js, N_iter, task_name="NARMA",
        dt=10, axis=['z', 'x', 'y'], caxis=['z', 'x', 'y'], ccaxis=['zx', 'xy', 'zy'], Vmp=1,
        max_bound_input=None, seed=None, store=True,
        sweep_param="W", sweep_values=None, fixed_h=None, fixed_W=None, rewrite=False, qtasks=[],
        Dmp=1, N_rep=1, **kwargs):
        """
        Compute QRC performance over a sweep of W or h, or with both fixed.

        Additional parameters:
            sweep_param (str): "W", "h", or None (no sweep)
            sweep_values (array): values to sweep over if sweep_param is not None
            fixed_W (float): fixed W value (required if sweep_param != "W")
            fixed_h (float): fixed h value (required if sweep_param != "h")
            N_iter (int): number of iterations
        """
        if sweep_param not in ["W", "h", None]:
            raise ValueError("sweep_param must be 'W', 'h', or None.")

        # Get observables
        sx, sy, sz, correlations_x, correlations_y, correlations_z, correlations_xy, correlations_zx, correlations_zy = Hamiltonian.get_all_observables(L=L)
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 1e9, size=(N_iter if sweep_param is None else len(sweep_values) * N_iter))

        # Helper to compute one config
        def run_for_config(h, W, i_offset=0):

            if 'Entanglement' in qtasks:
                dir_path = f'results/data/{task_name}/QRC/werner/L{L}_Js{Js}_h{h}_W{W}_dt{dt}_V{Vmp}_D{Dmp}_Nrep{N_rep}/'
            else:
                dir_path = f'results/data/{task_name}/QRC/L{L}_Js{Js}_h{h}_W{W}_dt{dt}_V{Vmp}_D{Dmp}_Nrep{N_rep}/'

            missing_k = []

            if rewrite:

                missing_k = range(N_iter)

            else:

                for k in range(N_iter):
                    file_path = dir_path + f'Iter_{k}.npz'
                    if not os.path.exists(file_path):
                        missing_k.append(k)

                if not missing_k:
                    print(f"All iterations already computed for h = {h}, W = {W}. Skipping.")
                    return

            args = [
                (L, Js, h, W, task_name, dt, idx_iter, seeds[i_offset + k], sx, sy, sz,
                correlations_x, correlations_y, correlations_z,
                correlations_xy, correlations_zx, correlations_zy,
                max_bound_input, axis, caxis, ccaxis, Vmp, store, Dmp, N_rep, qtasks)
                for k, idx_iter in enumerate(missing_k)
            ]

            results = Parallel(n_jobs=-1)(
                delayed(QuantumReservoirDynamics.qrc_worker)(*arg, **kwargs)
                for arg in args
            )

            return results

        # Mode 1: sweep W
        if sweep_param == "W":
            if fixed_h is None:
                raise ValueError("Must provide fixed_h when sweeping W.")
            if sweep_values is None:
                raise ValueError("Must provide sweep values list for W.")
            for i, W in enumerate(sweep_values):
                results = run_for_config(fixed_h, W, i*N_iter)

        # Mode 2: sweep h
        elif sweep_param == "h":
            if fixed_W is None:
                raise ValueError("Must provide fixed_W when sweeping h.")
            if sweep_values is None:
                raise ValueError("Must provide sweep values list for h.")
            for i, h in enumerate(sweep_values):
                results = run_for_config(h, fixed_W, i*N_iter)

        # Mode 3: fixed W and h
        else:
            if fixed_h is None or fixed_W is None:
                raise ValueError("Must provide both fixed_h and fixed_W when no sweep.")
            results = run_for_config(fixed_h, fixed_W)

        return results


    @staticmethod 
    def qrc_performance(
        L, Js, sweep_values=None, fixed_param=None, fixed_W=None, fixed_h=None,
        N_iter=10, task_name="NARMA", n_min_delay=0, n_max_delay=10,
        dt=10, axis=['z'], caxis=[], ccaxis=[], Vmp=1, pm='Capacity', Dmp=1, N_rep=1,
        store=True, sweep_param="W", seed=None, load_obs=False, qtasks=[], inp_type='qubit', **kwargs):
        
        """
        Compute QRC performance either by sweeping h or W, or by keeping both fixed.

        Parameters:
            sweep_param (str or None): 'W', 'h', or None for no sweep
            sweep_values (array or None): values to sweep (W or h), required if sweep_param is not None
            fixed_param (float): required when sweep_param is not None
            fixed_W, fixed_h (float): required when sweep_param is None
            N_iter (int): number of iterations per setting
        """

        if task_name == 'PC' and n_min_delay < 1:
            n_min_delay = 1
            print('For PC task the minimum delay is 1: n_min_delay set to 1')

        delays = list(range(n_min_delay, n_max_delay))
        task = Tasks(n_max_delay=0, task_name=task_name, pm=pm, qtasks=qtasks, inp_type=inp_type)
        strqtasks = task.strqtasks

        obs_idx, _ = get_obs_idx(L, axis, caxis, ccaxis, Vmp, Dmp)
        ax_str, cax_str = ax_to_str(axis, caxis)

        if sweep_param not in ['W', 'h', None]:
            raise ValueError("sweep_param must be 'W', 'h', or None.")

        if sweep_param is None:

            # --- Fixed h and W: performance as a function of delay only ---
            if fixed_W is None or fixed_h is None:
                raise ValueError("When sweep_param is None, both fixed_W and fixed_h must be provided.")

            if task_name == 'Qinp':
                C = np.full((len(delays), N_iter, task.nqtasks), np.nan)
            else:
                C = np.full((len(delays), N_iter), np.nan)

            if load_obs == False:

                data = QuantumReservoirDynamics.qrc_obs(
                    L=L, Js=Js, N_iter=N_iter, task_name=task_name, dt=dt, Vmp=Vmp, seed=seed,
                    store=False, sweep_param=sweep_param, fixed_h=fixed_h, fixed_W=fixed_W, 
                    rewrite=True, Dmp=Dmp, N_rep=N_rep, qtasks=qtasks, **kwargs)

            for it in range(N_iter):

                if load_obs:

                    data = load_observables_data(L, Js, fixed_W, fixed_h, dt, Vmp, Dmp, N_rep, task_name, it, inp_type)
                    inp = data['inp']
                    obs = data['obs']
                
                else:

                    sample = data[it]
                    obs = sample[0]
                    inp = sample[1]
                
                task.x_out = obs[:, obs_idx]
                task.input_signals = inp

                if task_name == 'Qinp':
                    task.reset_quantum_input_features()              

                for j, n_delay in enumerate(delays): 

                    task.n_max_delay = n_delay
                    task.get_output_signal(reshapeflag=True)
                    C[j,it] = task.performance(qflag=True)

            C_mean = np.mean(C, axis=1)
            C_std = np.std(C, axis=1)

            if store:
                
                if task_name == 'Qinp':
                    pathlib.Path(f'results/data/{task_name}/{strqtasks}/QRC/{inp_type}').mkdir(parents=True, exist_ok=True)
                    fname = f'results/data/{task_name}/{strqtasks}/QRC/{inp_type}/{pm}_L{L}_Js{Js}_V{Vmp}_D{Dmp}_Nrep{N_rep}_h{fixed_h}_W{fixed_W}_dt{dt}_ax_{ax_str}_cax_{cax_str}_sweep_delay'
                    q_task_dict = {}
                    for i, qtask in enumerate(task.qtasks):
                        q_task_dict['C_mean '+qtask] = C_mean[:,i]
                        q_task_dict['C_std '+qtask] = C_std[:,i]                
                
                    np.savez_compressed(fname, delays=delays, **q_task_dict)

                else:
                    
                    pathlib.Path(f'results/data/{task_name}/QRC').mkdir(parents=True, exist_ok=True)
                    fname = f'results/data/{task_name}/QRC/{pm}_L{L}_Js{Js}_V{Vmp}_D{Dmp}_Nrep{N_rep}_h{fixed_h}_W{fixed_W}_dt{dt}_ax_{ax_str}_cax_{cax_str}_sweep_delay'
                    np.savez_compressed(fname, delays=delays, C_mean=C_mean, C_std=C_std)

            return delays, C_mean, C_std

        else:

            # --- Sweeping h or W ---
            if sweep_values is None or fixed_param is None:
                raise ValueError("Must provide sweep_values and fixed_param when sweeping.")
            
            if task_name == 'Qinp':
                C_store = np.full((len(sweep_values), N_iter, task.nqtasks), np.nan)
            else:
                C_store = np.full((len(sweep_values), N_iter), np.nan)

            for i, val in enumerate(sweep_values):
                
                if sweep_param == "W":
                    W = val
                    h = fixed_param
                    fixed_str = 'h'
                else:
                    h = val
                    W = fixed_param
                    fixed_str = 'W'
                    print('h', h)
                
                if load_obs == False:

                    data = QuantumReservoirDynamics.qrc_obs(
                        L=L, Js=Js, N_iter=N_iter, task_name=task_name, dt=dt, Vmp=Vmp, seed=seed,
                        store=False, sweep_param=None, fixed_h=h, fixed_W=W, rewrite=True, Dmp=Dmp, N_rep=N_rep,
                        qtasks=qtasks, **kwargs)

                for it in range(N_iter):

                    if load_obs:

                        data = load_observables_data(L, Js, W, h, dt, Vmp, Dmp, N_rep, task_name, it, inp_type)
                        inp = data['inp']
                        obs = data['obs']

                    else:

                        sample = data[it]
                        obs = sample[0]
                        inp = sample[1]

                    task.x_out = obs[:,obs_idx]
                    task.input_signals = inp

                    if task_name == 'Qinp':
                        task.reset_quantum_input_features()

                    C_delay = 0

                    for _, n_delay in enumerate(delays): 
                        
                        task.n_max_delay = n_delay
                        task.get_output_signal(reshapeflag=True)
                        
                        C = task.performance(qflag=True)

                        C_delay += C

                    C_store[i, it] = C_delay

                C_mean = np.mean(C_store, axis=1)
                C_std = np.std(C_store, axis=1)

            if store:
                
                if task_name == 'Qinp':

                    pathlib.Path(f'results/data/{task_name}/{strqtasks}/QRC/{inp_type}').mkdir(parents=True, exist_ok=True)
                    fname = f'results/data/{task_name}/{strqtasks}/QRC/{inp_type}/{pm}_L{L}_Js{Js}_V{Vmp}_D{Dmp}_Nrep{N_rep}_{fixed_str}{fixed_param}_dt{dt}_ax_{ax_str}_cax_{cax_str}_sweep{sweep_param}'

                    q_task_dict = {}
                    for i, qtask in enumerate(task.qtasks):
                        q_task_dict['C_mean '+qtask] = C_mean[:,i]
                        q_task_dict['C_std '+qtask] = C_std[:,i]
                    np.savez_compressed(fname, sweep_values=sweep_values, **q_task_dict)

                else:

                    pathlib.Path(f'results/data/{task_name}/QRC').mkdir(parents=True, exist_ok=True)
                    fname = f'results/data/{task_name}/QRC/{pm}_L{L}_Js{Js}_V{Vmp}_D{Dmp}_Nrep{N_rep}_{fixed_str}{fixed_param}_dt{dt}_ax_{ax_str}_cax_{cax_str}_sweep{sweep_param}'
                    np.savez_compressed(fname, sweep_values=sweep_values, C_mean=C_mean, C_std=C_std)

            return sweep_values, C_mean, C_std