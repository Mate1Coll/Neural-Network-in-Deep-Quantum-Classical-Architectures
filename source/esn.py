""" This file contains Esn and EsnDynamics classes. """

from .base import *
from .tasks import Tasks
import itertools
import pathlib
import os
from .utils import input_full_esn, ax_to_str, load_state_data

class Esn(BaseSeededClass):

    """ Class to define the ESN """

    def __init__(self, N_esn, g, l, d=1, func='sigmoid', N_rep=1, **kwargs):

        """
        Initialize the ESN class.
        Parameters:
        N_esn (int): number of neurons in the ESN
        g (float): feedback gain for the ESN
        l (int): input gain for the ESN
        d (int): dimension of the input signal (set to 1 by default)
        func (str): activation function for the reservoir neurons
        seed (int): random seed for reproducibility
        """

        super().__init__(**kwargs) # initialize the parent class

        if func not in ['tanh', 'sigmoid']:
            raise ValueError("The activation function must be either 'tanh' or 'sigmoid'.")

        self.N_esn = N_esn
        self.g = g
        self.l = l
        self.func = func
        self.N_rep = N_rep

        self.d = d

    def __repr__(self):

        """ Return a representation of the Esn class. """

        return f"Esn(N_esn={self.N_esn}, g={self.g}, l = {self.l}, d={self.d}, func={self.func}, seed={self.seed})"
    
    def __str__(self):

        """Return a description of the Esn class. """

        return f"Echo State Networ with {self.N_esn} neurons, g = {self.g} feedback gain, l = {self.l} input gain and activation function '{self.func}'."        

    def set_weigths(self):

        """ Set the weights of the ESN 
        W_esn: internal connections between the neurons (the weigths are randomly generated uniformaly between -1 and 1,
        and then normalized to have a spectral radius of 1)
        W_in: input connections (the weights are randomly generated uniformaly between -1 and 1)
        x0: initial state of the reservoir (the initial state is randomly generated uniformaly between -1 and 1)"""

        self.W_esn = self.rng.uniform(-1, 1, size=(self.N_esn, self.N_esn)) # reservoir weights
        self.W_in = self.rng.uniform(-1, 1, size=(self.N_esn, self.d)) # input weights
        self.x0 = self.rng.uniform(-1, 1, size=(self.N_esn)) # initial state of the reservoir

        rho = np.max(np.abs(np.linalg.eigvals(self.W_esn))) # spectral radius of the reservoir weights
        self.W_esn = self.W_esn / rho # normalize the reservoir weights

        return self.W_esn, self.W_in, self.x0
    
    def activation_f(self):

        """ Activation function """

        if self.func == 'tanh':
            self.af = np.tanh

        if self.func == 'sigmoid':
            self.af = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            
        return
    

    
class EsnDynamics(Esn, Tasks):

    """ Class to compute the Echo State Network (ESN) dynamics. """

    def __init__(self, N_esn, g, l, func='sigmoid', **kwargs):

        super().__init__(N_esn=N_esn, g=g, l=l, func=func, **kwargs)
        
        self.activation_f() # set the activation function

    def __repr__(self):

        """ Return a represenation of EsnDynamics """

        return Esn.__repr__(self) + "\n" + Tasks.__repr__(self)

    def __str__(self):

        """ Returna description of EsnDynamics """

        return Esn.__str__(self) + "\n" + Tasks.__str__(self)
    
    # def get_qinput_to_classical(self):

    #     """ Transforms the quantum input as classical """

    #     qtL = self.nqtasks
    #     qinp_to_clas = np.zeros((qtL, self.n_steps))
    #     j=0

    #     for qtask in self.qtasks:
    #         if qtask == "Trace":
    #             dm2 = np.array([i*i for i in self.input_signals])
    #             qinp_to_clas[j] = np.array([m.tr() for m in dm2])
    #             j+=1
    #         if qtask == "Det":
    #             qinp_to_clas[j] = np.array([abs(np.linalg.det(i.full())) for i in self.input_signals])
    #             j+=1
    #         if qtask == "Entropy":
    #             eigval = np.array([i.eigenenergies() for i in self.input_signals])
    #             qinp_to_clas[j] = np.array([-np.sum(e*np.log(e)) for e in eigval])
    #             j+=1

    #     return qinp_to_clas

    
    def input_signal_reshape(self):
    
        """ Reshape the input signal to match the dimension of W_in for proper matrix multiplication """

        if not hasattr(self, 'input_signals'):
            print('There is no input signal, generating ...')
            u = self.get_input_signal() # input signal

        else:
            u = self.input_signals

        self.u = u.reshape(-1, 1) # reshape the input signal to match the dimension of W_in
        self.u = self.u * 1 / self.max_bound_input # Reescale input to ESN such u in [0,1]

        return self.u
    
    def echo_states(self):

        """ Computes the Echo states of the ESN given the input signal """

        self.d = self.u.shape[1]
        self.set_weigths() # set the weights of the ESN

        self.x_out = np.zeros((self.n_steps, self.N_esn))
        self.x_out[0] = self.x0 # initial state of the reservoir
        arg=np.zeros_like(self.x_out)

        for i in range(1, self.n_steps):
            # Compute the echo state
            x_prev = self.x_out[i-1]
            for _ in range(self.N_rep):
                rec_term = (self.g * (self.W_esn @ x_prev))
                in_term  = (self.l * (self.W_in @ self.u[i]))
                x_new = self.af(rec_term + in_term) # update the state of the reservoir
                arg[i] = rec_term + in_term
                x_prev = x_new

            self.x_out[i] = x_new

        return self.x_out
    
    def classical_output_weights(self):

        """ Computes the output weights of the ESN using linear regression """
        self.input_signal_reshape()
        return self.echo_states()
    

    @staticmethod
    def esn_worker(N_esn, g, l, task_name, idx_iter, axis=['x'], qtasks=[], seed=None, store=True, inp_type='qubit'):

        """ Computes the state reservoir dynamics of the readout layer of the ESN """

        esn = EsnDynamics(N_esn, g, l, task_name=task_name, qtasks=qtasks, n_max_delay = 0, seed=seed, inp_type=inp_type)
        esn.get_input_signal()

        if task_name == "Qinp":
            esn.u = input_full_esn(esn.input_signals, inp_type, axis=axis)
            esn.echo_states()
        else:
            esn.classical_output_weights()

        if store:

            if task_name == "Qinp":
                ax_str , _ = ax_to_str(axis, caxis=[])
                pathlib.Path(f'results/data/{task_name}/ESN/{inp_type}/ESN_Nesn{N_esn}_g{g}_l{l}_ax_{ax_str}_{inp_type}').mkdir(parents=True, exist_ok=True)
                np.savez_compressed(f'results/data/{task_name}/ESN/{inp_type}/ESN_Nesn{N_esn}_g{g}_l{l}_ax_{ax_str}_{inp_type}/Iter_{idx_iter}', states=esn.x_out, inp=esn.input_signals)

            else:
                pathlib.Path(f'results/data/{task_name}/ESN/ESN_Nesn{N_esn}_g{g}_l{l}').mkdir(parents=True, exist_ok=True)
                np.savez_compressed(f'results/data/{task_name}/ESN/ESN_Nesn{N_esn}_g{g}_l{l}/Iter_{idx_iter}', states=esn.x_out, inp=esn.u)

        return esn.x_out, esn.input_signals
    
    @staticmethod
    def esn_state_g_l(N_esn, g_sweep_val, l_sweep_val, task_name, N_iter, qtasks=[], axis=['x'],
                      seed=None, rewrite=False, store=True, inp_type='qubit'):

        """ Computes the state of the reservoir for various combiantions of g and l values. """

        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 1e9, size=N_iter)

        ax_str , _ = ax_to_str(axis=axis, caxis=[])

        if type(g_sweep_val) in [int, float]:
            g_sweep_val = [g_sweep_val]
        if type(l_sweep_val) in [int, float]:
            l_sweep_val = [l_sweep_val]

        for g in g_sweep_val:
            for l in l_sweep_val:

                if qtasks == 'Qinp':
                    dir_path = f'results/data/{task_name}/ESN/{inp_type}/ESN_Nesn{N_esn}_g{g}_l{l}_ax_{ax_str}/'
                else:
                    dir_path = f'results/data/{task_name}/ESN/ESN_Nesn{N_esn}_g{g}_l{l}/'
                missing_k = []

                if rewrite:

                    missing_k = range(N_iter)

                else:

                    for k in range(N_iter):
                        file_path = dir_path + f'Iter_{k}.npz'
                        if not os.path.exists(file_path):
                            missing_k.append(k)

                    if not missing_k:
                        print(f"All iterations already computed for g = {g}, l = {l}. Skipping.")
                        continue

                args = [
                    (N_esn, g, l, task_name, idx_iter, axis, qtasks, seeds[k], store, inp_type)
                    for k, idx_iter in enumerate(missing_k)
                ]

                results = Parallel(n_jobs=-1)(
                    delayed(EsnDynamics.esn_worker)(*arg) for arg in args
                )

        return results
    
    def esn_performance(
        N_esn, task_name, n_min_delay, n_max_delay, N_iter,
        sweep=True,  g_fixed=None, l_fixed=None, g_sweep_val=None, l_sweep_val=None,
        pm = 'Capacity', qtasks=[], axis=['x'], store=True, seed=None, load_data=False,
        inp_type='qubit'):

        """ 
        Computes ESN performance either by sweeping l and g, or by keeping both fixed.

        Parameters:
            sweep (bool): If True sweeps l and g, False for no sweep
            g_fixed, h_fixed (float or None): required when sweep is False
            g_sweep_val, l_sweep_val (array or None): required when sweep is True
            N_iter (int): number of iterations to compute performance
        """

        if task_name == 'PC' and n_min_delay < 1:
            n_min_delay = 1
            print('For PC task the minimum delay is 1: n_min_delay set to 1')

        delays = list(range(n_min_delay, n_max_delay))
        task = Tasks(n_max_delay=0, task_name=task_name, pm=pm, qtasks=qtasks)
        ax_str , _ = ax_to_str(axis=axis, caxis=[]); nqtasks = task.nqtasks

        if sweep:

            if g_sweep_val is None or l_sweep_val is None:
                raise ValueError("When sweep is True, g_sweep_val and l_sweep_val must be provided.")
            
            C_shape = (len(g_sweep_val), len(l_sweep_val), N_iter, nqtasks) if task_name == 'Qinp' else (len(g_sweep_val), len(l_sweep_val), N_iter)
            C_store = np.full(C_shape, fill_value=np.nan)

            for i, g in enumerate(g_sweep_val):
                for j, l in enumerate(l_sweep_val):

                    if not load_data:
                    
                        data = EsnDynamics.esn_state_g_l(
                            N_esn, g_sweep_val=[g], l_sweep_val=[l], task_name=task_name,
                            qtasks=qtasks, axis=axis, N_iter=N_iter, seed=seed,
                            rewrite=True, store=False, inp_type=inp_type)
                        
                    for it in range(N_iter):

                        if load_data:

                            data = load_state_data(N_esn, g, l, task_name, it, ax_str=ax_str, inp_type=inp_type)
                            inp = data['inp']
                            states = data['states']
                        
                        else:

                            sample = data[it]
                            states = sample[0]
                            inp = sample[1]

                        task.input_signals = inp
                        task.x_out = states
                        C_delay = 0

                        if task_name == 'Qinp':
                            task.reset_quantum_input_features()

                        for _, n_delay in enumerate(delays):

                            task.n_max_delay = n_delay
                            task.get_output_signal(reshapeflag=True)
                            C = task.performance(qflag=True)

                            C_delay += C

                        C_store[i,j,it] = C_delay

            
            C_mean = np.mean(C_store, axis=2)
            C_std = np.std(C_store, axis=2)

            if store:

                if task_name == 'Qinp':
                    
                    strqtasks = task.strqtasks
                    pathlib.Path(f'results/data/{task_name}/{strqtasks}/ESN/{inp_type}').mkdir(parents=True, exist_ok=True)
                    fname = f'results/data/{task_name}/{strqtasks}/ESN/{inp_type}/{pm}_Nesn{N_esn}_sweep_gl_ax_{ax_str}'

                    q_task_dict = {}
                    for i, qtask in enumerate(task.qtasks):
                        q_task_dict['C_mean '+qtask] = C_mean[:,:,i]
                        q_task_dict['C_std '+qtask] = C_std[:,:,i]
                    np.savez_compressed(fname, g_val=g_sweep_val, l_val=l_sweep_val, **q_task_dict)

                else:

                    fname = f'results/data/{task_name}/ESN/{pm}_Nesn{N_esn}_sweep_gl'
                    np.savez_compressed(fname, g_val=g_sweep_val, l_val=l_sweep_val, C_mean=C_mean, C_std=C_std)

            return g_sweep_val, l_sweep_val, C_mean, C_std
        
        else: 

            if g_fixed is None or l_fixed is None:
                raise ValueError("When sweep is False, both g_fixed and l_fixed must be provided.")
            
            C_shape = (len(delays), N_iter, nqtasks) if task_name == 'Qinp' else (len(delays), N_iter)
            C = np.full(C_shape, fill_value=np.nan)

            if not load_data:
                    
                data = EsnDynamics.esn_state_g_l(
                    N_esn, g_sweep_val=[g_fixed], l_sweep_val=[l_fixed], task_name=task_name,
                    qtasks=qtasks, axis=axis, N_iter=N_iter, seed=seed, 
                    rewrite=True, store=False, inp_type=inp_type)
                
            for it in range(N_iter):

                if load_data:

                    data = load_state_data(N_esn, g, l, task_name, it, ax_str=ax_str, inp_type=inp_type)
                    inp = data['inp']
                    states = data['states']
                        
                else:

                    sample = data[it]
                    states = sample[0]
                    inp = sample[1]

                task.x_out = states
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

                    strqtasks = task.strqtasks
                    ax_str , _ = ax_to_str(axis=axis, caxis=[])
                    pathlib.Path(f'results/data/{task_name}/{strqtasks}/ESN/{inp_type}').mkdir(parents=True, exist_ok=True)
                    fname = f'results/data/{task_name}/{strqtasks}/ESN/{inp_type}/{pm}_Nesn{N_esn}_g{g_fixed}_l{l_fixed}_ax_{ax_str}_sweep_delay'
                    
                    q_task_dict = {}
                    for i, qtask in enumerate(task.qtasks):
                        q_task_dict['C_mean '+qtask] = C_mean[:,i]
                        q_task_dict['C_std '+qtask] = C_std[:,i]                
                
                    np.savez_compressed(fname, delays=delays, **q_task_dict)
                
                else:

                    fname = f'results/data/{task_name}/ESN/{pm}_Nesn{N_esn}_g{g_fixed}_l{l_fixed}_sweep_delay'
                    np.savez_compressed(fname, delays=delays, C_mean=C_mean, C_std=C_std)

            return delays, C_mean, C_std