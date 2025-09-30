"""This file reutrns a 2D plot of the performance of the ESN as a fucntion of the hyperparameters l and g."""

import context # Sets the path correctly
from source.esn import EsnDynamics, np

if __name__ == "__main__":

    states = False
    performance = True
    load_states = False # True if esn_performance load the stored data. False if computes the data without storing esn states

    # Common 
    N_esn = 45
    task_name = "Qinp"
    pm = "Capacity"
    N_iter = 1
    seed = 39

    # If task_name == Qinp
    qtasks = ['Sigman']
    axis = ['z', 'x', 'y']

    # Sweep values
    sweep = False # If sweeping g and l (for performance)
    n_val = 5 # Number of sweeping values for parameter
    
    if task_name == "STM" or task_name == "NARMA":
        g_sweep_val = np.linspace(0.001, 1.5, num=n_val) 
        l_sweep_val = np.logspace(-5, -0.3, num=n_val)
        g_fixed = 0.9
        l_fixed = 0.0001
    if task_name == "PC": 
        g_sweep_val = np.logspace(-4, 1.7, num=n_val)
        l_sweep_val = np.linspace(0.01, 20, num=n_val)
        g_fixed = 0.01
        l_fixed = 5
    if task_name == 'Qinp':
        if any(x =='Sigmaz' or x == 'Sigman' or x == 'Fidelity' for x in qtasks):
            g_sweep_val = np.linspace(0.001, 8, num=n_val) 
            l_sweep_val = np.logspace(-6, -0.3, num=n_val)
            g_fixed = 4
            l_fixed = 1e-5
        if any(x =='Tracerho2' or x == 'Entropy' for x in qtasks):
            g_sweep_val = np.linspace(0.001, 1.5, num=n_val) 
            l_sweep_val = np.logspace(-5, -0.3, num=n_val)
            g_fixed = 0.25
            l_fixed = 1e-3

    n_min_delay = 0
    n_max_delay = 15

    store = False
    rewrite = False

    if states:

        start = context.time.time()
        EsnDynamics.esn_state_g_l(
            N_esn=N_esn, g_sweep_val=g_sweep_val, l_sweep_val=l_sweep_val,
            task_name=task_name, qtasks=qtasks, axis=axis, N_iter=N_iter, 
            seed=seed, rewrite=rewrite)
        end = context.time.time()
        print(f"Time taken for states : {end - start} seconds {(end - start) / 60} minutes, {(end - start) / 3600} hours")

    if performance:

        start = context.time.time()
        EsnDynamics.esn_performance(
            N_esn=N_esn, task_name=task_name, n_min_delay=n_min_delay, n_max_delay=n_max_delay,
            N_iter=N_iter, sweep=sweep, g_fixed=g_fixed, l_fixed=l_fixed, qtasks=qtasks, axis=axis,
            g_sweep_val=g_sweep_val, l_sweep_val= l_sweep_val,
            pm=pm, load_data=load_states, seed=seed)
        end = context.time.time()
        print(f"Time taken for performance : {end - start} seconds {(end - start) / 60} minutes, {(end - start) / 3600} hours")

    print("Finished")