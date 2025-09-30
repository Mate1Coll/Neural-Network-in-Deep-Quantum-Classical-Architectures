
import context
from source.quantum_reservoir import QuantumReservoirDynamics, np
import sys
import yaml
import argparse
import copy

params_to_section = {
    "L": "qrc_params",
    "dt": "qrc_params",
    "Dmp": "qrc_params",
    "Vmp": "qrc_params",
    "pm": "perf_params",
    "qtasks": "qrc_params",
    "fixed_h": "qrc_params"
}

def run_config(config, sweep_values=None):

    qrc_params = config["qrc_params"]
    perf_params = config["perf_params"]
    flags = config["flags"]
    obs = flags["obs"]; performance = flags["performance"]; load_obs = flags["load_obs"]
    store = flags["store"]; rewrite = flags["rewrite"]

    if obs:

        # Computes the observables dynamics
        start = context.time.time()
        QuantumReservoirDynamics.qrc_obs(
            **qrc_params, rewrite=rewrite, store=store)
        end = context.time.time()
        print(f"Time taken for observables : {end - start} seconds {(end - start) / 60} minutes, {(end - start) / 3600} hours")
    
    if performance:
    
        if qrc_params["sweep_param"] == 'W':
            fixed_param = qrc_params["fixed_h"]
        elif qrc_params["sweep_param"] == 'h':
            fixed_param = qrc_params["fixed_W"]
        else:
            fixed_param = None
        
        # Performance of QRC
        start = context.time.time()
        QuantumReservoirDynamics.qrc_performance(
            **qrc_params, **perf_params, store=store, load_obs=load_obs, sweep_values=sweep_values,
            fixed_param=fixed_param)
        end = context.time.time()
        print(f"Time taken for performance : {end - start} seconds {(end - start) / 60} minutes, {(end - start) / 3600} hours")

    print('Finished')

    return

def print_config(config):

    print(config["qrc_params"])
    print(config["perf_params"])

    return

if __name__ == "__main__":

    # Open data from yaml file
    with open('configs/config_qrc.yaml', 'r') as f:
        config = yaml.safe_load(f)

        # Parse CLI overrides
    parser = argparse.ArgumentParser() # Create parser instance
    parser.add_argument("-s", "--sweep", type=str, choices=['h', 'W'], default=None,
                        help= 'Determines which type of parameter sweeping we apply. \n' 
                        '- h: sweep h and fix W. \n' 
                        '- W: sweep W and fix h. \n') 
    parser.add_argument("-qt" , "--qtasks", type=str, nargs='+', default=['Tracerho2', 'Entropy', 'Sigman', 'Entanglement', 'Fidelity'],
                        help='Determine which quantum tasks to consider. \n'
                        'If none, considers all the tasks \n '
                        'Example: -qt Tracerho2 Entropy')
    parser.add_argument("-qs", "--qinpsweep", action="store_true",
                        help= "On/off flag to compute Qinp performance for all axis.") 
    parser.add_argument("-dr", "--dry-run", action="store_true",
                        help="On/off flag to practise a particular performance (configuration in yaml file)")
    parser.add_argument("-Vmp", "--Vmp", type=int, default=None,
                        help=" Overwrites the value of Vmp.")
    parser.add_argument("-Dmp", "--Dmp", type=int, default=None,
                        help=" Overwrites the value of Dmp.")
    parser.add_argument("-dt", "--dt", type=float, default=None,
                        help=" Overwrites the value of dt only if there is sweep.")
    parser.add_argument("-ax", "--axis", type=str, default=None, choices=['z','x','y','xy','zxy'],
                        help= 'Overwrites combination of parameters of a given axis (iterable slurm).')
    
    args = parser.parse_args() # Runs the parser and places the extracted data

    if not args.dry_run:

        if config["qrc_params"]["task_name"] != "Qinp":

            local_config = copy.deepcopy(config)

            if args.sweep:

                if args.dt:
                    local_config["qrc_params"]["dt"] = args.dt

                sweep_dict = local_config["sweeps"][f'{args.sweep}']
                print(f'Sweep {args.sweep}: ', sweep_dict, ' ...')

                local_config["qrc_params"]["sweep_param"] = sweep_dict["sweep_param"]
                sweep_values = np.logspace(
                    sweep_dict["start_exp"],
                    sweep_dict["end_exp"],
                    num=sweep_dict["num"]
                )

                print_config(local_config)
                run_config(copy.deepcopy(local_config), sweep_values=sweep_values)

            if args.Vmp:

                local_config["qrc_params"]["Vmp"] = args.Vmp
                print_config(local_config)
                run_config(copy.deepcopy(local_config))

            if args.Dmp:

                local_config["qrc_params"]["Vmp"] = args.Vmp
                print_config(local_config)
                run_config(copy.deepcopy(local_config))

        else:

            for qtasks in args.qtasks:

                local_config = copy.deepcopy(config)
                print(qtasks)
                for k, v in local_config["all_qtasks"][qtasks].items():
                    section = params_to_section.get(k)
                    local_config[section][k] = v

                if args.qinpsweep:
                    sweep_dict = local_config["sweeps"][f'sweep{args.sweep}']
                    keys = list(sweep_dict.keys())
                    values = list(sweep_dict.values())
                    if args.sweeptype == 'zip':
                        combinations = zip(*values)

                    for values in combinations:
                        combo_dict = dict(zip(keys,values))
                        for k, v in combo_dict.items():
                            section = params_to_section.get(k)
                            local_config[section][k] = v

                        print_config(local_config)
                        run_config(copy.deepcopy(local_config))


                if args.axis:

                    local_config["perf_params"]["axis"] = [ax for ax in args.axis]
                    local_config["perf_params"]["caxis"] = [ax for ax in args.axis]
                    print_config(local_config)
                    run_config(copy.deepcopy(local_config))

                else:

                    print("Running with default configuration (optimized parameters for quantum task)")
                    print_config(local_config)
                    run_config(copy.deepcopy(local_config))

    else:
        
        print("Running with full default configuration (testing)")
        print_config(config)
        run_config(copy.deepcopy(config))