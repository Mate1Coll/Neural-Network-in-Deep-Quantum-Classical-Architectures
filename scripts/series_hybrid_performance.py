""" This file computes the series hybrid performance for n_delays """

import context
from source.hybrid import HybridDynamics
from source.quantum_reservoir import QuantumReservoirDynamics
import sys
import yaml
from itertools import product
import argparse
import copy

params_to_section = {
    "L": "qrc_params",
    "dt": "qrc_params",
    "N_esn": "esn_params",
    "g": "esn_params",
    "l": "esn_params",
    "axis": "perf_params",
    "caxis": "perf_params",
    "ratio_delay_qrc": "perf_params",
    "pm": "perf_params",
    "qtasks": "common_params",
    "N_meas": "perf_params"
}

def run_config(config):

    qrc_params = config["qrc_params"]
    common_params = config["common_params"]
    esn_params = config["esn_params"]
    perf_params = config["perf_params"]
    
    flags = config["flags"]
    obs = flags["obs"]; rewrite = flags["rewrite"]; performance = flags["performance"]
    load_obs = flags["load_obs"]

    if obs:

        start = context.time.time()
        QuantumReservoirDynamics.qrc_obs(
            **qrc_params, **common_params, sweep_param=None,
            rewrite=rewrite
        )
        end = context.time.time()
        print(f"Time taken for states : {end - start} seconds {(end - start) / 60} minutes, {(end - start) / 3600} hours")

    if performance:

        qrc_params['h'] = qrc_params['fixed_h']
        qrc_params['W'] = qrc_params['fixed_W']

        start = context.time.time()
        HybridDynamics.serie_hybrid_performance(
            **qrc_params, **esn_params, **common_params, **perf_params,
            load_obs=load_obs,
        )
        end = context.time.time()
        print(f"Time taken for performance : {end - start} seconds {(end - start) / 60} minutes, {(end - start) / 3600} hours")

    print("Finished")

    return

def print_config(config):

    print(config["qrc_params"])
    print(config["esn_params"])
    print(config["common_params"])
    print(config["perf_params"])

    return

if __name__ == "__main__":

    printed = None

    # Open data from yaml file
    with open('configs/config_hybrid.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Parse CLI overrides
    parser = argparse.ArgumentParser() # Create parser instance
    parser.add_argument("-s", "--sweep", type=int, choices=[1, 2, 3], default=None,
                        help= 'Determines which type of parameter sweeping we apply. \n' 
                        '- 1: ratio qrc delay. \n' 
                        '- 2: axis in which observables are made. \n' 
                        '- 3: number of computational nodes for each reservoir (quantum and classical).')
    parser.add_argument("-st", "--sweeptype", type=str, choices=['product', 'zip'], default=None,
                        help='Define which type of combinatory iterator is applied when the number of sweeping parameter is > 1: \n'
                        '- product: cartesian product (x,y) for x in A and for b in B. \n'
                        '- zip: match parameters one to one (first with first, etc.)')
    parser.add_argument("-qt" , "--qtasks", type=str, nargs='+', default=['Tracerho2', 'Entropy', 'Sigman', 'Entanglement', 'Fidelity'],
                        help='Determine which quantum tasks to consider. \n'
                        'If none, considers all the tasks \n '
                        'Example: -qt Tracerho2 Entropy')
    parser.add_argument("-dr", "--dry-run", action="store_true",
                        help="On/off flag to practise a particular performance (configuration in yaml file)")
    
    parser.add_argument("-qrc", "--ratio-qrc-delay", type=float, choices=[0.0, 0.5, 1.0], default=None,
                        help='Determines ratio delay of qrc for performance task.')
    
    parser.add_argument("-N", "--N-esn", type=int, default=None,
                        help=" Overwrites the value of N_esn.")
    
    parser.add_argument("-L", "--L", type=int, default=None,
                        help=" Overwrites the value of L.")
    
    parser.add_argument("-ax", "--axis", type=str, default=None, choices=['z','x','y','xy','zxy'],
                        help= 'Overwrites combination of parameters of a given axis (iterable slurm).')
    
    parser.add_argument("-Nm", "--N-meas", type=int, default=None,
                        help=" Overwrites the value of N_meas.")
    
    args = parser.parse_args() # Runs the parser and places the extracted data

    if not args.dry_run:

        for qtasks in args.qtasks:

            local_config = copy.deepcopy(config)
            print(qtasks)
            for k, v in local_config["all_qtasks"][qtasks].items():
                section = params_to_section.get(k)
                local_config[section][k] = v

            if args.sweep:

                sweep_dict = local_config["sweeps"][f'sweep{args.sweep}']
                print(f'Sweep {args.sweep}: ', sweep_dict, ' ...')

                if len(sweep_dict) > 1:
                    keys = list(sweep_dict.keys())
                    values = list(sweep_dict.values())
                    if args.sweeptype == 'zip':
                        combinations = zip(*values)
                    elif args.sweeptype == 'product':
                        combinations = product(*values)

                    for values in combinations:
                        combo_dict = dict(zip(keys,values))
                        for k, v in combo_dict.items():
                            section = params_to_section.get(k)
                            local_config[section][k] = v

                        print_config(local_config)
                        run_config(copy.deepcopy(local_config))

                else:

                    key = next(iter(sweep_dict))
                    section = params_to_section.get(key)
                    for v in sweep_dict[key]:
                        local_config[section][key] = v
                        print_config(local_config)
                        run_config(copy.deepcopy(local_config))
                
                printed = True

            if args.ratio_qrc_delay is not None:
                
                print(args.ratio_qrc_delay)
                local_config["perf_params"]["ratio_delay_qrc"] = float(args.ratio_qrc_delay)
                print_config(local_config)
                run_config(copy.deepcopy(local_config))
                printed = True

            if args.N_esn and args.L:

                local_config["esn_params"]["N_esn"] = args.N_esn
                local_config["qrc_params"]["L"] = args.L
                print_config(local_config)
                run_config(copy.deepcopy(local_config))
                printed = True


            if args.axis:

                dict_slurm = local_config["sweepslurm"][args.axis]
                local_config["perf_params"]["axis"] = dict_slurm["axis"]
                local_config["perf_params"]["caxis"] = dict_slurm["axis"]
                local_config["esn_params"]["N_esn_solely"] = dict_slurm["N_esn_solely"]
                print_config(local_config)
                run_config(copy.deepcopy(local_config))
                printed = True
            
            if args.N_meas:
                print(args.N_meas)
                local_config["perf_params"]["N_meas"] = int(args.N_meas)
                print_config(local_config)
                run_config(copy.deepcopy(local_config))
                printed = True

            if not printed:
                print("Running with default configuration (optimized parameters for quantum task)")
                print_config(local_config)
                run_config(copy.deepcopy(local_config))

    else:

        print("Running with full default configuration (testing)")
        run_config(copy.deepcopy(config))