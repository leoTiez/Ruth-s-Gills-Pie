import os
import sys
import argparse
from pathlib import Path
import multiprocessing

os.chdir('..')


def parse_args(args):
    arg_parser = argparse.ArgumentParser('Run simulations.')
    arg_parser.add_argument('--sim_type', type=str, required=True,
                            help='Simulation to be run.')
    arg_parser.add_argument('--trial_prefix', type=str, default='',
                            help='Prefix that can be set for trial.')
    arg_parser.add_argument('--n_cpus', type=int, default=1,
                            help='Number of cpus used.')
    arg_parser.add_argument('--n_trials', type=int, default=10,
                            help='Number of trials.')
    arg_parser.add_argument('--interpreter', type=str, default='python3.9',
                            help='Python interpreter to be used.')
    return arg_parser.parse_args(args)


def main(args):
    sim_type = args.sim_type
    trial_prefix = args.trial_prefix
    n_cpus = args.n_cpus
    n_trials = args.n_trials
    python_interpreter = args.interpreter

    if sim_type.lower() == 'single_static':
        sim_file = 'examples/singleStatic.py'
        n_cells = 100
        sampling_time = 130
    elif sim_type.lower() == 'single_force':
        sim_file = 'examples/singleForce.py'
        n_cells = 20
        sampling_time = 500
    elif sim_type.lower() == 'double_force':
        sim_file = 'examples/doubleForce.py'
        n_cells = 100
        sampling_time = 500
    else:
        raise ValueError('Simulation type %s not accepted.' % sim_type)
    
    verbosity = 4
    smoothing = 100
    n_samples = 1
    uncertainty = 100

    Path('logs').mkdir(exist_ok=True, parents=True)
    with multiprocessing.Pool(processes=n_cpus) as parallel:
        for i_trial in range(n_trials):
            parallel.apply_async(os.system, args=(
                '%s gSimulation.py '
                '--simulation_file=%s '
                '--save_sim_result '
                '--save_fig '
                '--verbosity=%d '
                '--smoothing=%d '
                '--n_samples=%d ' 
                '--uncertainty=%d '
                '--sampling_time=%.2f '
                '--n_cells=%d '
                '--trial_prefix=%s%s%d '
                '--do_train=False '
                '1> logs/out_%s%s%d.out 2> logs/err_%s%s%d.err' % (
                    python_interpreter,
                    sim_file,
                    verbosity,
                    smoothing,
                    n_samples,
                    uncertainty,
                    sampling_time,
                    n_cells,
                    trial_prefix, sim_type, i_trial,
                    trial_prefix, sim_type, i_trial,
                    trial_prefix, sim_type, i_trial
                ),
            ))

        parallel.close()
        parallel.join()


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
