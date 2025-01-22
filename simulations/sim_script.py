import os
import sys
import argparse
from pathlib import Path

os.chdir('..')


def parse_args(args):
    arg_parser = argparse.ArgumentParser('Run simulations.')
    arg_parser.add_argument('--sim_type', type=str, required=True,
                            help='Simulation to be run.')
    arg_parser.add_argument('--trial_prefix', type=str, default='',
                            help='Prefix that can be set for trial.')
    arg_parser.add_argument('--interpreter', type=str, default='python3.9',
                            help='Python interpreter to be used.')
    return arg_parser.parse_args(args)


def main(args):
    sim_type = args.sim_type
    trial_prefix = args.trial_prefix
    python_interpreter = args.interpreter

    if sim_type.lower() == 'single_static':
        sim_file = 'examples/singleStatic.py'
        n_cells = 100
        sampling_time = 30
    elif sim_type.lower() == 'single_force':
        sim_file = 'examples/singleForce.py'
        n_cells = 20  # make simulations quicker
        sampling_time = 300
    elif sim_type.lower() == 'double_force':
        sim_file = 'examples/doubleForce.py'
        n_cells = 20
        sampling_time = 300
    else:
        raise ValueError('Simulation type %s not accepted.' % sim_type)
    
    verbosity = 4
    smoothing = 100
    n_samples = 1
    uncertainty = 100

    Path('logs').mkdir(exist_ok=True, parents=True)
    os.system(
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
        '--trial_prefix=%s%s '
        '--do_train=False '
        '1> logs/out_%s%s.out 2> logs/err_%s%s.err' % (
            python_interpreter,
            sim_file,
            verbosity,
            smoothing,
            n_samples,
            uncertainty,
            sampling_time,
            n_cells,
            trial_prefix, sim_type,
            trial_prefix, sim_type,
            trial_prefix, sim_type
        )
    )


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
