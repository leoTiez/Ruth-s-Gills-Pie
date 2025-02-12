import os
import sys
import argparse
import multiprocessing
from pathlib import Path
import numpy as np
from subprocess import check_output

os.chdir('..')


def parse_args(args):
    arg_parser = argparse.ArgumentParser('Estimate data profiles.')
    arg_parser.add_argument('--train_type', type=str, required=True,
                            help='Training setup to be used.')
    arg_parser.add_argument('--trial_prefix', type=str, default='',
                            help='Prefix that can be set for trial.')
    arg_parser.add_argument('--n_cpus', type=int, default=1,
                            help='Number of cpus used.')
    arg_parser.add_argument('--n_trials', type=int, default=100,
                            help='Number of trials.')
    arg_parser.add_argument('--interpreter', type=str, default='python3.9',
                            help='Python interpreter to use')
    return arg_parser.parse_args(args)


def main(args):
    train_type = args.train_type
    trial_prefix = args.trial_prefix
    n_cpus = args.n_cpus
    n_trials = args.n_trials
    python_interpreter = args.interpreter

    if train_type.lower() == 'single_static':
        train_file = 'examples/singleStatic.py'
        data_path = 'data/simulated-data/single_staticnoforce_data_estimation.tsv'
        tol = 0.003
        sampling_boost = .1
        n_cells = 100
        n_epoch = 500
    elif train_type.lower() == 'single_force':
        train_file = 'examples/singleForce.py'
        data_path = 'data/simulated-data/single_forceforce_data_estimation.tsv'
        tol = 0.003
        n_cells = 20
        sampling_boost = .1
        n_epoch = 7500
    elif train_type.lower() == 'double_force':
        train_file = 'examples/doubleForce.py'
        data_path = 'data/simulated-data/double_forcedouble_force_estimation.tsv'
        tol = 0.0005
        n_cells = 20
        sampling_boost = .01
        n_epoch = 7500
    else:
        raise ValueError('Training type %s not accepted.' % train_type)

    verbosity = 4
    smoothing = 100
    n_samples = 1
    uncertainty = 100
    seq_momentum = .0

    Path('logs/').mkdir(exist_ok=True, parents=True)
    with multiprocessing.Pool(processes=n_cpus) as parallel:
        for i_trial in np.random.permutation(np.arange(n_trials)):
            parallel.apply_async(os.system, args=(
                '%s gTraining.py '
                '--training_file=%s '
                '--save_fig '
                '--verbosity=%d '
                '--n_cells=%d '
                '--n_epoch=%d '
                '--smoothing=%d '
                '--uncertainty=%d '
                '--n_samples=%d '
                '--tol=%.10f '
                '--sampling_boost=%.3f '
                '--save_params '
                '--save_error '
                '--seq_momentum=%.4f '
                '--trial_prefix=%s%s%s '
                '--do_train=True '
                '--data_path=%s '
                '--save_error 1> logs/out_%s%s%s.out 2> logs/err_%s%s%s.err' % (
                    python_interpreter,
                    train_file,
                    verbosity,
                    n_cells,
                    n_epoch,
                    smoothing,
                    uncertainty,
                    n_samples,
                    tol,
                    sampling_boost,
                    seq_momentum,
                    train_type, trial_prefix, i_trial,
                    data_path,
                    train_type, trial_prefix, i_trial,
                    train_type, trial_prefix, i_trial
                ),
            ))

        parallel.close()
        parallel.join()


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
