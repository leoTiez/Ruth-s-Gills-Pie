#!/usr/bin/env python3
import sys
import os.path
import warnings

import torch
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from src.interactants import InteractantList, DNASpeciesReactant, UNSPECIFIC, DNAReactant, DNA_REACTANT, DNA_SPECIES_REACTANT
from src.utils import load_unknown_cmd_params
from src.gillespie import Gillespy

# ###########################################################
# When using mac, set agg background globally and comment out
# ###########################################################
# matplotlib.use("Agg")


def parse_args(args):
    arg_parser = argparse.ArgumentParser('Find reaction parameters for the GillesPy simulation.'
                                         'Set additional arguments outside of the training file.')

    arg_parser.add_argument('--training_file', type=str, required=True,
                            help='Path to simulation file that defines the parameters and rules. Has the same'
                                 'layout as the training file. Therefore, even unnecessary parameters and return values'
                                 'must be defined. See template file for more information.')
    arg_parser.add_argument('--save_fig', action='store_true', dest='save_fig',
                            help='If set, figures and animations are saved to file.')
    arg_parser.add_argument('--verbosity', type=int, default=2,
                            help='Set the verbosity level. The higher the more output is produced.')
    arg_parser.add_argument('--n_cells', type=int, default=20,
                            help='Set number of simulated cells')
    arg_parser.add_argument('--n_epoch', type=int, default=100,
                            help='Number of epochs')
    arg_parser.add_argument('--smoothing', type=int, default=200,
                            help='Size of smoothing window that is applied during plotting.')
    arg_parser.add_argument('--uncertainty', type=int, default=100,
                            help='Uncertainty that is added to the sampled position to simulate cell culture')
    arg_parser.add_argument('--n_samples', type=int, default=None,
                            help='Number of cells that are sampled for updating cell culture state.')
    arg_parser.add_argument('--tol', type=float, default=1e-15,
                            help='Error tolerance.')
    arg_parser.add_argument('--use_parameter_stop', action='store_true', dest='use_parameter_stop',
                            help='If set, use the convergence criterion --tol parameter update difference.')
    arg_parser.add_argument('--sampling_boost', type=float, default=.1,
                            help='If there are not enough update steps for each time point in data, '
                                 'increase all parameters by a ratio of sampling_boost * parameter.')
    arg_parser.add_argument('--save_params', action='store_true', dest='save_params',
                            help='If set, parameters are saved to file.')
    arg_parser.add_argument('--save_error', action='store_true', dest='save_error',
                            help='If set, save estimation error to file.')
    arg_parser.add_argument('--seq_momentum', type=float, default=.9,
                            help='Momentum for plotted profiles and error computation')
    arg_parser.add_argument('--ntorch_threads', type=int, default=1,
                            help='Number of threads that are used by pytorch. If you are running several estimations '
                                 'in parallel, set this parameter to 1 to avoid processes blocking each other.')
    arg_parser.add_argument('--use_gpu', action='store_true', dest='use_gpu',
                            help='If set, use gpu')
    arg_parser.add_argument('--trial_prefix', type=str, default='',
                            help='Set save prefix for particular trial.')
    parsed_params, unknown_params = arg_parser.parse_known_args(args)

    # Additional keywords are parsed as string and need to be converted separately by the user in the
    # simulation file
    kw_args = load_unknown_cmd_params(unknown_params)
    return parsed_params, kw_args


def main(argv):
    args, kw_args = parse_args(argv)
    sys.path.append(os.path.abspath('/'.join(args.training_file.split('/')[:-1])))
    train_module = __import__(args.training_file.split('/')[-1].replace('.py', ''))
    torch.set_num_threads(args.ntorch_threads)
    n_cells = args.n_cells
    save_fig = args.save_fig
    verbosity = args.verbosity
    smoothing = args.smoothing
    n_samples = args.n_samples
    uncertainty = args.uncertainty
    n_epoch = args.n_epoch
    tol = args.tol
    sampling_boost = args.sampling_boost
    use_parameter_stop = args.use_parameter_stop
    save_params = args.save_params
    save_error = args.save_error
    seq_momentum = args.seq_momentum
    use_gpu = args.use_gpu
    trial_prefix = args.trial_prefix

    device = torch.device('cpu')
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            warnings.warn('No GPU available. Use CPU instead.')
            use_gpu = False

    param_dict = train_module.get_parameters(**kw_args)
    param_dict['save_prefix'] = trial_prefix + param_dict['save_prefix']
    (
        interact_species_dict,
        interact_dna_dict,
        state_species_dict,
        state_dna_dict,
        rule_set,
        dna
    ) = train_module.define_rules(device, **kw_args)

    tp, data, data_description, data_names = train_module.get_data(
        interact_species_dict,
        state_species_dict,
        state_dna_dict,
        **kw_args
    )
    # plot only first time point
    for d, c, n in zip(data[0], param_dict['colours'], data_names):
        plt.plot(d, color=c, label=n)
    plt.xlabel('Position')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Data layout')
    if save_fig:
        Path('figures/data').mkdir(parents=True, exist_ok=True)
        plt.savefig('figures/data/%s_data.png' % param_dict['save_prefix'])
        plt.close()
    else:
        plt.show()

    proteins = torch.zeros(rule_set.n_species)
    for prot_name, value in param_dict['n_proteins'].items():
        proteins[interact_species_dict[prot_name]] = value

    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Initialise gillespy
    gillespy_estimator = Gillespy(
        data=data,
        rules=rule_set,
        D=param_dict['D'],
        proteins=proteins,
        n_cells=n_cells,
        dna=dna,
        uncertainty=uncertainty,
        seq_bias=smoothing,
        device=device
    )

    inter_list = []
    inter_names = []
    for probe in param_dict['probing']:
        if probe[-1] == DNA_SPECIES_REACTANT:
            inter_names.append('%s:%s' % (probe[0], probe[1]))
            probe_type = DNASpeciesReactant
            react = interact_species_dict[probe[0]]
            state = state_species_dict[probe[1]]
            n_reactant = len(interact_species_dict) - 1  # Reduce by unspecific
            n_state = len(state_species_dict) - 1  # Reduce by unspecific
        elif probe[-1] == DNA_REACTANT:
            inter_names.append('dna:%s' % probe[1])
            probe_type = DNAReactant
            react = interact_dna_dict[UNSPECIFIC]
            state = state_dna_dict[probe[1]]
            n_reactant = len(interact_dna_dict) - 1,  # Reduce by unspecific
            n_state = len(state_dna_dict) - 1  # Reduce by unspecific
        else:
            continue
        inter_list.append(probe_type(
            reactant=react,
            state=state,
            n_reactant=n_reactant,
            n_state=n_state
        ))
    seq_probing = InteractantList(inter_list=inter_list, device=device)
    dna_init_callback = param_dict['dna_init_callback'] if 'dna_init_callback' in param_dict else None
    dna_init_data_idc = param_dict['dna_init_data_idx'] if 'dna_init_data_idx' in param_dict else None
    error_weight = param_dict['error_weight'] if 'error_weight' in param_dict else None
    print('Start training procedure')
    gillespy_estimator.train(
        probing=seq_probing,
        seq_tp=tp if isinstance(tp, torch.Tensor) else torch.tensor(tp),
        lr=param_dict['lr'],
        lr_force=param_dict['lr_force'],
        decay=param_dict['decay'],
        grad_momentum=param_dict['momentum'],
        n_samples=n_samples,
        max_iter=n_epoch,
        tol=tol,
        use_parameter_stop=use_parameter_stop,
        lower_bound=param_dict['lower_bound'],
        upper_bound=param_dict['upper_bound'],
        min_force=param_dict['min_force'],
        max_force=param_dict['max_force'],
        error_weight=error_weight,
        sampling_boost=sampling_boost,
        dna_init_callback=dna_init_callback,
        dna_init_data_idc=dna_init_data_idc,
        colors=param_dict['colours'],
        save_fig=save_fig,
        save_prefix=param_dict['save_prefix'],
        save_params=save_params,
        save_error=save_error,
        seq_momentum=seq_momentum,
        verbosity=verbosity,
    )


if __name__ == '__main__':
    main(sys.argv[1:])


