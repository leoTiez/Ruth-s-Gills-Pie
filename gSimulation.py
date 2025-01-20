#!/usr/bin/env python3
import os.path
import sys
import warnings

import torch
import argparse
import matplotlib

from src.gillespie import Gillespy
from src.interactants import UNSPECIFIC, InteractantList, DNASpeciesReactant, DNAReactant, DNA_SPECIES_REACTANT, DNA_REACTANT
from src.utils import load_unknown_cmd_params

# ###########################################################
# When using mac, set agg background globally and comment out
# ###########################################################
# matplotlib.use("Agg")


def parse_args(args):
    arg_parser = argparse.ArgumentParser('Run the gGillesPy simulation. '
                                         'Set additional arguments outside of the simulation file.')
    arg_parser.add_argument('--simulation_file', type=str, required=True,
                            help='Path to simulation file that defines the parameters and rules. Has the same'
                                 'layout as the training file. Therefore, even unnecessary parameters and return values'
                                 'must be defined. See template file for more information.')
    arg_parser.add_argument('--save_sim_result', action='store_true', dest='save_sim_result',
                            help='If set, simulation results w/ trained parameters are saved to a csv file.')
    arg_parser.add_argument('--save_fig', action='store_true', dest='save_fig',
                            help='If set, figures and animations are saved to file.')
    arg_parser.add_argument('--verbosity', type=int, default=2,
                            help='Set the verbosity level. The higher the more output is produced.')
    arg_parser.add_argument('--plot_frequency', type=float, default=.1,
                            help='When saving a gif animation, this parameter sets when a frame is added to '
                                 'the gif queue. The parameter represents that each plot_frequency minute, a frame'
                                 'is saved.')
    arg_parser.add_argument('--smoothing', type=int, default=100,
                            help='Size of smoothing window that is applied during plotting.')
    arg_parser.add_argument('--n_samples', type=int, default=None,
                            help='Number of cells that are sampled for updating cell culture state.')
    arg_parser.add_argument('--uncertainty', type=int, default=200,
                            help='Uncertainty that is added to the sampled position to simulate cell culture')
    arg_parser.add_argument('--sampling_time', type=float, default=5.,
                            help='Simulated time.')
    arg_parser.add_argument('--n_cells', type=int, default=20,
                            help='Number of cells that is kept track of')
    arg_parser.add_argument('--ntorch_threads', type=int, default=1,
                            help='Number of threads that are used by pytorch. If you are running several estimations '
                                 'in parallel, set this parameter to 1 to avoid processes blocking each other.')
    arg_parser.add_argument('--use_gpu', action='store_true', dest='use_gpu',
                            help='If set, use gpu')
    arg_parser.add_argument('--trial_prefix', type=str, default='',
                            help='Set save prefix for particular trial.')
    parsed_params, unknown_params = arg_parser.parse_known_args(args)

    # Additional key words are parsed as string and need to be converted separately by the user in the
    # simulation file
    kw_args = load_unknown_cmd_params(unknown_params)
    return parsed_params, kw_args


def main(argv):
    args, kw_args = parse_args(argv)
    sys.path.append(os.path.abspath('/'.join(args.simulation_file.split('/')[:-1])))
    sim_module = __import__(args.simulation_file.split('/')[-1].replace('.py', ''))
    torch.set_num_threads(args.ntorch_threads)
    save_simulation_result = args.save_sim_result
    save_fig = args.save_fig
    verbosity = args.verbosity
    plot_frequency = args.plot_frequency
    smoothing = args.smoothing
    n_samples = args.n_samples
    uncertainty = args.uncertainty
    sampling_time = args.sampling_time
    n_cells = args.n_cells
    use_gpu = args.use_gpu
    trial_prefix = args.trial_prefix

    device = torch.device('cpu')
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            warnings.warn('No GPU available. Use CPU instead.')
            use_gpu = False

    param_dict = sim_module.get_parameters(**kw_args)
    param_dict['save_prefix'] = trial_prefix + param_dict['save_prefix']

    (
        interact_species_dict,
        interact_dna_dict,
        state_species_dict,
        state_dna_dict,
        rule_set,
        dna
    ) = sim_module.define_rules(device, **kw_args)

    tp, data, data_description, data_names = sim_module.get_data(
        interact_species_dict,
        state_species_dict,
        state_dna_dict,
        **kw_args
    )
    proteins = torch.zeros(rule_set.n_species)
    for prot_name, value in param_dict['n_proteins'].items():
        proteins[interact_species_dict[prot_name]] = value

    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    gillespy_simulator = Gillespy(
        data=data,
        D=param_dict['D'],
        rules=rule_set,
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
            n_reactant = len(interact_species_dict) - 1 # Reduce by unspecific
            n_state = len(state_species_dict) - 1 # Reduce by unspecific
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

    seq_probing = InteractantList(inter_list=inter_list)
    dna_init_callback = param_dict['dna_init_callback'] if 'dna_init_callback' in param_dict else None
    dna_init_data_idc = param_dict['dna_init_data_idx'] if 'dna_init_data_idx' in param_dict else None
    gillespy_simulator.run(
        sampling_time,
        probing=seq_probing,
        labels=inter_names,
        n_samples=n_samples,
        dna_init_callback=dna_init_callback,
        dna_init_data_idc=dna_init_data_idc,
        colors=param_dict['colours'],
        verbosity=verbosity,
        save_simulation_result=save_simulation_result,
        save_fig=save_fig,
        save_prefix=param_dict['save_prefix'],
        plot_frequency=plot_frequency
    )


if __name__ == '__main__':
    main(sys.argv[1:])


