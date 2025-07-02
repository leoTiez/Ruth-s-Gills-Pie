from __future__ import annotations
from typing import Tuple, List, Dict, Union
import torch
import numpy as np
from src.rules import DNA_REACTANT, SPECIES_REACTANT, DNA_SPECIES_REACTANT, DEFAULT, UNSPECIFIC, rule_set_factory
from src.interactants import convert_to_dict
from src.dna import DNALayout


# Global definitions
EX_PROTEIN = 'particle'
PROTEINS = [EX_PROTEIN]
TSS = 'tss'
DNA_TRANSCRIPT = 'transcript'
TES = 'tes'
ASSOCIATED_STATE = 'associated'

DNA_SIZE = 1000
N_PROTEINS = 10
N_CELLS = 100

MIN_PRECISION = 1e-8
MAX_VAL_DISSO = 5.
MAX_VAL_ASSO = 1.
MIN_P_FORCE = 1e-8
MAX_P_FORCE = 5.
MIN_FORCE = 1.
MAX_FORCE = 200.


def get_parameters(**kwargs) -> Dict:
    parameter_dict = {
        'dna_size': DNA_SIZE,
        'D': torch.tensor(100.),
        'n_proteins': {EX_PROTEIN: N_PROTEINS},
        'lr': torch.tensor([0., 0., 1e-6, 1e-1, 1e-1]),  # no training of random association and dissociation
        'lower_bound': torch.tensor([MIN_PRECISION, 5e-3, MIN_PRECISION, MIN_P_FORCE, MIN_PRECISION]),
        'upper_bound': torch.tensor([MIN_PRECISION, 5e-3, MAX_VAL_ASSO, MAX_P_FORCE, MAX_VAL_DISSO]),
        'lr_force': 5e5,
        'min_force': MIN_FORCE,
        'max_force': MAX_FORCE,
        'noise_theta': .0,
        'noise_force': .0,
        'momentum': .5,
        'decay': 0.,
        'max_grad_ratio': .5,
        'save_prefix': 'force_data_estimation',
        'probing': [(EX_PROTEIN, UNSPECIFIC, DNA_SPECIES_REACTANT)],
        'colours': ['tab:blue'],
        'error_weight': torch.tensor([1.])
    }
    return parameter_dict


def get_data(
        interact_dna_species_dict: Dict[str, int],
        state_dna_species_dict: Dict[str, int],
        state_dna_dict: Dict[str, int],
        do_train: str = 'False',
        data_path: str = 'data/simulated-data/force_data_estimation.tsv',
        **kwargs
) -> Tuple[torch.Tensor | None, torch.Tensor | None, List[Tuple[str, int, int]] | None, List[str] | None]:
    do_train = do_train == 'True'
    if do_train:
        data = torch.tensor(np.loadtxt(data_path, delimiter='\t'))
        data = torch.stack([data, data])
        time_points = torch.tensor([500, 600])
        data_description = [
            (DNA_SPECIES_REACTANT, interact_dna_species_dict[EX_PROTEIN], state_dna_species_dict[UNSPECIFIC]),
        ]
        data_names = [EX_PROTEIN]
        # data should be of the form time x species x position
        return time_points, data.reshape(2, 1, -1), data_description, data_names
    else:
        return None, None, None, None


def get_dna(
        dna_seg_dict,
        device: Union[torch.device, int] = torch.device('cpu'),
        **kwargs
) -> DNALayout:
    # define dna specification
    dna_specs = [
        (100, 200, dna_seg_dict[TSS]),
        (200, 600, dna_seg_dict[DNA_TRANSCRIPT]),
        (600, 700, dna_seg_dict[TES]),
    ]
    return DNALayout(DNA_SIZE, dna_specs, device=device)


def define_rules(device: Union[torch.device, int] = torch.device('cpu'), do_train: str = 'False', **kwargs) -> Tuple:
    do_train = do_train == 'True'
    dna_states = []
    dna_segments = [TSS, DNA_TRANSCRIPT, TES]
    species_dna_states = [ASSOCIATED_STATE]

    # Define particle names to integer IDs that are used by the system
    interact_species_dict = convert_to_dict(PROTEINS)
    interact_dna_dict = convert_to_dict(dna_segments)
    state_species_dna_dict = convert_to_dict(species_dna_states)
    state_dna_dict = convert_to_dict(dna_states)

    # create dna
    dna = get_dna(
        interact_dna_dict,
        device=device,
    )
    # create rule sets
    rule_set, = rule_set_factory(
        interact_species_dict,
        interact_dna_dict,
        state_species_dna_dict,
        state_dna_dict,
        n_rule_set=1,
        dna=dna,
        device=device
    )

    # The c values correspond to the true values with which the data were created. It will be overwritten w/ the
    # estimated parameters and have no influence on the parameter fitting.
    # Random association and dissociation to and from the everywhere on the DNA for all proteins
    rule_set.add_rule(
        reactants_presence=[[
            (UNSPECIFIC, UNSPECIFIC, SPECIES_REACTANT),
            (UNSPECIFIC, DEFAULT, DNA_SPECIES_REACTANT),
            (UNSPECIFIC, DEFAULT, DNA_REACTANT)
        ]],
        reactants_absence=[],
        products=[[
            (UNSPECIFIC, ASSOCIATED_STATE, DNA_SPECIES_REACTANT),
            (UNSPECIFIC, DEFAULT, DNA_REACTANT)
        ]],
        c=MIN_PRECISION
    )
    rule_set.add_rule(
        reactants_presence=[[(UNSPECIFIC, ASSOCIATED_STATE, DNA_SPECIES_REACTANT)]],
        reactants_absence=[],
        products=[[(UNSPECIFIC, DEFAULT, SPECIES_REACTANT)]],
        c=5e-3
    )

    # Example protein behaviour. Different interactions along the transcript.
    # Association
    rule_set.add_rule(
        reactants_presence=[[
            (EX_PROTEIN, UNSPECIFIC, SPECIES_REACTANT),
            (EX_PROTEIN, DEFAULT, DNA_SPECIES_REACTANT),
            (TSS, DEFAULT, DNA_REACTANT)
        ]],
        reactants_absence=[],
        products=[[
            (EX_PROTEIN, ASSOCIATED_STATE, DNA_SPECIES_REACTANT),
            (TSS, DEFAULT, DNA_REACTANT)
        ]],
        c=5e-4 if not do_train else np.minimum(
            np.exp(-35. * np.random.random()) * (MAX_VAL_ASSO - MIN_PRECISION) + MIN_PRECISION,
            1e-4
        )
    )
    # Force
    rule_set.add_rule(
        reactants_presence=[[
            (EX_PROTEIN, ASSOCIATED_STATE, DNA_SPECIES_REACTANT),
            ([TSS, DNA_TRANSCRIPT, TES], DEFAULT, DNA_REACTANT)
        ]],
        reactants_absence=[],
        products=[[
            (EX_PROTEIN, ASSOCIATED_STATE, DNA_SPECIES_REACTANT),
            ([TSS, DNA_TRANSCRIPT, TES], DEFAULT, DNA_REACTANT)
        ]],
        # c=.5 if not do_train else np.exp(-15 * np.random.random()) * (MAX_P_FORCE - MIN_P_FORCE) + MIN_P_FORCE,
        c=.5 if not do_train else MIN_PRECISION,
        force=100. if not do_train else np.random.random() * (MAX_FORCE - MIN_FORCE) + MIN_FORCE
    )
    # Dissociation
    rule_set.add_rule(
        reactants_presence=[[(EX_PROTEIN, ASSOCIATED_STATE, DNA_SPECIES_REACTANT), (TES, DEFAULT, DNA_REACTANT)]],
        reactants_absence=[],
        products=[[(EX_PROTEIN, DEFAULT, SPECIES_REACTANT), (TES, DEFAULT, DNA_REACTANT)]],
        c=1. if not do_train else np.exp(-15 * np.random.random()) * (MAX_VAL_DISSO - MIN_PRECISION) + MIN_PRECISION,
    )

    return (
        interact_species_dict,
        interact_dna_dict,
        state_species_dna_dict,
        state_dna_dict,
        rule_set,
        dna
    )

