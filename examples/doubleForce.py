from __future__ import  annotations
from typing import Tuple, List, Dict, Iterable, Union
import numpy as np
import pandas as pd
import torch
import pyBigWig

from src.rules import DNA_REACTANT, SPECIES_REACTANT, DNA_SPECIES_REACTANT, DEFAULT, UNSPECIFIC, rule_set_factory
from src.interactants import convert_to_dict
from src.dna import DNALayout

# Global definitions
PARTICLE_A = 'A'
PARTICLE_B = 'B'
PROTEINS = [PARTICLE_A, PARTICLE_B]

ACTIVE = 'active'
ASSOCIATED = 'associated'
SPECIES_STATES = [ASSOCIATED, ACTIVE]

PROMOTER = "promoter"
TSS = 'tss'
DNA_TRANSCRIPT = 'transcript'
TES = 'tes'
ASSOCIATED_STATE = 'associated'
DNA_SEGMENTS = [PROMOTER, TSS, DNA_TRANSCRIPT, TES]

DNA_SIZE = 1000
N_PROTEINS = 20
N_CELLS = 100

MIN_PRECISION = 1e-8
MAX_VAL_ASSO = 1.
MAX_VAL_DISSO = 5.
MAX_P_FORCE = 10.
MIN_FORCE = 50.
MAX_FORCE = 500.


def get_parameters(
        gene_num: str = '0',
        pad: str = '200',
        proteins_ratio: str = '.5',
        **kwargs
) -> Dict:

    parameter_dict = {
        'dna_size': DNA_SIZE,
        'D': torch.tensor(100.),
        'lr': torch.tensor([0., 0., 1e-5, 1e-2, 1e-3, 1e-1, 1e-2]),
        'lr_force': 1.,
        'lower_bound': torch.tensor([MIN_PRECISION, 5e-3, MIN_PRECISION, MIN_PRECISION, MIN_PRECISION, MIN_PRECISION, MIN_PRECISION]),
        'upper_bound': torch.tensor([MIN_PRECISION, 4e-3, MAX_VAL_ASSO, MAX_VAL_DISSO, MAX_VAL_ASSO, MAX_P_FORCE, MAX_VAL_DISSO]),
        'min_force': MIN_FORCE,
        'max_force': MAX_FORCE,
        'decay': 0.,
        'momentum': .5,
        'save_prefix': 'double_force_estimation',
        'probing': [(PARTICLE_A, UNSPECIFIC, DNA_SPECIES_REACTANT), (PARTICLE_B, UNSPECIFIC, DNA_SPECIES_REACTANT)],
        'colours': ['tab:orange', 'tab:green'],
        'n_proteins': {PARTICLE_A: N_PROTEINS * .4, PARTICLE_B: N_PROTEINS * .6},
        'error_weight': torch.tensor([1., 2.])
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
        time_points = torch.tensor([300, 350])
        data_description = [
            (DNA_SPECIES_REACTANT, interact_dna_species_dict[PARTICLE_A], state_dna_species_dict[UNSPECIFIC]),
            (DNA_SPECIES_REACTANT, interact_dna_species_dict[PARTICLE_B], state_dna_species_dict[UNSPECIFIC]),
        ]
        # data should be of the form time x species x position
        return time_points, data.reshape(2, 2, -1), data_description, PROTEINS
    else:
        return None, None, None, None


def get_dna(
        dna_seg_dict,
        device: Union[torch.device, int] = torch.device('cpu'),
        **kwargs
) -> DNALayout:
    # define dna specification
    dna_specs = [
        (150, 200, dna_seg_dict[PROMOTER]),
        (200, 300, dna_seg_dict[TSS]),
        (300, 700, dna_seg_dict[DNA_TRANSCRIPT]),
        (700, 800, dna_seg_dict[TES]),
    ]
    return DNALayout(DNA_SIZE, dna_specs, device=device)


def define_rules(device: Union[torch.device, int] = torch.device('cpu'), do_train: str = 'False', **kwargs) -> Tuple:
    do_train = do_train == 'True'

    # Create dictionaries that are important for simulation and parameter estimation
    interact_species_dict = convert_to_dict(PROTEINS)
    interact_dna_dict = convert_to_dict(DNA_SEGMENTS)
    state_species_dict = convert_to_dict(SPECIES_STATES)
    state_dna_dict = convert_to_dict([])

    # create dna
    dna = get_dna(
        interact_dna_dict,
        device=device,
    )

    rule_set, = rule_set_factory(
        interact_species_dict,
        interact_dna_dict,
        state_species_dict,
        state_dna_dict,
        n_rule_set=1,
        dna=dna,
        device=device
    )

    # ###
    # Define rules
    # ###
    rule_set.add_rule(
        reactants_presence=[[
            (UNSPECIFIC, DEFAULT, DNA_REACTANT),
            (UNSPECIFIC, UNSPECIFIC, SPECIES_REACTANT),
            (UNSPECIFIC, DEFAULT, DNA_SPECIES_REACTANT)
        ]],
        reactants_absence=[],
        products=[[
            (UNSPECIFIC, DEFAULT, DNA_REACTANT),
            (UNSPECIFIC, ASSOCIATED, DNA_SPECIES_REACTANT)
        ]],
        c=MIN_PRECISION
    )
    # Define random associate/dissociate of species to/from DNA
    rule_set.add_rule(
        reactants_presence=[[(UNSPECIFIC, UNSPECIFIC, DNA_SPECIES_REACTANT)]],
        reactants_absence=[],
        products=[[(UNSPECIFIC, DEFAULT, SPECIES_REACTANT)]],
        c=5e-3
    )

    # ###
    # Define particle A dynamics
    # ###
    # association
    rule_set.add_rule(
        reactants_presence=[[
            (PARTICLE_A, UNSPECIFIC, SPECIES_REACTANT),
            (PARTICLE_A, DEFAULT, DNA_SPECIES_REACTANT),
            (PROMOTER, DEFAULT, DNA_REACTANT),
        ]],
        # Only association when Particle A not already present
        reactants_absence=[[
            (PROMOTER, DEFAULT, DNA_REACTANT),
            (PARTICLE_A, [ASSOCIATED, ACTIVE], DNA_SPECIES_REACTANT)
        ]],
        products=[[
            (PROMOTER, DEFAULT, DNA_REACTANT),
            (PARTICLE_A, ACTIVE, DNA_SPECIES_REACTANT)
        ]],
        c=5e-4 if not do_train else np.exp(-20 * np.random.random()) * (MAX_VAL_ASSO - MIN_PRECISION) + MIN_PRECISION
    )

    # Rad3 Dissociation
    rule_set.add_rule(
        reactants_presence=[[
            (PROMOTER, DEFAULT, DNA_REACTANT),
            (PARTICLE_A, [ASSOCIATED, ACTIVE], DNA_SPECIES_REACTANT)
        ]],
        reactants_absence=[],
        products=[[
            (PROMOTER, DEFAULT, DNA_REACTANT),
            (PARTICLE_A, DEFAULT, SPECIES_REACTANT)
        ]],
        c=1. if not do_train else np.exp(-15 * np.random.random()) * (MAX_VAL_DISSO - MIN_PRECISION) + MIN_PRECISION
    )

    # ###
    # Define Particle B dynamics
    # ###
    # particle A association dependent on presence of particle B
    # particle A dissociates with association of particle B
    rule_set.add_rule(
        # Make Pol2 movement dependent on 2 protein positions
        reactants_presence=[
            [
                (TSS, DEFAULT, DNA_REACTANT),
                (PARTICLE_B, UNSPECIFIC, SPECIES_REACTANT),
                (PARTICLE_B, DEFAULT, DNA_SPECIES_REACTANT)
            ],
            [
                (PROMOTER, DEFAULT, DNA_REACTANT),
                (PARTICLE_A, ACTIVE, DNA_SPECIES_REACTANT)
            ]
        ],
        products=[
            [
                (TSS, DEFAULT, DNA_REACTANT),
                (PARTICLE_B, ACTIVE, DNA_SPECIES_REACTANT)
            ],
            [
                (PROMOTER, DEFAULT, DNA_REACTANT),
                (PARTICLE_A, DEFAULT, SPECIES_REACTANT)
            ]
        ],
        c=5e-2 if not do_train else np.exp(-20 * np.random.random()) * (MAX_VAL_ASSO - MIN_PRECISION) + MIN_PRECISION
    )

    # Movement of particle B
    rule_set.add_rule(
        reactants_presence=[[
            (DNA_SEGMENTS, DEFAULT, DNA_REACTANT),
            (PARTICLE_B, ACTIVE, DNA_SPECIES_REACTANT),
        ]],
        reactants_absence=[],
        products=[[
            (DNA_SEGMENTS, DEFAULT, DNA_REACTANT),
            (PARTICLE_B, ACTIVE, DNA_SPECIES_REACTANT),
        ]],
        c=2. if not do_train else np.exp(-10 * np.random.random()) * (MAX_P_FORCE - MIN_PRECISION) + MIN_PRECISION,
        force=200. if not do_train else np.random.random() * (MAX_FORCE - MIN_FORCE) + MIN_FORCE
    )

    # Dissociate particle B at TES
    rule_set.add_rule(
        reactants_presence=[[
            ([TES, DEFAULT], DEFAULT, DNA_REACTANT),
            (PARTICLE_B, ACTIVE, DNA_SPECIES_REACTANT)
        ]],
        reactants_absence=[],
        products=[[
            ([TES, DEFAULT], DEFAULT, DNA_REACTANT),
            (PARTICLE_B, DEFAULT, SPECIES_REACTANT)
        ]],
        c=2. if not do_train else np.exp(-15 * np.random.random()) * (MAX_VAL_DISSO - MIN_PRECISION) + MIN_PRECISION
    )

    return (
        interact_species_dict,
        interact_dna_dict,
        state_species_dict,
        state_dna_dict,
        rule_set,
        dna
    )
