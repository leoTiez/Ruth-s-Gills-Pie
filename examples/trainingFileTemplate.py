"""
Template file to explain the structure of a training file. This can be copy-pased and modified.
However, it's not possible to import the template file right away.
"""
from typing import Dict, Iterable, List, Tuple, Union
import torch
from src.rules import DNA_REACTANT, SPECIES_REACTANT, DNA_SPECIES_REACTANT, DEFAULT, UNSPECIFIC, rule_set_factory
from src.interactants import convert_to_dict
from src.dna import DNALayout

# ### START Remove when copy-paste to create your own file
import builtins  # Change import behaviour to make accidental import of template file impossible


def _import():
    raise ImportError('Template for training file cannot be imported')


builtins.__import__ = _import()
# ### END Remove when copy-paste to create your own file


def init_dna(
        dna_memory: torch.Tensor,
        cell_memory: torch.Tensor,
        init_data: torch.Tensor,
        interact_species_dict: Dict,
        interact_state_dict: Dict,
        dna_state_dict: Dict,
        # End mandatory values
        your_parameter: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Define and set the initial chromatin state. During training, the DNA is reset to this state after each training
    iteration
    :param dna_memory: defines chromatin state per cell in the form (cell x species x position). The DNA is at species
    position 0, whereas all other defined particles have the index equal to their id + 1
    :param cell_memory: defines the available particles in the well mixed solution
    :param init_data: the data for initialisation. This is the defined data at the specified position in the parameter
    dict (see below).
    :param interact_species_dict: Dictionary for all interacting species (e.g. proteins)
    :param interact_state_dict: Dictionary for all states of interacting species (e.g. ubiquitinated)
    :param dna_state_dict: Dictionary for all DNA states (e.g. damaged)
    :param your_parameter: Any other parameter.
    :return:
    """
    pass


def get_parameters(**kwargs) -> Dict:
    """
    Define parameters that govern training procedure as dictionary.
    :param kwargs: additional parameters passed by command line
    :return: Parameter dictionary
    """
    # Important: the values provided here are examples and should be modified for your simulation
    parameter_dict = {
        # Length of the simulated DNA. Can represent nucleotides or any arbitrary compartment
        'dna_size': 1000,
        # Noise in the positional update step when applying a force.
        'D': torch.tensor(100.),
        # Definition of sequenced properties. They must be defined in triples (name, state, type),
        # where type is referencing to whether the molecule is the DNA, a bound particle or a free particle. Free
        # particles are ignored during sequencing.
        'probing': [
            ('protein A', UNSPECIFIC, DNA_SPECIES_REACTANT),
            ('protein B', UNSPECIFIC, DNA_SPECIES_REACTANT),
            ('DNA segment such as core promoter', 'DNA state such as DMAGED', DNA_REACTANT)
        ],
        # Learning rate for sampling parameters.
        # Can be for each rule separately or a single value for all parameters.
        'lr': torch.tensor([1e-7, 1e-1, 1, 1e-2]),
        # Learning rate for force values.
        # Can be for each rule separately or a single value for all parameters.
        'lr_force': 1e8,
        # Lower bound for sampling parameters.
        # Can be for each rule separately or a single value for all parameters.
        'lower_bound': 1e-8,
        # Upper bound for sampling parameters.
        # Can be for each rule separately or a single value for all parameters.
        'upper_bound': torch.tensor([1e-3, 1., 100., 1.]),
        # Lower bound for force values.
        # Can be for each rule separately or a single value for all parameters.
        'min_force': torch.tensor([0., 1., 50., 10.]),
        # Upper bound for force values.
        # Can be for each rule separately or a single value for all parameters.
        'max_force': 500.,
        # Gradient momentum during parameter training.
        # Can be for each rule separately or a single value for all parameters.
        'momentum': [.0, .3, .0, .0],
        # Weight decay during parameter training.
        # Can be for each rule separately or a single value for all parameters.
        'decay': 0.,
        # Callback for initialising chromatin state as defined above
        'dna_init_callback': init_dna,
        # Data indices that are used for chromatin initialisation.
        'dna_init_data_idx': torch.tensor([[0, 0], [0, 1]], dtype=torch.long),  # First row time, second row species
        # save prefix that is used as an identifier for all saved files (if command line parameters are set accordingly)
        'save_prefix': 'transcription_data_estimation_easy',
        # plotting colours in the same order as the probing list
        'colours': ['tab:orange', 'tab:green', 'tab:blue'],
        # number of proteins per species
        'n_proteins': {'protein A': 400, 'protein B': 200},
    }

    return parameter_dict


def get_data(
        interact_dna_species_dict: Dict[str, int],
        state_dna_species_dict: Dict[str, int],
        state_dna_dict: Dict[str, int],
        data_paths: Iterable[str],
        **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[str, int, int]], List[str]]:
    """
    Data loading and appropriate scaling. Note that the applied smoothing is affecting the aplitude.
    Create data description of the form
    List[Tuple[
        str[data type in system i.e. DNA_SPECIES_REACTANT or DNA_REACTANT],
        str[Interactant/reactant type, e.g. protein/particle name]
        str[reactant state, e.g. default or phosphorylated etc].
    ].
    Define time points (either load from data for given time course or define time until which equilibrium state is reached.)
    Only used for simulation.
    :param interact_dna_species_dict: dictionary that maps associated particle state names to integers used in the system as ID
    :param state_dna_species_dict: dictionary that maps DNA state names to integers used in the system as ID
    :param state_dna_dict: dictionary that maps DNA state name to an id
    :param data_paths: paths to data
    :param kwargs: additional parameters passed by command line
    :return: Tuple[
        torch.Tensor[time points],
        torch.Tensor[data of dimension time x species x position],
        List[data description]
        List[data names]
    ]
    """
    pass


def get_dna(
        transcript_str_to_id: Dict[str, int],
        device: Union[torch.device, int] = torch.device('cpu'),
        **kwargs
) -> DNALayout:
    """
    Implement the DNA definition by a segmentation list of tuples of the form
    Tuple[int[segment ID], int[start], int[end]]. This function is facultative, but helps to
    structure the source code. The DNA must be defined and returned by the define_rules function
    :param transcript_str_to_id: map from segment name to int ID
    :param device: pytorch device, which can be either cpu or any specified gpu
    :param kwargs: additional parameters passed by command line
    :return: DNALayout definition
    """
    pass


def define_rules(device: Union[torch.device, int] = torch.device('cpu'), **kwargs) -> Tuple:
    """
    Define rules as a rule set. Rules are defined for interactions along and w/ DNA.
    This function creates:
    - the mapping from particle names to IDs
    - the mapping from DNA segment names to IDs
    - the mapping from associated particle states to IDs
    - the mapping from DNA states to IDs
    - the DNA
    - a list of rule sets for DNA interactions.
    See other example files for mor information
    :param device: pytorch device, which can be either cpu or any specified gpu
    :param kwargs: additional parameters passed by command line
    :return: Tuple[
        Dict[str, int]: particle names to ids,
        Dict[str, int]: DNA segment names to ids,
        Dict[str, int]: particle states to ids,
        Dict[str, int]: DNA states to ids,
        RuleSet: the defined rules and reactions
        DNALayout: the dna definition.
    ]
    """
    pass


