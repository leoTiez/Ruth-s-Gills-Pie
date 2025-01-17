import warnings
from abc import ABC, abstractmethod
from typing import List,  Type, Union, Iterable, Dict, Tuple
import numpy as np
import torch
from src.utils import check_cell_state

DNA_REACTANT = 'dna'
DNA_SPECIES_REACTANT = 'dna species'
SPECIES_REACTANT = 'species'

DEFAULT = 'default'
DEFAULT_ID = 0
UNSPECIFIC = 'unspecific'
UNSPECIFIC_ID = -1


class Interactant(ABC):
    def __init__(
            self,
            reactant: Union[int, List[int], np.ndarray, torch.Tensor],
            state: Union[int, List[int], np.ndarray, torch.Tensor],
            n_reactant: int,
            n_state: int,
            device: Union[torch.device, int] = torch.device('cpu')
    ):
        self.device = device
        # allow only 256 states and reactants per type
        if isinstance(reactant, int):
            reactant = [reactant]
        if isinstance(state, int):
            state = [state]
        if isinstance(reactant, torch.Tensor):
            self.reactant = reactant.type(torch.int8).to(self.device)
        else:
            self.reactant = torch.tensor(np.asarray(reactant), dtype=torch.int8).to(self.device)
        if isinstance(state, torch.Tensor):
            self.state = state.type(torch.int8).to(self.device)
        else:
            self.state = torch.tensor(np.asarray(state), dtype=torch.int8).to(self.device)
        self.n_reactant = n_reactant
        self.n_state = n_state

    @abstractmethod
    def get_reactant(self, replace_unspecific: bool = True, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def get_state(self, replace_unspecific: bool = True, **kwargs) -> torch.Tensor:
        pass

    def __eq__(self, other) -> bool:
        if type(other) != type(self):
            return False

        if len(self.get_reactant(replace_unspecific=True)) != len(other.get_reactant(replace_unspecific=True)):
            return False

        if len(self.get_state(replace_unspecific=True)) != len(other.get_state(replace_unspecific=True)):
            return False

        equal_reactant = torch.all(
            self.get_reactant(replace_unspecific=True) == other.get_reactant(replace_unspecific=True))
        equal_state = torch.all(self.get_state(replace_unspecific=True) == other.get_state(replace_unspecific=True))
        return bool(equal_state) and bool(equal_reactant)

    def __str__(self) -> str:
        string = '%s. ' % str(type(self)).replace('<class \'src.rules.', '').replace('\'>', '')
        string += 'Reactants: '
        if len(self.reactant.shape) == 0:
            string += str(self.reactant)
        else:
            for r in self.reactant:
                string += '%d, ' % r
        string += '\tStates: '
        if len(self.state.shape) == 0:
            string += str(self.state)
        else:
            for s in self.state:
                string += '%d, ' % s

        return string

    def __repr__(self) -> str:
        return str(self)


class DNAReactant(Interactant):
    def __init__(
            self,
            reactant: Union[int, List[int]],
            state: Union[int, List[int]],
            n_reactant: int,
            n_state: int,
            device: Union[torch.device, int] = torch.device('cpu')
    ):
        super(DNAReactant, self).__init__(reactant, state, n_reactant, n_state, device=device)

    def get_reactant(self, replace_unspecific: bool = True, return_dimension: bool = True) -> torch.Tensor:
        # In a cell state array, index 0 holds the DNA state
        if return_dimension:
            return torch.zeros(1, dtype=torch.long).to(self.device)
        if replace_unspecific:
            if torch.any(torch.isin(UNSPECIFIC_ID, self.reactant)):
                return torch.arange(self.n_reactant).to(self.device)

        return self.reactant.type(torch.long)

    def get_state(self, replace_unspecific: bool = True, **kwargs) -> torch.Tensor:
        if replace_unspecific:
            if torch.any(torch.isin(UNSPECIFIC_ID, self.state)):
                return torch.arange(self.n_state).to(self.device)

        return self.state.type(torch.long)


class DNASpeciesReactant(Interactant):
    def __init__(
            self,
            reactant: Union[int, List[int]],
            state: Union[int, List[int]],
            n_reactant: int,
            n_state: int,
            device: Union[torch.device, int] = torch.device('cpu')
    ):
        super(DNASpeciesReactant, self).__init__(reactant, state, n_reactant, n_state, device)
        if len(self.state) != 1 and torch.any(self.state == DEFAULT_ID):
            warnings.warn('DNA Species reactant states contain more than Default ID.'
                          'Default ID represent an free unbound state. This can lead to undesired behaviour')

    def get_reactant(
            self,
            replace_unspecific: bool = True,
            for_cell_state: bool = True,
            except_reactant: torch.Tensor = torch.tensor([]),
            **kwargs
    ) -> torch.Tensor:
        # Position 0 is DNA state in cell_state
        start_idx = 1 if for_cell_state else 0
        if for_cell_state:
            if torch.any(torch.isin(UNSPECIFIC_ID, self.reactant)):
                return torch.arange(start_idx, self.n_reactant + start_idx, dtype=torch.long).to(self.device)

        return self.reactant.type(torch.long) + start_idx

    def get_state(self, replace_unspecific: bool = True, **kwargs) -> torch.Tensor:
        if replace_unspecific:
            if torch.any(torch.isin(UNSPECIFIC_ID, self.state)):
                # Default state is unbound, so remove this possible state
                bound_state_mask = torch.ones(self.n_state, dtype=torch.bool)
                bound_state_mask[DEFAULT_ID] = False
                return torch.arange(self.n_state, dtype=torch.long)[bound_state_mask].to(self.device)

        return self.state.type(torch.long)


class SpeciesReactant(Interactant):
    def __init__(
            self,
            reactant: Union[int, List[int]],
            state: Union[int, List[int]],
            n_reactant: int,
            n_state: int,
            device: Union[torch.device, int] = torch.device('cpu')
    ):
        super(SpeciesReactant, self).__init__(reactant, state, n_reactant, n_state, device)

    def get_reactant(self, replace_unspecific: bool = True, **kwargs) -> torch.Tensor:
        if replace_unspecific:
            if torch.any(torch.isin(UNSPECIFIC_ID, self.reactant)):
                return torch.arange(self.n_reactant, dtype=torch.long).to(self.device)

        return self.reactant.type(torch.long)

    def get_state(self, replace_unspecific: bool = True, **kwargs) -> torch.Tensor:
        if replace_unspecific:
            if torch.any(torch.isin(UNSPECIFIC_ID, self.state)):
                return torch.arange(self.n_state, dtype=torch.long).to(self.device)

        return self.state.type(torch.long)


class InteractantList:
    def __init__(
            self,
            reactants: Union[List[List[int]], List[int]] = (),
            states: Union[List[List[int]], List[int]] = (),
            type_l: List[str] = (),
            n_species: Union[None, int] = None,
            n_interact_dna: Union[None, int] = None,
            n_state_species:  Union[None, int] = None,
            n_state_dna: Union[None, int] = None,
            inter_list: Union[None, List[Interactant]] = None,
            device: Union[torch.device, int] = torch.device('cpu')
    ):
        if len(reactants) != len(states) != len(type_l):
            raise ValueError('Reactants, states, and reactant types must have the same length')
        self.device = device

        if len(reactants) > 0:
            self.interactant_list = [
                self.create_interactant(
                    interactand_id,
                    r,
                    s,
                    n_species,
                    n_interact_dna,
                    n_state_species,
                    n_state_dna,
                    device=device
                )
                for interactand_id, r, s in zip(type_l, reactants, states)]
        else:
            self.interactant_list = []

        if inter_list is not None:
            for i in inter_list:
                self.interactant_list.append(i)

    def __str__(self) -> str:
        string = ''
        for i in self.interactant_list:
            string += '\t' + str(i)
            string += '\n'
        string += '\n'

        return string

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other) -> bool:
        if not isinstance(other, InteractantList):
            return False
        if len(other) != len(self):
            return False

        return all([x in self for x in other])

    def __iter__(self) -> Iterable:
        return iter(self.interactant_list)

    def __len__(self) -> int:
        return len(self.interactant_list)

    def __contains__(self, item: Interactant) -> bool:
        return any(map(lambda x: x == item, self.interactant_list))

    def __getitem__(self, item: int) -> Interactant:
        return self.interactant_list[item]

    def __setitem__(self, key: int, value: Interactant):
        self.interactant_list[key] = value

    def __delitem__(self, key):
        del self.interactant_list[key]

    @staticmethod
    def create_interactant(
            interactant_id: str,
            reactant: Union[int, List[int]],
            state: Union[int, List[int]],
            n_species: int,
            n_interact_dna: int,
            n_state_species: int,
            n_state_dna: int,
            device: Union[torch.device, int] = torch.device('cpu')
    ) -> Interactant:
        if interactant_id.lower() == DNA_REACTANT.lower():
            if n_interact_dna is None or n_state_dna is None:
                raise ValueError('Number of dna interactants or number dna states must not be None.')
            return DNAReactant(reactant, state, n_interact_dna, n_state_dna, device=device)
        elif interactant_id.lower() == DNA_SPECIES_REACTANT.lower():
            if n_species is None or n_state_species is None:
                raise ValueError('Number of species or number species states must not be None.')
            return DNASpeciesReactant(reactant, state, n_species, n_state_species, device=device)
        elif interactant_id.lower() == SPECIES_REACTANT.lower():
            if n_species is None or n_state_species is None:
                raise ValueError('Number of species or number species states must not be None.')
            return SpeciesReactant(reactant, state, n_species, n_state_species, device=device)
        else:
            raise ValueError('Interactant id is not accepted: %s' % interactant_id)

    def do_contain_type(self, interact_type: Type) -> bool:
        return any(map(lambda x: isinstance(x, interact_type), self.interactant_list))

    def do_contain_str_type(self, str_interact_type: str) -> bool:
        if str_interact_type.lower() == DNA_REACTANT.lower():
            interact_type = DNAReactant
        elif str_interact_type.lower() == DNA_SPECIES_REACTANT.lower():
            interact_type = DNASpeciesReactant
        elif str_interact_type.lower() == SPECIES_REACTANT.lower():
            interact_type = SpeciesReactant
        else:
            raise ValueError('Interactant id is not accepted: %s' % str_interact_type)

        return self.do_contain_type(interact_type)

    def _isin_cell_mask(
            self,
            cell_state: torch.Tensor,
            exclude_react: Union[torch.Tensor, None] = None,
            exclude_state: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        masks = []
        for react in self.interactant_list:
            r = react.get_reactant()
            if exclude_react is not None:
                r = r[~exclude_react[:, r].reshape(-1)]
            s = react.get_state()
            if exclude_state is not None:
                s = s[~exclude_state[:, s].reshape(-1)]
            if len(s) == 0 or len(r) == 0:
                masks.append(torch.ones((1, cell_state.shape[1]), dtype=torch.bool).to(self.device))
                continue
            if isinstance(react, DNASpeciesReactant):
                masks.append(torch.any(torch.isin(cell_state[:, :, r], s), dim=-1))
            elif isinstance(react, DNAReactant):
                masks.append(torch.any(torch.isin(cell_state[:, :, [0]], s), dim=-1))
        return torch.dstack(masks).to(self.device)

    def num_interactants(self, cell_state: torch.Tensor) -> torch.Tensor:
        cell_state = check_cell_state(cell_state)
        return torch.sum(self._isin_cell_mask(cell_state).type(torch.float), dim=-1)

    def isin_cell(
            self,
            cell_state: torch.Tensor,
            exclude_react: Union[torch.Tensor, None] = None,
            exclude_state: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        if len(self) == 0:
            return torch.zeros(*cell_state.shape[:2], dtype=torch.bool).to(self.device)
        cell_state = check_cell_state(cell_state)
        return torch.all(
            self._isin_cell_mask(cell_state, exclude_react, exclude_state), dim=-1
        ).reshape(cell_state.shape[0], -1)

    def get_idc(self, n_react: int, n_state: int) -> Tuple[torch.Tensor, torch.Tensor]:
        all_react, all_state = [], []
        for interactant in self:
            if isinstance(interactant, SpeciesReactant):
                continue
            react_mask = torch.zeros(n_react, dtype=torch.bool).to(self.device)
            state_mask = torch.zeros(n_state, dtype=torch.bool).to(self.device)
            react_mask[interactant.get_reactant()] = True
            state_mask[interactant.get_state()] = True
            all_react.append(react_mask)
            all_state.append(state_mask)
        return torch.stack(all_react), torch.stack(all_state)

    def fetch(
            self,
            interact_type: Union[None, Type[Interactant]] = None,
            reactant: Union[None, torch.Tensor] = None,
            state: Union[None, torch.Tensor] = None
    ) -> Iterable[Interactant]:
        query_items = self.interactant_list
        if interact_type is not None:
            query_items = filter(lambda x: type(x) == interact_type, query_items)
        if reactant is not None:
            query_items = filter(lambda x: torch.any(torch.isin(x.get_reactant(), reactant)), query_items)
        if state is not None:
            query_items = filter(lambda x: torch.any(torch.isin(x.get_state(), state)), query_items)
        return query_items


def convert_to_dict(names: List[str]) -> Dict[str, int]:
    dictionary = {name: num + 1 for num, name in enumerate(names)}
    dictionary[UNSPECIFIC] = -1
    dictionary[DEFAULT] = 0
    return dict(sorted(dictionary.items(), key=lambda item: item[1]))