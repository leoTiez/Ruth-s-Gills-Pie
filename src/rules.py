import warnings
from typing import List, Iterable, Dict, Tuple, Type, Callable, Union
from decimal import Decimal
import numpy as np
import torch
import networkx as nx

from src.dna import DNALayout
from src.interactants import InteractantList, Interactant, DNAReactant, SpeciesReactant, DNASpeciesReactant, SPECIES_REACTANT, DNA_REACTANT, DNA_SPECIES_REACTANT, UNSPECIFIC, UNSPECIFIC_ID, DEFAULT, DEFAULT_ID
from src.errors import InternalException, ValueNotInitialized, PossibleInternalException
from src.utils import check_cell_state


class Rule:
    def __init__(
            self,
            dna: DNALayout,
            reactants_presence: Union[InteractantList, List[InteractantList]],
            reactants_absence: Union[InteractantList, List[InteractantList]],
            products: Union[InteractantList, List[InteractantList]],
            n_species: int,
            n_state_species: int,
            c: float,
            force: float = 0.,
            device: Union[torch.device, int] = torch.device('cpu')
    ):
        self.device = device
        self.reactants_presence = reactants_presence if isinstance(reactants_presence, list) else [reactants_presence]
        self.reactants_absence = reactants_absence if isinstance(reactants_absence, list) else [reactants_absence]
        self.products = products if isinstance(products, list) else [products]

        if len(self.reactants_presence) != len(self.products):
            raise ValueError('Each AND connection (sublists) in present reactants '
                             'must have a corresponding list in the product.'
                             'Then length of both lists is expected to be the same.')
        if len(self.reactants_presence) == 0:
            raise ValueError('Present reactants for a reaction rule must be non-empty')
        if len(self.products) == 0:
            raise ValueError('Products for a reaction rule must be non-empty')

        self.dna_mask_presence = self._create_dna_mask(dna, self.reactants_presence, self.device)
        self.dna_mask_absence = self._create_dna_mask(dna, self.reactants_absence, self.device)
        self.dna_mask_product = self._create_dna_mask(dna, self.products, self.device)
        self.protein_mask_presence = self._create_protein_masks(
            self.reactants_presence,
            n_species,
            n_state_species,
            self.device
        )
        self.protein_mask_absence = self._create_protein_masks(
            self.reactants_absence,
            n_species,
            n_state_species,
            self.device
        )
        self.protein_mask_product = self._create_protein_masks(
            self.products,
            n_species,
            n_state_species,
            self.device
        )

        self.c = torch.tensor(c, dtype=torch.double).to(self.device)
        self.force = torch.tensor(force, dtype=torch.double).to(self.device)

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        is_react_pres_eq = all([x in self.reactants_presence for x in other.reactants_presence])
        is_react_abs_eq = all([x in self.reactants_absence for x in other.reactants_absence])
        is_prod_eq = all([x in self.products for x in other.products])
        is_c_eq = torch.all(self.c == other.c)
        is_force_eq = torch.all(self.force == other.force)

        return is_react_pres_eq and is_react_abs_eq and is_prod_eq and is_c_eq and is_force_eq

    def __str__(self) -> str:
        string = ''
        string += 'Reactants that must be present:\n' + '\n'.join([str(r_p) for r_p in self.reactants_presence])
        string += '\nReactants that must be absent:\n' + '\n'.join([str(r_a) for r_a in self.reactants_absence])
        string += '\nProducts:\n' + '\n'.join([str(p) for p in self.products])
        string += '\nProbability: %.2E' % Decimal(float(self.c))
        string += '\nForce:\n\t' + '\n\t'.join(['%.3f' % dm for dm in self.force])

        return string

    def __repr__(self) -> str:
        return str(self)

    def update_dna_mask(self, dna: DNALayout):
        self.dna_mask_presence = self._create_dna_mask(dna, self.reactants_presence, self.device)
        self.dna_mask_absence = self._create_dna_mask(dna, self.reactants_absence, self.device)
        self.dna_mask_product = self._create_dna_mask(dna, self.products, self.device)

    @staticmethod
    def _create_dna_mask(
            dna: DNALayout,
            interactant_list: Union[InteractantList, List[InteractantList]],
            device: Union[torch.device, int] = torch.device('cpu')
    ) -> torch.Tensor:
        if isinstance(interactant_list, InteractantList):
            interactant_list = [interactant_list]
        dna_mask = torch.ones((len(interactant_list), dna.size), dtype=torch.bool).to(device)
        if len(interactant_list) == 0:
            return torch.zeros_like(dna_mask).to(device)

        for out_num, il in enumerate(interactant_list):
            if len(il) == 0:
                return torch.zeros_like(dna_mask).to(device)

            for pr in il:
                if isinstance(pr, DNAReactant):
                    dna_mask[out_num, :] = torch.logical_and(
                        dna_mask[out_num, :],
                        dna.get_mask(pr.get_reactant(return_dimension=False))
                    )

        return dna_mask

    @staticmethod
    def _create_protein_masks(
            interactant_list: Union[InteractantList, List[InteractantList]],
            n_species: int,
            n_states: int,
            device: Union[torch.device, int] = torch.device('cpu')
    ) -> torch.Tensor:
        if isinstance(interactant_list, InteractantList):
            interactant_list = [interactant_list]

        if len(interactant_list) == 0:
            return torch.zeros((n_species, n_states), dtype=torch.bool).to(device)
        protein_mask = []
        for out_num, il in enumerate(interactant_list):
            if len(il) == 0:
                return torch.zeros((1, n_species, n_states), dtype=torch.bool).to(device)
            free_mask = torch.zeros((n_species, n_states), dtype=torch.bool).to(device)
            for pr in il:
                if isinstance(pr, SpeciesReactant):
                    # for each reactant all states
                    free_mask[pr.get_reactant().reshape(-1, 1), pr.get_state()] = True
            protein_mask.append(free_mask)
        return torch.stack(protein_mask).to(device)

    def get_presence_information(self) -> Tuple[List[InteractantList], torch.Tensor, torch.Tensor]:
        return self.reactants_presence, self.dna_mask_presence, self.protein_mask_presence

    def get_absence_information(self) -> Tuple[List[InteractantList], torch.Tensor, torch.Tensor]:
        return self.reactants_absence, self.dna_mask_absence, self.protein_mask_absence

    def get_product_information(self) -> Tuple[List[InteractantList], torch.Tensor, torch.Tensor]:
        return self.products, self.dna_mask_product, self.protein_mask_product

    def does_interact(
            self,
            cell_state: torch.Tensor,
            reactants: InteractantList,
            n_max_states: int,
            other_c: float
    ) -> torch.Tensor:
        cell_state = check_cell_state(cell_state)
        react_mask, state_mask = reactants.get_idc(cell_state.shape[-1], n_max_states)
        interact_mask = torch.ones(cell_state.shape[1], dtype=torch.bool).to(self.device)
        did_update = False
        for dna_mask, present_list in zip(self.dna_mask_presence, self.reactants_presence):
            # Can rules interact?
            if any([r in present_list for r in reactants if isinstance(r, DNASpeciesReactant)]):
                did_update = True
                # Where could this possibly happen, presuming interactants where already present?
                interact_mask = torch.logical_and(
                    interact_mask,
                    torch.logical_and(
                        dna_mask.reshape(1, -1),
                        present_list.isin_cell(cell_state, react_mask, state_mask)
                    )
                )
                # Is that likely to happen wrt the other rule? Use random sampling here
                interact_mask = torch.logical_and(
                    interact_mask,
                    torch.rand(cell_state.shape[1]).to(self.device) < self.c / (self.c + other_c)
                )
        return interact_mask if did_update else torch.zeros_like(interact_mask, dtype=torch.bool).to(self.device)

    def apply_react_mask(self, cell_state: torch.Tensor, i_mask: int) -> torch.Tensor:
        cell_state = check_cell_state(cell_state)
        mask = torch.zeros(*cell_state.shape[:2], dtype=torch.bool).to(self.device)
        blocking_mask = torch.zeros(cell_state.shape[0], dtype=torch.bool).to(self.device)
        for absence_mask, aqs in zip(self.dna_mask_absence, self.reactants_absence):
            blocking_mask = torch.logical_or(
                blocking_mask,
                torch.any(
                    torch.logical_and(absence_mask.reshape(1, -1), aqs.isin_cell(cell_state)),
                    dim=-1
                )
            )

        mask[~blocking_mask] = torch.logical_and(
            self.dna_mask_presence[i_mask].reshape(1, -1),
            self.reactants_presence[i_mask].isin_cell(cell_state)
        )[~blocking_mask]
        return mask

    def apply_product_mask(self, cell_state: torch.Tensor, i_mask: int) -> torch.Tensor:
        cell_state = check_cell_state(cell_state)
        return torch.logical_and(
            self.dna_mask_product[i_mask].reshape(1, -1),
            torch.ones((cell_state.shape[0], 1), dtype=torch.bool).to(self.device)
        )

    def is_present_reactant(
            self,
            reactant: Interactant,
            return_array: bool = False,
    ) -> Union[bool, torch.Tensor]:
        is_present = [reactant in r_p for r_p in self.reactants_presence]
        if return_array:
            return torch.tensor(np.asarray(is_present), dtype=torch.bool).to(self.device)
        else:
            return any(is_present)

    def is_present_type(
            self,
            interact_type: Type,
            return_array: bool = False
    ) -> Union[bool, torch.Tensor]:
        is_present = [r_p.do_contain_type(interact_type) for r_p in self.reactants_presence]
        if return_array:
            return torch.tensor(np.asarray(is_present), dtype=torch.bool).to(self.device)
        else:
            return any(is_present)

    def is_present_str_type(
            self,
            str_interact_type: str,
            return_array: bool = False
    ) -> Union[bool, torch.Tensor]:
        is_present = [r_p.do_contain_str_type(str_interact_type) for r_p in self.reactants_presence]
        if return_array:
            return torch.tensor(np.asarray(is_present), dtype=torch.bool).to(self.device)
        else:
            return any(is_present)

    def is_absent_reactant(
            self,
            reactant: Interactant,
            return_array: bool = False
    ) -> Union[bool, torch.Tensor]:
        is_absent = [reactant in r_a for r_a in self.reactants_absence]
        if return_array:
            return torch.tensor(np.asarray(is_absent), dtype=torch.bool).to(self.device)
        else:
            return any(is_absent)

    def is_absent_type(
            self,
            interact_type: Type,
            return_array: bool = False
    ) -> Union[bool, torch.Tensor]:
        is_absent = [r_a.do_contain_type(interact_type) for r_a in self.reactants_absence]
        if return_array:
            return torch.tensor(np.asarray(is_absent), dtype=torch.bool).to(self.device)
        else:
            return any(is_absent)

    def is_absent_str_type(
            self,
            str_interact_type: str,
            return_array: bool = False
    ) -> Union[bool, torch.Tensor]:
        is_absent = [r_a.do_contain_str_type(str_interact_type) for r_a in self.reactants_absence]
        if return_array:
            return torch.tensor(np.asarray(is_absent), dtype=torch.bool).to(self.device)
        else:
            return any(is_absent)

    def is_product(
            self,
            product: Interactant,
            return_array: bool = False
    ) -> Union[bool, torch.Tensor]:
        is_product = [product in p for p in self.products]
        if return_array:
            return torch.tensor(np.asarray(is_product), dtype=torch.bool).to(self.device)
        else:
            return any(is_product)

    def is_product_type(
            self,
            interact_type: Type,
            return_array: bool = False
    ) -> Union[bool, torch.Tensor]:
        is_product = [p.do_contain_type(interact_type) for p in self.products]
        if return_array:
            return torch.tensor(np.asarray(is_product), dtype=torch.bool).to(self.device)
        else:
            return any(is_product)

    def is_product_str_type(
            self,
            str_interact_type: str,
            return_array: bool = False
    ) -> Union[bool, torch.Tensor]:
        is_product = [p.do_contain_str_type(str_interact_type) for p in self.products]
        if return_array:
            return torch.tensor(np.asarray(is_product), dtype=torch.bool).to(self.device)
        return any(is_product)


class RuleSet:
    def __init__(
            self,
            interact_species: Dict[str, int],
            interact_dna: Dict[str, int],
            state_species_dict: Dict[str, int],
            state_dna_dict: Dict[str, int],
            dna: [DNALayout, None] = None,
            rule_list: Union[List[Rule], None] = None,
            no_force_no_noise: bool = True,
            device: Union[torch.device, int] = torch.device('cpu')
    ):
        def reduce(definition_dict: Dict[str, int], keyword: str = UNSPECIFIC) -> int:
            return sum([keyword in definition_dict.keys()])
        self.device = device
        self.interact_species = interact_species
        self.interact_dna = interact_dna
        self.state_species = state_species_dict
        self.state_dna = state_dna_dict
        self.dna = dna
        if rule_list is not None:
            self.rule_list = rule_list
        else:
            self.rule_list = []
        self.no_force_no_noise = no_force_no_noise

        self.num_rules = len(self.rule_list)

        # Do not count unspecific as a separate state or interactant
        self.n_species = len(self.interact_species) - reduce(self.interact_species)
        self.n_interact_dna = len(self.interact_dna) - reduce(self.interact_dna)
        self.n_state_species = len(self.state_species) - reduce(self.state_species)
        self.n_state_dna = len(self.state_dna) - reduce(self.state_dna)
        self.n_max_states = int(torch.maximum(torch.tensor(self.n_state_species), torch.tensor(self.n_state_dna)))

    def __iter__(self) -> Iterable:
        return iter(self.rule_list)

    def __len__(self) -> int:
        return len(self.rule_list)

    def __getitem__(self, item: int) -> Rule:
        return self.rule_list[item]

    def __setitem__(self, key: int, value: Rule):
        self.rule_list[key] = value

    def __delitem__(self, key):
        del self.rule_list[key]

    def __call__(
            self,
            rule_num: int,
            cell_state: torch.Tensor,
            free_proteins: torch.Tensor,
            tau: float,
            D: torch.Tensor = torch.tensor(1.),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cell_state = check_cell_state(cell_state)
        if cell_state.shape[0] != 1:
            raise ValueError('Rule update can only be performed for single cell. cell_state is in incorrect shape.')
        if len(free_proteins.shape) == 3:
            if free_proteins.shape[0] == 0:
                free_proteins = free_proteins.reshape(*free_proteins.shape[1:])
            else:
                raise ValueError('Rule update can only be performed for single cell. '
                                 'free_proteins is in incorrect shape.')
        if len(free_proteins.shape) != 2:
            raise ValueError('Rule update cannot be performed. free_proteins must be of shape species x states.')

        if self.dna.updated_dna:
            for rule in self.rule_list:
                rule.update_dna_mask(self.dna)
            self.dna.unset_updated_dna()

        cell_state, free_proteins, all_sampled_reactants, sampled_info_react = self._update_reactants(
            self.rule_list[rule_num],
            cell_state,
            free_proteins
        )

        cell_state, free_proteins, sampled_info_prod = self._update_products(
            self.rule_list[rule_num],
            cell_state,
            free_proteins,
            all_sampled_reactants,
            sampled_info_react[:, 0].type(torch.int),
            tau,
            D
        )
        # Probability of sampling starting position
        sampled_info_prod[:, -1] *= sampled_info_react[:, -1]
        return cell_state, free_proteins, sampled_info_react, sampled_info_prod

    def _update_reactants(
            self,
            rule: Rule,
            cell_state: torch.Tensor,
            free_proteins: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[InteractantList], torch.Tensor]:
        all_sampled_reactants = []
        sampled = []
        for num, pres_react in enumerate(rule.reactants_presence):
            sampled_reactants = []
            # Initiate mask
            mask = rule.apply_react_mask(cell_state, num)
            if not torch.any(mask):
                raise InternalException('Reactants cannot be updated anywhere along the DNA')

            # Sample position
            pos = torch.multinomial(mask.type(torch.float), num_samples=1, replacement=False)
            react_prob = 1. / torch.sum(mask.type(torch.float))
            # remove reactants and save to list
            for p_r in pres_react:
                reacts = p_r.get_reactant(replace_unspecific=True, return_dimension=False).to(self.device)
                states = p_r.get_state(replace_unspecific=True).to(self.device)
                # Have different behaviours for different types of species
                # Case 1: Species reactant
                if isinstance(p_r, SpeciesReactant):
                    if torch.any(free_proteins[reacts.reshape(-1, 1), states] > 0):
                        involved_proteins = free_proteins[reacts.reshape(-1, 1), states].reshape(-1)
                        # Sample a protein type and species according to protein distribution
                        total_proteins = torch.sum(involved_proteins)
                        tmp_idx = torch.searchsorted(
                            torch.cumsum(involved_proteins, dim=0) / total_proteins,
                            torch.rand(1).to(self.device)
                        )[0]
                        react_prob *= involved_proteins[tmp_idx] / total_proteins
                        sample_react_idx = tmp_idx // states.shape[0]
                        sample_state_idx = tmp_idx % states.shape[0]
                        free_proteins[reacts[sample_react_idx], states[sample_state_idx]] -= 1

                        sampled_reactants.append(SpeciesReactant(
                            reacts[sample_react_idx],
                            states[sample_state_idx],
                            self.n_species,
                            self.n_state_species,
                            device=self.device
                        ))
                    else:
                        raise InternalException('No free proteins available during update reactants')

                # Case 2: DNA reactant
                elif isinstance(p_r, DNAReactant):
                    # If sampled DNA position is not of a required type, there is an error in the sampling
                    if torch.any(torch.isin(self.dna[pos], reacts)):
                        # Add first DNA species to list before resetting it
                        sampled_reactants.append(DNAReactant(
                            self.dna[pos.reshape(-1)],
                            cell_state[0, pos.reshape(-1), 0],  # Knowing that there is only one cell
                            self.n_interact_dna,
                            self.n_state_dna,
                            device=self.device
                        ))
                        cell_state[:, pos, 0] = DEFAULT_ID
                        # No update of probability cos cell state and react are defined by position
                    else:
                        raise InternalException('Reactants cannot be updated anywhere along the DNA')

                # Case 3: Species bound to DNA
                # States of bound species in cell state come after index 0
                elif isinstance(p_r, DNASpeciesReactant):
                    involved_reactants = torch.where(torch.isin(cell_state[:, pos, reacts].reshape(-1), states))[0]
                    if len(involved_reactants) == 0:
                        raise InternalException('Species not present at sampled DNA position')
                    react_prob *= 1. / float(len(involved_reactants))
                    # sample reactant
                    interactant_idx = involved_reactants[torch.randint(len(involved_reactants), size=(1,))]
                    sampled_react = reacts[interactant_idx]

                    # If DEFAULT_STATE, then the requirement is that position needs to be empty.
                    # If fulfilled, no need to update. Interactant did not participate in interaction, as
                    # position is anyways
                    if cell_state[:, pos, sampled_react] == DEFAULT_ID:
                        continue
                    sampled_reactants.append(DNASpeciesReactant(
                        sampled_react - 1,  # Correct for index bias
                        cell_state[:, pos, sampled_react],
                        self.n_species,
                        self.n_state_species,
                        device=self.device
                    ))
                    cell_state[:, pos, sampled_react] = 0

            sampled.append(torch.tensor([float(pos), react_prob]).to(self.device))
            all_sampled_reactants.append(InteractantList(inter_list=sampled_reactants, device=self.device))
        return cell_state, free_proteins, all_sampled_reactants, torch.stack(sampled).type(torch.double).to(self.device)

    def _update_products(
            self,
            rule: Rule,
            cell_state: torch.Tensor,
            free_proteins: torch.Tensor,
            sampled_reaction_l: List[InteractantList],
            pos_l: torch.Tensor,
            tau: float,
            D: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled_info = []
        for num, (pos, product) in enumerate(zip(pos_l, rule.products)):
            # do not set probability of sampling position as this is given and set when probabilities are returned
            prob_prod = 1.
            # Add states to cell state or free proteins
            dna_update = []
            sampled = []
            for i_p, p in enumerate(product):
                reacts = p.get_reactant(replace_unspecific=True, return_dimension=False).to(self.device)
                states = p.get_state(replace_unspecific=True).to(self.device)
                unspecified_reactant = len(reacts) > 1
                unspecified_state = len(states) > 1
                # Retrieve the reactant and state update
                if unspecified_reactant or unspecified_state:
                    interactants = sampled_reaction_l[num]
                    if len(interactants) == 0:
                        sampled_react = reacts[torch.randint(len(reacts), size=(1,))]
                        sampled_state = reacts[torch.randint(len(states), size=(1,))]
                        prob_prod *= 1. / float(len(reacts))
                        prob_prod *= 1. / float(len(states))
                    # If there is more than one possible solution, use interactant at same index
                    else:
                        sampled_react = interactants[i_p].get_reactant().to(self.device) if unspecified_reactant else reacts
                        sampled_state = interactants[i_p].get_state().to(self.device) if unspecified_state else states
                        # Correct index
                        if type(p) != type(interactants[i_p]):
                            if isinstance(p, DNAReactant) or isinstance(interactants[i_p], DNAReactant):
                                raise ValueError('Mismatch to deduce product reactant type and state in rule %s. \n '
                                                 'Type at same index for SpeciesReactant or DNASpeciesReactant must '
                                                 'not be DNAReactant.' % rule)
                            if isinstance(p, DNASpeciesReactant):
                                sampled_react += 1
                            else:
                                sampled_react -= 1
                        # No update of probability as this is defined by reaction probability
                else:
                    sampled_react = reacts
                    sampled_state = states
                    prob_prod *= 1. / float(len(reacts))
                    prob_prod *= 1. / float(len(states))

                # Case 1: Update of species reactant
                if isinstance(p, SpeciesReactant):
                    free_proteins[sampled_react, sampled_state] += 1
                elif isinstance(p, DNASpeciesReactant):
                    dna_update.append(torch.tensor([sampled_react, sampled_state]))
                    sampled.append(DNASpeciesReactant(
                        sampled_react - 1,
                        sampled_state,
                        self.n_species,
                        self.n_state_species,
                        device=self.device
                    ))
                elif isinstance(p, DNAReactant):
                    dna_update.append(torch.tensor([0, sampled_state]))
                    # Do not append to sample, as DNA does not interact w/ other DNA

            pos_update = pos
            # Determine position
            if len(dna_update) > 0:
                update_idx = torch.stack(dna_update)
                sampled = InteractantList(inter_list=sampled, device=self.device)
                # prevent over overshooting range by large tau
                if torch.sign(rule.force) != torch.sign(torch.ceil(rule.force * tau)):
                    pos_update = cell_state.shape[0]
                else:
                    if rule.force != 0 or not self.no_force_no_noise:
                        dpos = torch.round(torch.normal(
                            rule.force * tau,
                            torch.sqrt(2 * D * tau))
                        ).type(torch.long)
                    else:
                        dpos = 0
                    pos_update = pos + dpos
                    if dpos != 0 and len(sampled) > 0:
                        start = torch.minimum(pos, pos_update)
                        end = torch.maximum(pos, pos_update)
                        move_mask = torch.zeros((1, cell_state.shape[1]), dtype=torch.bool).to(self.device)
                        move_mask[:, start:end] = True
                        for interact_rule in self:
                            if interact_rule != rule:
                                interact_mask = torch.logical_and(
                                    interact_rule.does_interact(cell_state, sampled, self.n_max_states, rule.c),
                                    move_mask
                                )
                                interact_pos = torch.where(interact_mask)[1]
                                if len(interact_pos) > 0:
                                    if dpos > 0 and pos_update > torch.min(interact_pos):
                                        pos_update = torch.min(interact_pos)
                                    elif dpos < 0 and pos_update < torch.max(interact_pos):
                                        pos_update = torch.max(interact_pos)

                # If updated position is outside simulated DNA, remove proteins and add to free pool
                # shape[0] = n_cells, which is for the rule update always 1
                if not 0 <= pos_update < cell_state.shape[1]:
                    # Put into default state as there is no information to what state it should change to
                    # Correct for different index position
                    free_proteins[update_idx[update_idx[:, 0] != 0] - 1, DEFAULT_ID] += 1
                    pos_update = pos
                else:
                    free_mask = (cell_state[:, pos_update, update_idx[:, 0]] == 0).reshape(-1)
                    cell_state[:, pos_update, update_idx[free_mask, 0]] = update_idx[free_mask, 1].type(torch.int8).to(self.device)
                    for i_prod in torch.where(~free_mask)[0]:
                        alt_pos = torch.where(cell_state[:, :, update_idx[i_prod, 0]] == 0)[1]
                        cell_state[
                            :,
                            alt_pos[torch.argmin(torch.abs(alt_pos - pos_update))],
                            update_idx[i_prod, 0]
                        ] = update_idx[i_prod, 1].type(torch.int8).to(self.device)

            sampled_info.append(torch.tensor([float(pos_update), prob_prod]))
        return cell_state, free_proteins, torch.stack(sampled_info).type(torch.double).to(self.device)

    def _create_parameter_l(
            self,
            interact_l: List[Tuple[Union[str, List[str]], Union[str, List[str]], str]]
    ) -> List[Tuple[
        Union[int, List[int]],
        Union[int, List[int]],
        str
    ]]:
        parameter_l = []
        for inter, s, interactant_id in interact_l:
            if interactant_id.lower() == SPECIES_REACTANT:
                interact_dict = self.interact_species
                state_dict = self.state_species
            elif interactant_id.lower() == DNA_SPECIES_REACTANT:
                interact_dict = self.interact_species
                state_dict = self.state_species
            elif interactant_id.lower() == DNA_REACTANT:
                interact_dict = self.interact_dna
                state_dict = self.state_dna
            else:
                raise ValueError('Interactant id not accepted: %s' % interactant_id)
            parameter_l.append((
                [interact_dict[i_inter] for i_inter in inter] if isinstance(inter, list) else interact_dict[inter],
                [state_dict[s_i] for s_i in s] if isinstance(s, list) else state_dict[s],
                interactant_id
            ))

        return parameter_l

    def _get_rules(self, fun: Callable, return_bool: bool = False, *args) -> Iterable:
        if return_bool:
            return map(lambda x: fun(x, *args), self.rule_list)
        else:
            return filter(lambda x: fun(x, *args), self.rule_list)

    @staticmethod
    def x_plus_distribution(
            x_plus: torch.Tensor,
            x_minus: torch.Tensor,
            force: torch.Tensor,
            tau: torch.Tensor,
            D: torch.Tensor
    ) -> torch.Tensor:
        smoluchowski = torch.exp(- (x_plus - x_minus - force * tau) ** 2 / (4 * D * tau))
        return smoluchowski / torch.sqrt(4 * torch.pi * D * tau)

    def determine_backproperties(
            self,
            cell_state: torch.Tensor,
            tau: torch.Tensor,
            D: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cell_state = check_cell_state(cell_state)
        backprop_factors = torch.zeros((
            cell_state.shape[0],
            len(self.rule_list),
            cell_state.shape[1],
            self.n_species + 1,
            self.n_max_states
        ), dtype=torch.double).to(self.device)
        backprop_force = torch.zeros((
            cell_state.shape[0],
            len(self.rule_list),
            cell_state.shape[1],
            self.n_species + 1,
            self.n_max_states
        ), dtype=torch.double).to(self.device)
        # Dimension: cell x rule
        # Take mean as the Gillespie approach assumes that no reaction has happened until tau
        x_plus = torch.arange(cell_state.shape[1]).reshape(-1, 1).to(self.device)
        for i_rule, rule in enumerate(self.rule_list):
            for i_mask in range(len(rule.reactants_presence)):
                rule_react = rule.apply_react_mask(cell_state, i_mask)
                if not torch.any(rule_react):
                    continue
                rule_product = rule.apply_product_mask(cell_state, i_mask)
                rule_product[~torch.any(rule_react, dim=1)] = False
                react_cell, react_pos = torch.where(rule_react)
                # Set indicator / impact function
                react_all = []
                state_all = []
                pos_all = []
                n_interactants_all = []
                for r in rule.reactants_presence[i_mask]:
                    if isinstance(r, SpeciesReactant):
                        continue
                    pos_idx, react_idx = torch.where(torch.isin(cell_state[
                                                       react_cell.reshape(-1, 1),
                                                       react_pos.reshape(-1, 1), r.get_reactant()
                                                   ], r.get_state()))

                    react = r.get_reactant()[react_idx]
                    state = cell_state[react_cell[pos_idx], react_pos[pos_idx], react].type(torch.long)
                    n_interactants = torch.sum(
                        torch.isin(cell_state[rule_react][:, r.get_reactant()], r.get_state()).type(torch.float),
                        dim=1
                    )
                    backprop_factors[
                        react_cell[pos_idx],
                        [i_rule],
                        react_pos[pos_idx],
                        react,
                        state
                    ] -= 1. / n_interactants[pos_idx]
                    pos_all.append(pos_idx)
                    react_all.append(react)
                    state_all.append(state)
                    n_interactants_all.append(n_interactants)
                for i_p, p in enumerate(rule.products[i_mask]):
                    if isinstance(p, SpeciesReactant):
                        continue
                    pos_idx = pos_all[i_p]
                    if len(p.get_reactant()) > 1:
                        react = react_all[i_p]
                    else:
                        react = torch.ones_like(pos_idx) * p.get_reactant()
                    if len(p.get_state()) > 1:
                        state = state_all[i_p]
                        if len(torch.unique(react)) == 1:
                            react = torch.ones_like(pos_idx) * p.get_reactant()
                    else:
                        state = torch.ones_like(pos_idx) * p.get_state()
                    if rule.force != 0. or not self.no_force_no_noise:
                        # determine protein movements
                        for i_cell in torch.unique(react_cell[pos_idx]):
                            x_minus = react_pos[pos_idx][react_cell[pos_idx] == i_cell]
                            n_interactants = n_interactants_all[i_p][react_cell[pos_idx] == i_cell]
                            # do not consider movements of DNA
                            cell_react = torch.unique(react[pos_idx][react_cell[pos_idx] == i_cell])
                            cell_react_state = torch.unique(state[pos_idx][react_cell[pos_idx] == i_cell])
                            x_plus_dist = self.x_plus_distribution(
                                x_plus,
                                x_minus.reshape(1, -1),
                                rule.force,
                                tau,
                                D
                            )
                            prod_impact = torch.sum(x_plus_dist / n_interactants.reshape(1, -1), dim=1)
                            backprop_factors[[i_cell], [i_rule], :, cell_react.reshape(
                                -1, 1), cell_react_state] += prod_impact / float(
                                (len(cell_react) * len(cell_react_state)))
                            force_impact = torch.sum(
                                x_plus_dist * (x_plus - x_minus.reshape(1, -1) - rule.force * tau)
                                / (n_interactants.reshape(1, -1) * (2 * D)), dim=1
                            )
                            backprop_force[[i_cell], [i_rule], :, cell_react.reshape(
                                -1, 1), cell_react_state] += force_impact / float(
                                (len(cell_react) * len(cell_react_state)))
                    else:
                        backprop_factors[
                            react_cell[pos_idx],
                            [i_rule],
                            react_pos[pos_idx],
                            react,
                            state
                        ] += 1. / n_interactants_all[i_p][pos_idx]

                if torch.any(torch.abs(backprop_factors) > 1.):
                    warnings.warn('The absolute value of the backproperties indicator function should '
                                  'be below 1. However, this can occur when the sampled tau is very '
                                  'small. This is commonly considered too low and might cause problems. '
                                  'Consider to reduce the sampling parameters. '
                                  'However, this could also indicate other problems. '
                                  'Check that there is only one DNA segment per sub-rule. '
                                  'Make sure your rule set is coherent. You can visualise them using '
                                  'the provided ruleGraph module.')
                divisor = torch.sum(rule_react.type(torch.double), dim=1).reshape(-1, 1, 1, 1)
                divisor = torch.where(divisor == 0., 1., divisor)
                backprop_factors[:, i_rule] /= divisor
                backprop_force[:, i_rule] /= divisor
        # has form cells x rules x pos x react x state
        return backprop_factors, backprop_force

    def create_graph(self) -> nx.DiGraph:
        dependency_graph = nx.DiGraph()
        for u, r in enumerate(self):
            exist_force = torch.any(r.force != 0.)
            free_association = r.is_present_type(SpeciesReactant)
            free_dissociation = r.is_product_type(SpeciesReactant)
            if not exist_force and not free_association and not free_dissociation:
                dependency_graph.add_node(u, is_dna_dependent=True)
            else:
                dependency_graph.add_node(u, is_dna_dependent=False)
            for reactants in r.reactants_presence:
                parents_l = []
                # don't create reactions for rules that are only dependent on the absence of a molecule at DNA
                if all([
                    (isinstance(pr, DNASpeciesReactant) and torch.any(pr.get_state() == DEFAULT_ID))
                    or isinstance(pr, DNAReactant) for pr in reactants
                ]):
                    continue
                for pr in reactants:
                    parents_l.append(torch.tensor(np.asarray(list(self.get_rules_with_product(
                        product=pr,
                        return_bool=True,
                        return_array=False
                    )))))
                if len(parents_l) == 0:
                    continue
                parents = torch.all(torch.stack(parents_l), dim=0)
                for v, is_parent in enumerate(parents):
                    if not is_parent or v == u:
                        continue
                    dependency_graph.add_edge(v, u)
        return dependency_graph

    @staticmethod
    def get_dependent_rules(dependency_graph: nx.DiGraph, rule_idx: int) -> List:
        return list(dependency_graph.successors(rule_idx))

    def update_c(
            self,
            delta: torch.Tensor,
            min_val: [None, float, torch.Tensor] = 1e-10,
            max_val: [None, float, torch.Tensor] = 1000.
    ):
        if len(delta) != self.num_rules:
            raise ValueError('You must pass one c update value per rule.')

        for i_rule in range(len(self.rule_list)):
            self.rule_list[i_rule].c += delta[i_rule]
            # Clip values
            if min_val is not None:
                if isinstance(min_val, float):
                    minv = torch.tensor(min_val)
                else:
                    if torch.isnan(min_val[i_rule]):
                        minv = self.rule_list[i_rule].c
                    else:
                        minv = min_val[i_rule]
                self.rule_list[i_rule].c = torch.maximum(minv, self.rule_list[i_rule].c)
            if max_val is not None:
                if isinstance(max_val, float):
                    maxv = torch.tensor(max_val)
                else:
                    if torch.isnan(max_val[i_rule]):
                        maxv = self.rule_list[i_rule].c
                    else:
                        maxv = max_val[i_rule]
                self.rule_list[i_rule].c = torch.minimum(maxv, self.rule_list[i_rule].c)

    def update_force(
            self,
            delta: torch.Tensor,
            min_val: [None, float, torch.Tensor] = None,
            max_val: [None, float, torch.Tensor] = None
    ):
        for i_rule in range(len(self.rule_list)):
            if delta[i_rule] == 0.:
                continue
            self.rule_list[i_rule].force += delta[i_rule]
            # Clip values
            if min_val is not None:
                if isinstance(min_val, float):
                    minv = torch.tensor(min_val)
                else:
                    if torch.isnan(min_val[i_rule]):
                        minv = self.rule_list[i_rule].force
                    else:
                        minv = min_val[i_rule]
                self.rule_list[i_rule].force = torch.maximum(minv, self.rule_list[i_rule].force)
            if max_val is not None:
                if isinstance(max_val, float):
                    maxv = torch.tensor(max_val)
                else:
                    if torch.isnan(max_val[i_rule]):
                        maxv = self.rule_list[i_rule].force
                    else:
                        maxv = max_val[i_rule]
                self.rule_list[i_rule].force = torch.minimum(maxv, self.rule_list[i_rule].force)

    def get_c(self) -> torch.Tensor:
        return torch.stack([x.c for x in self.rule_list]).to(self.device)

    def get_c_str(self, sep: str = '\n') -> str:
        return sep.join([r'$\theta_{%d}$: %.3E' % (i_c, Decimal(float(x.c)))
                         for i_c, x in enumerate(self.rule_list)])

    def set_c(self, c: torch.Tensor):
        if len(c) != len(self):
            raise ValueError('Parameter tensor must have the same size as rules')
        for i_rule in range(len(self.rule_list)):
            self.rule_list[i_rule].c = c[i_rule]

    def set_force(self, force: torch.Tensor):
        if len(force) != len(self):
            raise ValueError('Force tensor must have the same size as rules')
        for i_rule in range(len(self.rule_list)):
            self.rule_list[i_rule].force = force[i_rule]

    def get_force(self, return_per_rule: bool = False) -> torch.Tensor:
        if return_per_rule:
            return torch.stack([r.force for r in self.rule_list]).to(self.device)
        else:
            return torch.tensor([r.force for r in self if not torch.all(r.force == 0)]).to(self.device)

    def get_force_str(self, sep: str = '\n') -> str:
        return sep.join([r'$A_{%d}$: %.3E' % (i_c, Decimal(float(r.force)))
                         for i_c, r in enumerate(self) if torch.any(r.force > 0)])

    def add_rule(
            self,
            reactants_presence: List[List[Tuple[Union[List[str], str], Union[List[str], str], str]]],
            products: List[List[Tuple[Union[List[str], str], Union[List[str], str], str]]],
            c: float,
            reactants_absence: List[List[Tuple[Union[List[str], str], Union[List[str], str], str]]] = [],
            force: float = 0.
    ):
        """
        Every element in list is seen as an independent Interactant list.
        This allows modelling independent AND connections.
        Each list in products corresponds to the list at the same index in reactants_presence.
        """
        if self.dna is None:
            raise ValueNotInitialized('DNA Layout', 'add_rule')

        if reactants_presence:
            react_pres_int = []
            for and_rp in reactants_presence:
                and_rp_parameter_l = self._create_parameter_l(and_rp)
                react_pres_int.append(InteractantList(
                    *zip(*and_rp_parameter_l),
                    n_species=self.n_species,
                    n_interact_dna=self.n_interact_dna,
                    n_state_species=self.n_state_species,
                    n_state_dna=self.n_state_dna,
                    device=self.device
                ))
        else:
            react_pres_int = [InteractantList([], [], [], device=self.device)]
        if reactants_absence:
            react_abs_int = []
            for and_ra in reactants_absence:
                and_ra_parameter_l = self._create_parameter_l(and_ra)
                react_abs_int.append(InteractantList(
                    *zip(*and_ra_parameter_l),
                    n_species=self.n_species,
                    n_interact_dna=self.n_interact_dna,
                    n_state_species=self.n_state_species,
                    n_state_dna=self.n_state_dna,
                    device=self.device
                ))
        else:
            react_abs_int = [InteractantList([], [], [], device=self.device)]
        if products:
            prod_int = []
            for p in products:
                p_parameter_l = self._create_parameter_l(p)
                prod_int.append(InteractantList(
                    *zip(*p_parameter_l),
                    n_species=self.n_species,
                    n_interact_dna=self.n_interact_dna,
                    n_state_species=self.n_state_species,
                    n_state_dna=self.n_state_dna,
                    device=self.device
                ))
        else:
            prod_int = [InteractantList([], [], [], device=self.device)]
        self.rule_list.append(Rule(
            self.dna,
            react_pres_int,
            react_abs_int,
            prod_int,
            self.n_species,
            self.n_state_species,
            c,
            force,
            device=self.device
        ))
        self.num_rules += 1
        self.dna.unset_updated_dna()
        self.create_graph()

    def get_all_force_idx(self, time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rule_idx, force_idx = [], []
        for i_r, r in enumerate(self):
            if r.force != 0:
                rule_idx.append(torch.ones(len(r.force), dtype=torch.long).to(self.device) * i_r)
                force_idx.append(torch.ceil(r.force.reshape(-1, 1) * time).type(torch.long))

        return torch.cat(rule_idx), torch.stack(force_idx)

    def add_rules(self, rule_l: List[Rule]):
        self.rule_list.extend(rule_l)
        self.num_rules = len(self.rule_list)
        self.create_graph()

    def get_rules_with_present_reactant(
            self,
            reactant: Interactant,
            return_bool: bool = False,
            return_array: bool = False,
    ) -> Iterable:
        return self._get_rules(Rule.is_present_reactant, return_bool, reactant, return_array)

    def get_rules_with_present_type(
            self,
            interact_type: Type,
            return_bool: bool = False,
            return_array: bool = False
    ) -> Iterable:
        return self._get_rules(Rule.is_present_type, return_bool, interact_type, return_array)

    def get_rules_with_present_str_type(
            self,
            str_interact_type: str,
            return_bool: bool = False,
            return_array: bool = False
    ) -> Iterable:
        return self._get_rules(Rule.is_present_str_type, return_bool, str_interact_type, return_array)

    def get_rules_with_absent_reactant(
            self,
            reactant: Interactant,
            return_bool: bool = False,
            return_array: bool = False,
    ) -> Iterable:
        return self._get_rules(Rule.is_absent_reactant, return_bool, reactant, return_array)

    def get_rules_with_absent_type(
            self,
            interact_type: Type,
            return_bool: bool = False,
            return_array: bool = False
    ) -> Iterable:
        return self._get_rules(Rule.is_absent_type, return_bool, interact_type, return_array)

    def get_rules_with_absent_str_type(
            self,
            str_interact_type: str,
            return_bool: bool = False,
            return_array: bool = False
    ) -> Iterable:
        return self._get_rules(Rule.is_absent_str_type, return_bool, str_interact_type, return_array)

    def get_rules_with_product(
            self,
            product: Interactant,
            return_bool: bool = False,
            return_array: bool = False,
    ) -> Iterable:
        return self._get_rules(Rule.is_product, return_bool, product, return_array)

    def get_rules_with_product_type(
            self,
            interact_type: Type,
            return_bool: bool = False,
            return_array: bool = False
    ) -> Iterable:
        return self._get_rules(Rule.is_product_type, return_bool, interact_type, return_array)

    def get_rules_with_product_str_type(
            self,
            str_interact_type: str,
            return_bool: bool = False,
            return_array: bool = False
    ) -> Iterable:
        return self._get_rules(Rule.is_product_str_type, return_bool, str_interact_type, return_array)

    def get_rule_idx(self, idx: int) -> Rule:
        return self.rule_list[idx]

    def get_reactants_absence_idx(self, idx: int) -> InteractantList:
        return self.rule_list[idx].reactants_absence

    def get_reactants_presence_idx(self, idx: int) -> InteractantList:
        return self.rule_list[idx].reactants_presence

    def get_products_idx(self, idx: int) -> InteractantList:
        return self.rule_list[idx].products

    def get_c_idx(self, idx: int) -> float:
        return self.rule_list[idx].c

    def get_force_idx(self, idx: int) -> float:
        return self.rule_list[idx].force


def rule_set_factory(
        interact_species: Dict[str, int],
        interact_dna: Dict[str, int],
        state_species_dict: Dict[str, int],
        state_dna_dict: Dict[str, int],
        n_rule_set: int = 1,
        dna: Union[DNALayout, None] = None,
        device: Union[torch.device, int] = torch.device('cpu')

):
    return (RuleSet(
        interact_species,
        interact_dna,
        state_species_dict,
        state_dna_dict,
        dna=dna,
        device=device
    ) for _ in range(n_rule_set))


