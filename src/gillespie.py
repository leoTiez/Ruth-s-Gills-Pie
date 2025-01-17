import warnings
from typing import Tuple, List, Union, Iterable, Callable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cycler import cycler
from pathlib import Path
import numpy as np
import torch
import imageio

from src.interactants import SpeciesReactant, InteractantList
from src.rules import RuleSet
from src.dna import DNALayout
from src.utils import get_img, print_progress, check_cell_state, check_train_params
from src.tensorOperations import smooth_tensor, create_tensor_kernel, MaxOccurrencePool
from src.errors import ReactionError, ValueNotInitialized


# Globally show raised warnings when using this module
warnings.simplefilter('always', UserWarning)


class GillespieSampler:
    def __init__(
            self,
            rules: RuleSet,
            dna: DNALayout,
            proteins: torch.Tensor,
            D: torch.Tensor = torch.tensor(1.),
            n_cells: int = 20,
            uncertainty: int = 50,
            seq_bias: int = 100,
            smoothing_window: str = 'hann',
            device: Union[torch.device, int] = torch.device('cpu')
    ):
        self.device = device
        self.dna = dna
        self.rules = rules
        # Uncertainty kernel extends associated species in each cell sample
        # That can be visualized as follows
        # ------0-------
        # ---0000000----
        # It therefore represents an uncertainty about the exact position of an associated species.
        # However, use it with caution, as over several cell samples, that can result in higher protein levels
        # (ie reaching 100 % occupancy)
        if uncertainty % 2 == 1:
            uncertainty += 1
        self.culture_kernel = MaxOccurrencePool(
            kernel_size=uncertainty,
            minlength=self.rules.n_state_species,
            device=self.device
        ).requires_grad_(False).to(self.device)
        # Create smoothing tensor to simulate sequencing bias
        self.sequencing_kernel = create_tensor_kernel(seq_bias, smoothing_window).to(self.device)

        # Create cell memory
        self.n_cells = n_cells
        self.proteins = proteins
        self.D = D
        self.dna_memory = torch.zeros((
            self.n_cells,
            self.dna.size,
            self.rules.n_species + 1
        ), dtype=torch.int8).to(self.device)
        self.cell_memory = torch.zeros((self.n_cells, self.rules.n_species, self.rules.n_state_species),
                                       dtype=torch.double)

        self.time = 0.
        self.step = 0
        self.reset()

    def __call__(
            self,
            n_samples: int = 1,
            cell_idx: Union[None, torch.Tensor] = None,
            a_react: Union[None, torch.Tensor] = None,
            tau: Union[None, torch.Tensor] = None,
            mu: Union[None, torch.Tensor] = None
    ) -> torch.Tensor:
        if n_samples > self.n_cells:
            warnings.warn('Passed more samples than tracked cells. Reduced value to n_samples to %d' % self.n_cells)
            n_samples = self.n_cells
        # Sample cells
        if cell_idx is None:
            cell_idx = torch.multinomial(
                torch.ones(self.n_cells),
                replacement=False,
                num_samples=n_samples
            ).to(self.device)
        # Calculate a value state
        if a_react is None:
            a_react = self.calc_a(cell_idx)
        # Sample reaction
        if tau is None or mu is None:
            tau, mu = self.reaction_sampler(a_react, self.device)
        # Update state
        for num, (t, m, cidx) in enumerate(zip(tau, mu, cell_idx)):
            if m == -1:
                raise ReactionError()

            self.dna_memory[cidx], self.cell_memory[cidx], react_prob, prod_prob = self.rules(
                m,
                self.dna_memory[cidx],
                self.cell_memory[cidx],
                t,
                D=self.D
            )

        # Update time
        dtau = torch.mean(tau)
        self.time += dtau
        self.step += 1
        return dtau

    def calc_a(self, sampled_cells: torch.Tensor) -> torch.Tensor:
        cell_state = check_cell_state(self.dna_memory[sampled_cells])
        # Altho not cell state, it reshapes it correctly to the expected dimensions
        free_proteins = check_cell_state(self.cell_memory[sampled_cells])
        # expect cell_state to be of shape cell x pos x n_species + 1
        # expect free proteins to be of shape cell x species x state
        a_react = []
        for i_rule, rule in enumerate(self.rules):
            presence_states_list, presence_dna_mask, presence_protein_mask = rule.get_presence_information()
            absence_states_list, absence_dna_mask, absence_protein_mask = rule.get_absence_information()
            h_mu = torch.ones(cell_state.shape[0], dtype=torch.double).to(self.device) * self.dna.size
            n_proteins = torch.ones(cell_state.shape[0]).to(self.device)
            for apm in absence_protein_mask:
                # Number of proteins inhibiting reactions is divided
                # This assumes that if there are several proteins per reactant list, they can be used as logical or
                inhibit_factor = torch.sum(
                    free_proteins[..., apm].reshape(free_proteins.shape[0], -1),
                    dim=1
                )
                n_proteins /= torch.where(inhibit_factor > .0,  inhibit_factor, 1.)

            for i_react, ppm in enumerate(presence_protein_mask):
                # Number of proteins possible reactions is multiplied
                # This assumes that if there are several proteins per reactant list, they can be used as logical OR
                if torch.any(ppm):
                    n_proteins *= torch.sum(
                        free_proteins[..., ppm].reshape(free_proteins.shape[0], -1),
                        dim=1
                    )
                h_mu = torch.minimum(h_mu, torch.sum(rule.apply_react_mask(cell_state, i_react), dim=-1))
            a_react.append(self.rules[i_rule].c * h_mu * n_proteins)
        a_react = torch.stack(a_react).to(self.device).T
        return a_react

    @staticmethod
    def reaction_sampler(a_react: torch.Tensor, device: Union[torch.device, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        a0 = torch.sum(a_react, dim=1)
        r1, r2 = torch.rand((2, a_react.shape[0]), dtype=torch.double).to(device)
        tau, mu = -torch.ones(a_react.shape[0], dtype=torch.double).to(device), -torch.ones(
            a_react.shape[0], dtype=torch.long).to(device)
        tau[a0 > 0.] = 1. / a0[a0 > 0.] * torch.log(1. / r1[a0 > 0.])
        mu[a0 > 0.] = torch.searchsorted(
            torch.cumsum(a_react[a0 > 0.], dim=1),
            (a0 * r2)[a0 > 0.].reshape(-1, 1)
        ).reshape(-1).type(torch.long).to(device)
        mu[mu == a_react.shape[1]] = -1
        return tau, mu

    def backprop_call(
            self,
            n_samples: int = 1,
            max_t: Union[float, torch.Tensor] = 30.,
            min_t: float = 1e-2,
            epsilon: float = .1,
            qmask: Union[None, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ####
        # Retrieve all information before update
        # ####
        # Compute cell idc
        cell_idx = torch.multinomial(torch.ones(self.n_cells), replacement=False, num_samples=n_samples).to(self.device)
        # Need to make that reactant and species specific. As not all contribute to the error
        # dimensions a_react: cell x rule
        a_react = self.calc_a(cell_idx)
        tau, mu = self.reaction_sampler(a_react, self.device)

        # ###
        # Start still under research
        # ###
        # Exploration in time
        rand_mask = torch.rand(mu.shape[0]) < epsilon
        tau[rand_mask] = min_t
        # sample_mask = (a_react > 0)
        # if qmask is not None:
        #     sample_mask[:, ~qmask] = False
        # if torch.sum(sample_mask[rand_mask, :].type(torch.float)) == 0.:
        #     sample_mask[:, ~qmask] = True
        # rand_mask[~torch.any(sample_mask, dim=1)] = False
        # rand_mu = torch.tensor([])
        # if torch.any(sample_mask[rand_mask]):
        #     rand_mu = torch.multinomial(
        #         sample_mask[rand_mask, :].type(torch.float),
        #         num_samples=1,
        #         replacement=True
        #     ).reshape(-1).to(self.device)
        # if len(rand_mu) > 0:
        #     mu[rand_mask] = rand_mu

        # ###
        # End still under research
        # ###
        if isinstance(max_t, float):
            max_t = torch.tensor(max_t)
        tau = torch.minimum(tau, max_t)
        dtau = torch.mean(tau)
        # dimensions h_react: cell x rule
        h_react = a_react / self.rules.get_c().reshape(1, -1)

        # dimensions backprop_factors: cells x rule x pos x react x state
        indicator_react, indicator_force = self.rules.determine_backproperties(
            cell_state=self.dna_memory[cell_idx],
            tau=dtau,
            D=self.D
        )

        # dimensions a0: cell x 1
        a0 = torch.sum(a_react, dim=1).reshape(-1, 1)
        # dimensions exp_a0: cell x 1
        exp_a0 = torch.exp(-a0 * dtau)

        # dimensions: cell x rule x pos x react x state
        right_term = indicator_react * exp_a0.reshape(*exp_a0.shape, 1, 1, 1)
        # determine dimensions
        cell_dim, rule_dim, pos_dim, react_dim, state_dim = right_term.shape

        left_term = torch.sum(
            -indicator_react * (a_react * exp_a0 * dtau).reshape(cell_dim, rule_dim, 1, 1, 1),
            dim=1
        ).reshape(cell_dim, 1, pos_dim, react_dim, state_dim)

        # average over all cells. dimensions: rule x pos x react x state
        der_theta = torch.mean((left_term + right_term) * h_react.reshape(cell_dim, rule_dim, 1, 1, 1), dim=0)
        der_force = torch.mean((exp_a0 * a_react).reshape(cell_dim, rule_dim, 1, 1, 1) * indicator_force, dim=0)
        self(n_samples, cell_idx=cell_idx, a_react=a_react, mu=mu, tau=tau)
        # Sum over positions
        # Final dimension: rule x pos x react x state
        return der_theta, der_force

    def reset(self, init_dna: Union[Callable, None] = None, init_data: Union[torch.Tensor, None] = None):
        # The first position in the last dimension represent the dna state
        self.dna_memory[...] = 0
        self.cell_memory[...] = 0.
        if len(self.proteins.shape) == 1:
            self.cell_memory[:, :, 0] = self.proteins.reshape(1, -1)
        elif len(self.proteins.shape) == 2:
            self.cell_memory[:] = self.proteins.reshape(1, *self.cell_memory.shape)
        else:
            raise ValueError('Initialisation for proteins must be given as a one or two dimensional array.')
        if init_dna is not None:
            dm, cm = init_dna(
                self.dna_memory,
                self.cell_memory,
                init_data,
                # Always passed by default
                self.rules.interact_species,
                self.rules.state_species,
                self.rules.state_dna
            )
            # verify that shapes match
            self.dna_memory[:] = dm
            self.cell_memory[:] = cm
        self.time = 0.
        self.step = 0

    def sequence(self, probing: InteractantList) -> Tuple[torch.Tensor, torch.Tensor]:
        # Blur
        simulated_sample, density = self.culture_kernel(self.dna_memory)

        # Smoothing to simulate sequencing inaccuracies and overlay cell states
        sequencing_mean = torch.zeros((self.dna.size, len(probing)), dtype=torch.double).to(self.device)
        sequencing_density = torch.zeros((self.dna.size, len(probing)), dtype=torch.double).to(self.device)
        sequencing_var = torch.zeros((self.dna.size, len(probing)), dtype=torch.double).to(self.device)
        for num, pr in enumerate(probing):
            if isinstance(pr, SpeciesReactant):
                warnings.warn('Cannot sequence unbound proteins: %s' % pr)
                continue
            react = pr.get_reactant().to(self.device)
            state = pr.get_state().to(self.device)
            is_in_mask = torch.isin(simulated_sample[:, :, react], state).type(torch.float).to(self.device)
            sequencing_density[:, num] = smooth_tensor(
                torch.sum(density[:, :, react], dim=(0, -1)).reshape(1, -1) / self.n_cells,
                self.sequencing_kernel,
                device=self.device
            ).reshape(-1)
            sequencing_mean[:, num] = smooth_tensor(
                torch.mean(is_in_mask, dim=0).T,
                self.sequencing_kernel,
                device=self.device
            ).reshape(-1)
            if self.n_cells > 1:
                sequencing_var[:, num] = smooth_tensor(
                    torch.var(is_in_mask, dim=0).T,
                    self.sequencing_kernel,
                    device=self.device
                ).reshape(-1)

        return self.culture_kernel.postcorrection(sequencing_mean, sequencing_density), sequencing_var

    def plot(
            self,
            probing: InteractantList,
            labels: List[str],
            var_alpha: float = .2,
            colors: List = None,
            ax: plt.Axes = None,
            plot_var: bool = True,
            set_legend: bool = True
    ) -> plt.Axes:
        if ax is None:
            plt.figure()
            ax = plt.gca()
        if len(labels) != len(probing):
            warnings.warn('Num probed species is not equal num labels')
        if colors is not None:
            if len(probing) != len(colors) != len(labels):
                warnings.warn('Num of colors does not match num of probed species and labels.'
                              ' Set to default color cycle.')
                colors = None

        all_lines = list(ax.get_lines())
        sequencing_res, sequencing_var = self.sequence(probing)
        sequencing_res = sequencing_res.to(torch.device('cpu'))
        sequencing_var = sequencing_var.to(torch.device('cpu'))
        if colors is None:
            colors = plt.get_cmap('tab10').colors

        if len(all_lines) == 0 or plot_var:
            ax.clear()

        ax.set_prop_cycle(cycler('color', colors))
        if len(all_lines) == 0 or plot_var:
            ax.plot(sequencing_res)
            top = 0.
            if plot_var:
                ax.set_prop_cycle(cycler('color', colors))
                for seq_mean, seq_var in zip(sequencing_res.T, sequencing_var.T):
                    ax.fill_between(
                        np.arange(self.dna.size),
                        seq_mean - seq_var,
                        seq_mean + seq_var,
                        alpha=var_alpha
                    )
        else:
            _, top = ax.get_ylim()
            for data, line in zip(sequencing_res.T, all_lines):
                line.set_ydata(data.detach().numpy())
        ax.set_xlabel('DNA Position')
        ax.set_ylabel('Presence')
        max_val = torch.max(sequencing_res + sequencing_var) if self.n_cells > 1 else 1.
        ax.set_ylim(0, np.maximum(max_val + .05 * max_val, top))  # Add 5 percent padding
        if set_legend:
            ax.legend(loc='upper right')

        return ax


class Gillespy:
    def __init__(
            self,
            data: Union[torch.Tensor, None],
            rules: RuleSet,
            dna: DNALayout,
            proteins: torch.Tensor,
            D: torch.Tensor = torch.tensor(1.),
            n_cells: int = 20,
            uncertainty: int = 200,
            seq_bias: int = 100,
            smooth_data: bool = True,
            smoothing_window: str = 'hann',
            device: Union[torch.device, int] = torch.device('cpu')
    ):
        # Data must be of the shape time x sample x position
        self.device = device
        self.smooth_kernel = create_tensor_kernel(seq_bias, smoothing_window, do_3d=False).to(self.device)
        self.data = None
        if data is not None:
            self.set_data(data, smooth_data)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-10).to(self.device)
        self.sampler = GillespieSampler(
            rules=rules,
            dna=dna,
            proteins=proteins,
            D=D,
            n_cells=n_cells,
            uncertainty=uncertainty,
            seq_bias=seq_bias,
            smoothing_window=smoothing_window,
            device=self.device
        )

        # Dimensions: rules x size x reactants x states
        self.grad_update_theta = torch.zeros((
            len(self.sampler.rules),
            len(self.sampler.dna),
            self.sampler.rules.n_species + 1,
            self.sampler.rules.n_max_states
        ), dtype=torch.double).to(self.device)
        self.grad_update_force = torch.zeros((
            len(self.sampler.rules),
            len(self.sampler.dna),
            self.sampler.rules.n_species + 1,
            self.sampler.rules.n_max_states
        ), dtype=torch.double).to(self.device)

    def set_data(self, data: torch.Tensor, smooth_data: bool):
        self.data = data.to(self.device)
        if smooth_data:
            # Iterate over time points
            for i in torch.arange(self.data.shape[0]):
                self.data[i] = smooth_tensor(self.data[i], self.smooth_kernel, device=self.device)

    def _reset_grad(self):
        self.grad_update_theta = torch.zeros_like(self.grad_update_theta)
        self.grad_update_force = torch.zeros_like(self.grad_update_force)

    def _backprop_tt(
            self,
            probing: InteractantList,
            seq: torch.Tensor,
            tp_idx: int,
            lr: Union[torch.Tensor, float],
            lr_force: Union[torch.Tensor, float],
            decay: Union[torch.Tensor, float],
            error_weight: Union[torch.Tensor, None] = None,
            max_grad_ratio: float = .1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # expected shape species x positions
        error_grad = self.error_derivative(tp_idx, prediction=seq, weight=error_weight)
        grad_theta = torch.zeros(len(self.sampler.rules), dtype=torch.double).to(self.device)
        grad_force = torch.zeros(len(self.sampler.rules), dtype=torch.double).to(self.device)
        # Important: do not average over time! gradients were weighted by dtau
        update_theta_dt = self.grad_update_theta
        update_force_dt = self.grad_update_force
        # Calculate the contribution of parameters theta per probe
        for i_prob, probe in enumerate(probing):
            if isinstance(probe, SpeciesReactant):
                continue
            reacts = probe.get_reactant().reshape(-1, 1)
            states = probe.get_state()
            grad_theta += torch.sum(
                update_theta_dt[..., reacts, states]
                * error_grad[i_prob].reshape(1, -1, 1, 1),
                dim=(1, 2, 3)
            )
            grad_force += torch.sum(
                update_force_dt[..., reacts, states]
                * error_grad[i_prob].reshape(1, -1, 1, 1),
                dim=(1, 2, 3)
            )
        grad_theta /= float(len(self.sampler.dna))
        grad_force /= float(len(self.sampler.dna))
        # Apply learning rate and weight decay and clip gradients
        gradient = -lr * (grad_theta + self.sampler.rules.get_c() * decay)
        clip_mask = torch.abs(gradient) > max_grad_ratio * self.sampler.rules.get_c()
        gradient[clip_mask] = torch.sign(gradient[clip_mask]) * max_grad_ratio * self.sampler.rules.get_c()[clip_mask]
        grad_force = -lr_force * grad_force
        clip_mask_force = torch.abs(grad_force) > max_grad_ratio * self.sampler.rules.get_force(return_per_rule=True)
        grad_force[clip_mask_force] = torch.sign(
            grad_force[clip_mask_force]
        ) * max_grad_ratio * self.sampler.rules.get_force(return_per_rule=True)[clip_mask_force]
        return gradient, grad_force

    def _boost(
            self,
            sampling_boost: float = .1,
            lr: Union[float, torch.Tensor] = 1e-3,
            lower_bound: Union[float, torch.Tensor, None] = 0.,
            upper_bound: Union[float, torch.Tensor, None] = None
    ):
        boost = self.sampler.rules.get_c() * sampling_boost
        if isinstance(lr, torch.Tensor):
            boost[lr == 0] = 0.
        self.sampler.rules.update_c(
            boost,
            min_val=lower_bound,
            max_val=upper_bound
        )

    def error_derivative(
            self,
            tp_idx: int,
            prediction: torch.Tensor,
            weight: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        if self.data is None:
            raise ValueNotInitialized('Data is not set.', 'error_derivative')
        if weight is None:
            weight = torch.ones(self.data.shape[0])
        sq_error_der = 2 * (prediction - self.data[tp_idx])
        return weight.reshape(-1, 1) * sq_error_der

    def calc_error(
            self,
            prediction: torch.Tensor,
            tp_idx: Union[int, None] = None,
            weight: Union[torch.Tensor, None] = None,
            do_average: bool = True
    ) -> torch.Tensor:
        if self.data is None:
            raise ValueNotInitialized('Data is not set.')
        if weight is None:
            weight = torch.ones(self.data.shape[0])
        if tp_idx is None:
            sq_error = weight.reshape(1, -1) * torch.mean((self.data[:, :len(prediction)] - prediction)**2, dim=-1)
        else:
            sq_error = weight * torch.mean((self.data[tp_idx, :len(prediction)] - prediction)**2, dim=-1)
        if do_average:
            return torch.mean(sq_error)
        else:
            return sq_error

    def run(
            self,
            sampling_time: float,
            probing: InteractantList = None,
            dna_init_callback: Union[Callable, None] = None,
            dna_init_data_idc: Union[torch.Tensor, None] = None,
            labels: List = None,
            colors: List = None,
            n_samples: Union[None, int] = None,
            plot_frequency: float = .1,
            verbosity: int = 0,
            save_simulation_result: bool = False,
            save_fig: bool = False,
            save_prefix: str = ''
    ):
        if n_samples is None:
            n_samples = int(.3 * self.sampler.n_cells)
        if n_samples < 1:
            n_samples = 1
        # Prepare plot if verbosity is set.
        if verbosity > 1:
            fig_list = []
            plt.ion()
            fig = plt.figure(figsize=(8, 3), constrained_layout=True)
            ax = plt.gca()
            if colors is None:
                colors = plt.get_cmap('tab:10')(np.linspace(0, 1, 10))
            handles = [Line2D([0], [0], color=c, lw=4) for cnum, c in enumerate(colors) if cnum < len(labels)]
            fig.legend(handles, labels, loc=7)
        if self.data is not None and dna_init_data_idc is not None:
            init_data = self.data[dna_init_data_idc[0], dna_init_data_idc[1]]
        else:
            init_data = None
        self.sampler.reset(init_dna=dna_init_callback, init_data=init_data)
        last_update = -1.
        # Simulate reaction pathway
        while True:
            if self.sampler.time > sampling_time:
                break
            if verbosity > 0:
                print_progress(
                    float(self.sampler.time / sampling_time),
                    prefix='Time %.3f' % self.sampler.time,
                    length=50
                )
            self.sampler(n_samples=n_samples)
            # Plot cell sample if verbosity is set
            if verbosity > 1:
                ax = self.sampler.plot(
                    probing,
                    labels,
                    colors=colors,
                    ax=ax,
                    set_legend=False,
                )
                fig.tight_layout()
                fig.canvas.draw()
                fig.canvas.flush_events()
                if self.sampler.time - last_update > plot_frequency:
                    if save_fig:
                        fig_list.append(get_img(fig))
                        last_update = self.sampler.time.clone()

        if verbosity > 0:
            print_progress(float(self.sampler.time / sampling_time), prefix='Time %.3f' % self.sampler.time, length=50)

        if save_fig:
            if verbosity > 0:
                print('Save simulation to gif')
            Path('figures/simulation').mkdir(parents=True, exist_ok=True)
            imageio.mimsave('figures/simulation/%s_simulation.gif' % save_prefix, fig_list, fps=10)
            plt.close('all')

        if save_simulation_result:
            if verbosity > 0:
                print('Save simulation result to txt file')
            seq_result, _ = self.sampler.sequence(probing)
            Path('data/simulated-data/').mkdir(parents=True, exist_ok=True)
            data_array = np.asarray(seq_result.to(torch.device('cpu')).T)
            np.savetxt('data/simulated-data/%s.tsv' % save_prefix, data_array, delimiter='\t')

    def train(
            self,
            probing: InteractantList,
            seq_tp: torch.Tensor,
            lr: Union[torch.Tensor, float] = 1e-3,
            lr_force: Union[torch.Tensor, float] = 0,
            decay: Union[torch.Tensor, float] = 0.,
            grad_momentum: float = .9,
            tol: float = 1e-8,
            max_iter: int = 1000,
            use_parameter_stop: bool = True,
            n_samples: Union[None, int] = None,
            colors: List = None,
            verbosity: int = 1,
            sampling_boost: float = .1,
            lower_bound: Union[torch.Tensor, float] = 1e-10,
            upper_bound: Union[torch.Tensor, float] = 1000.,
            min_force: Union[torch.Tensor, float] = 0.,
            max_force: Union[torch.Tensor, float] = 10000.,
            max_grad_ratio: float = .1,
            resample_tol_ratio: float = .3,
            seq_momentum: float = .9,
            error_weight: Union[None, torch.Tensor] = None,
            eps_boundaries: Union[None, Tuple[float, float, float]] = None, # (.8, 0.005, 300.) (eps start, eps end, eps decline)
            min_t_ratio: float = 5e-3,  # Increase by .5% during exploration
            dna_init_callback: Union[Callable, None] = None,
            dna_init_data_idc: Union[torch.Tensor, None] = None,
            plotting_window: int = 100,
            save_fig: bool = True,
            save_prefix: str = '',
            save_params: bool = False,
            save_error: bool = False
    ):
        # Initialize optimizer and verify data is in right shape
        if self.data is None:
            raise ValueError('Data not set. Training procedure cannot be run.')
        if len(self.data.shape) is None:
            raise ValueError('Data must be of the shape time x sample x position')
        if self.data.shape[1] > len(probing):
            raise ValueError('The number of NGS signals must be lower or equal to the number of probes')
        if not 0 <= seq_momentum < 1.:
            raise ValueError('Plotting momentum must be between 0 and 1.')

        if n_samples is None:
            n_samples = int(.3 * self.sampler.n_cells)
        if n_samples < 1:
            n_samples = 1
        seq_tp = seq_tp.to(self.device)
        if not isinstance(self.data, torch.Tensor):
            if isinstance(self.data, np.ndarray):
                self.data = torch.tensor(self.data).to(self.device)
            elif isinstance(self.data, Iterable):
                self.data = torch.stack(list(self.data)).to(self.device)
            else:
                raise ValueError('Data array must be of one of the following types: '
                                 'torch.Tensor | numpy.ndarray | Iterable.\n Please convert data accordingly')

        # Check whether training parameters are correctly passed
        lr = check_train_params(len(self.sampler.rules), lr, 'learning rate', self.device)
        decay = check_train_params(len(self.sampler.rules), decay, 'decay', self.device)
        grad_momentum = check_train_params(len(self.sampler.rules), grad_momentum, 'gradient momentum', self.device)
        lr_force = check_train_params(len(self.sampler.rules), lr_force, 'force learning rate', self.device)
        lower_bound = check_train_params(len(self.sampler.rules), lower_bound, 'lower parameter bound', self.device)
        upper_bound = check_train_params(len(self.sampler.rules), upper_bound, 'upper parameter bound', self.device)
        min_force = check_train_params(len(self.sampler.rules), min_force, 'minimum force', self.device)
        max_force = check_train_params(len(self.sampler.rules), max_force, 'maximum force', self.device)
        # Plot data if verbosity flag is set.
        if verbosity > 2:
            if colors is None:
                colors = plt.get_cmap('tab:10')(torch.linspace(0, 1, 10))
            plt.ion()
            lines = []
            fig_list = []
            gradfig_list = []
            fig, ax = plt.subplots(2, 1, figsize=(8, 8))
            # Iterate over time points
            for i in range(self.data.shape[0]):
                ax[0].set_prop_cycle(cycler('color', colors))
                ax[0].plot(self.data[i].T.to(torch.device('cpu')).detach().numpy())
                ax[0].set_prop_cycle(cycler('color', colors))
                lines.extend(ax[0].plot(np.zeros(self.data.shape[1:]).T, linestyle='--'))
            ax[0].set_title('Prediction')
            err_line = ax[1].plot([1.], [0.])[0]
            ax[1].set_title('Error')
            if verbosity > 3:
                fig_grad, ax_grad = plt.subplots(1, 1)
                scaling = 1 + len(self.sampler.rules) // 7
                grad_lines = ax_grad.plot(np.zeros((1, len(self.sampler.rules))))
                ax_grad.legend(
                    [r'$\theta_{%d}$' % i_theta for i_theta in range(len(self.sampler.rules))],
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.15 * scaling),
                    ncol=np.minimum(len(self.sampler.rules), 7)
                )
                ax_grad.set_title('Gradients iteration: 0')
                ax_grad.set_xlabel('Iteration')
                ax_grad.set_ylabel('Relative gradient size')
                fig_grad.tight_layout()
                fig_grad.subplots_adjust(bottom=0.17 * scaling)

            param_text = fig.text(
                .8,
                .7 - .02 * float(len(self.sampler.rules)),
                self.sampler.rules.get_c_str(),
                fontsize=14
            )
            force_text = fig.text(
                .8,
                .3 - .02 * float(len(self.sampler.rules.get_force())),
                self.sampler.rules.get_force_str(),
                fontsize=14
            )
            fig.suptitle('Iteration: 0')

        # Initialize time points and loss.
        seq_over_time = torch.zeros((len(seq_tp), len(probing), len(self.sampler.dna))).to(self.device)
        reaction_time = torch.max(seq_tp)
        old_loss, new_loss = torch.zeros(1).to(self.device), torch.ones(1).to(self.device) * 1e8
        grad_theta_old = torch.zeros(len(self.sampler.rules)).to(self.device)
        grad_force_old = torch.zeros(len(self.sampler.rules)).to(self.device)
        err_data = []
        grad_list = []
        old_param = torch.zeros_like(self.sampler.rules.get_c()) * 1e8
        for i in range(max_iter):
            if use_parameter_stop:
                did_converge = torch.all(
                    torch.abs(old_param - self.sampler.rules.get_c()) <= self.sampler.rules.get_c() * tol)
            else:
                did_converge = torch.all(new_loss <= tol)
            old_param = self.sampler.rules.get_c().clone()
            if did_converge:
                if verbosity > 0:
                    print('Converged to solution')
                break
            if verbosity > 0:
                print('Epoch %d' % i)

            old_loss = new_loss.clone()
            self._reset_grad()
            self.sampler.reset(
                init_dna=dna_init_callback,
                init_data=self.data[dna_init_data_idc[0], dna_init_data_idc[1]] if dna_init_data_idc is not None else None
            )
            grad_theta = torch.zeros_like(grad_theta_old)
            grad_force = torch.zeros_like(grad_force_old)
            last_sequenced = 0

            # Epsilon for exploration
            if eps_boundaries is not None:
                eps = eps_boundaries[1] + (eps_boundaries[0] - eps_boundaries[1]) * torch.exp(torch.tensor(
                    - i / eps_boundaries[2]))
                if verbosity > 0:
                    print('Epsilon: %.3f' % eps)
            else:
                eps = 0.

            # sample over t --> Forward pass
            sample_tp = 0
            while self.sampler.time < reaction_time:
                if verbosity > 1:
                    print_progress(float(self.sampler.time / reaction_time), prefix='Forward progress', length=50)
                dtheta, dforce = self.sampler.backprop_call(
                    n_samples=n_samples,
                    max_t=reaction_time,
                    min_t=reaction_time * min_t_ratio,  # Increase by min_t_ratio% during exploration
                    epsilon=eps,
                    qmask=None if isinstance(lr, float) else lr != 0.
                )
                self.grad_update_theta += dtheta
                self.grad_update_force += dforce

                # Calculate error gradient per time point in data
                if torch.any(torch.logical_and(self.sampler.time > seq_tp, last_sequenced < seq_tp)):
                    last_sequenced = seq_tp[torch.logical_and(self.sampler.time > seq_tp, last_sequenced < seq_tp)]
                    seq_result, seq_var = self.sampler.sequence(probing)
                    seq_result, seq_var = seq_result.T, seq_var.T
                    if i == 0:
                        seq_over_time[sample_tp, :, :] = seq_result
                    else:
                        seq_over_time[sample_tp, :, :] = seq_momentum * seq_over_time[sample_tp] + (
                                1. - seq_momentum) * seq_result

                    # Intermediate backward pass
                    gt, gf = self._backprop_tt(
                        probing,
                        seq_result,
                        sample_tp,
                        lr,
                        lr_force,
                        decay,
                        error_weight=error_weight,
                        max_grad_ratio=max_grad_ratio,
                    )
                    grad_theta += gt
                    grad_force += gf
                    sample_tp += 1

            if verbosity > 1:
                if verbosity > 1:
                    print_progress(1., prefix='Forward progress', length=50)
                print('\n')

            if sample_tp + 1 < self.data.shape[0]:
                warnings.warn('Could not sequence all time points. Increase all parameters by %.1f%% to sample more '
                              'reactions in the given time. Alternatively, you can stop the training procedure and '
                              'increase the given time.' % (100. * sampling_boost))
                self._boost(sampling_boost, lr, lower_bound, upper_bound)
                continue

            # Do last update after full forward pass if sampling was not complete
            if sample_tp + 1 == self.data.shape[0]:
                seq_result, seq_var = self.sampler.sequence(probing)
                seq_result, seq_var = seq_result.T, seq_var.T
                if i == 0:
                    seq_over_time[sample_tp, :, :] = seq_result
                else:
                    seq_over_time[sample_tp, :, :] = seq_momentum * seq_over_time[sample_tp] + (
                            1. - seq_momentum) * seq_result

                gt, gf = self._backprop_tt(
                    probing,
                    seq_result,
                    sample_tp,
                    lr,
                    lr_force,
                    decay,
                    error_weight=error_weight,
                    max_grad_ratio=max_grad_ratio,
                )
                grad_theta += gt
                grad_force += gf

            grad_theta /= float(len(seq_tp))
            grad_force /= float(len(seq_tp))
            # only if the gradient has been already updated, use momentum
            if i > 0:
                grad_theta = grad_momentum * grad_theta_old + (1 - grad_momentum) * grad_theta
                grad_force = grad_momentum * grad_force_old + (1 - grad_momentum) * grad_force
            new_loss = self.calc_error(seq_over_time, weight=error_weight)
            err_data.append(new_loss.clone().to(torch.device('cpu')))
            self.sampler.rules.update_c(
                grad_theta,
                min_val=lower_bound,
                max_val=upper_bound
            )
            if torch.sum((torch.sum(self.grad_update_theta, dim=(1, 2, 3)) == 0).type(torch.double)) \
                    >= resample_tol_ratio * len(self.sampler.rules):
                warnings.warn('Could not sample sufficiently many rules. Increase all parameters by %.1f%% to sample '
                              'more reactions in the given time. Alternatively, you can stop the training procedure '
                              'and increase time.' % (100. * sampling_boost))
                self._boost(sampling_boost, lr, lower_bound, upper_bound)
            self.sampler.rules.update_force(grad_force, min_val=min_force, max_val=max_force)
            grad_theta_old = grad_theta.clone()
            grad_force_old = grad_force.clone()
            if verbosity > 1:
                print('Loss: %.10f' % new_loss)
                # Plot result
                if verbosity > 2:
                    line_i = 0
                    max_seq = torch.tensor(0.)
                    for seq_t in seq_over_time:
                        for seq in seq_t:
                            lines[line_i].set_ydata(seq.to(torch.device('cpu')).detach().numpy())
                            line_i += 1
                            if torch.max(seq) > max_seq:
                                max_seq = torch.max(seq)
                    ax[0].set_ylim(0, torch.maximum(max_seq, self.data.max()).to(torch.device('cpu')).detach().numpy())

                    if i > 0:
                        err_line.set_xdata(torch.arange(1, i + 2, device='cpu'))
                        err_line.set_ydata(err_data)
                        max_err = torch.max(torch.stack(err_data).to(torch.device('cpu')))
                        ax[1].set_ylim(0, max_err + .05 * max_err)
                        ax[1].set_xlim(0, i + 2)

                    param_text.set_text(self.sampler.rules.get_c_str())
                    force_text.set_text(self.sampler.rules.get_force_str())

                    fig.suptitle('Iteration: %d' % i)
                    fig.tight_layout()
                    fig.subplots_adjust(right=0.75)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    fig_list.append(get_img(fig))
                    if verbosity > 3:
                        print('Parameters')
                        print(self.sampler.rules.get_c().to(torch.device('cpu')))
                        print('Force')
                        print(self.sampler.rules.get_force().to(torch.device('cpu')))
                    if verbosity > 3:
                        if isinstance(grad_list, list):
                            grad_list = grad_theta.reshape(1, -1).clone()
                        else:
                            grad_list = torch.cat([
                                grad_list[-plotting_window:, :],
                                grad_theta.reshape(1, -1)
                            ]).reshape(-1, len(self.sampler.rules))
                        plot_grad_list = grad_list.clone()
                        max_grad_norm = torch.max(torch.abs(plot_grad_list), dim=0).values
                        plot_grad_list[:, max_grad_norm > 0] /= max_grad_norm[max_grad_norm > 0].reshape(1, -1)
                        for num, g in enumerate(plot_grad_list.T):
                            grad_lines[num].set_xdata(torch.arange(len(g), device='cpu'))
                            grad_lines[num].set_ydata(g.to(torch.device('cpu')).detach().numpy())
                        grad_max = torch.max(plot_grad_list).to(torch.device('cpu'))
                        grad_min = torch.min(plot_grad_list).to(torch.device('cpu'))
                        ax_grad.set_ylim(grad_min - .05 * torch.abs(grad_min), grad_max + .05 * torch.abs(grad_max))
                        ax_grad.set_xlim(0, plot_grad_list.shape[0] + 1)
                        ax_grad.set_title('Gradients iteration: %d' % i)
                        fig_grad.canvas.draw()
                        fig_grad.canvas.flush_events()
                        gradfig_list.append(get_img(fig_grad))

        if save_fig:
            Path('figures/training').mkdir(parents=True, exist_ok=True)
            imageio.mimsave('figures/training/%s_training.gif' % save_prefix, fig_list, fps=10)
            if verbosity > 3:
                imageio.mimsave('figures/training/%s_grad_training.gif' % save_prefix, gradfig_list, fps=10)
            plt.close('all')
        if save_params:
            Path('data/parameters').mkdir(parents=True, exist_ok=True)
            np.savetxt(
                'data/parameters/%s_parameters.csv' % save_prefix,
                self.sampler.rules.get_c().to(torch.device('cpu')).detach().numpy(),
                delimiter=','
            )
            Path('data/force').mkdir(parents=True, exist_ok=True)
            np.savetxt(
                'data/force/%s_force.csv' % save_prefix,
                self.sampler.rules.get_force(return_per_rule=True).to(torch.device('cpu')).detach().numpy(),
                delimiter=','
            )

        if save_error:
            Path('data/error').mkdir(parents=True, exist_ok=True)
            np.savetxt(
                'data/error/%s_error.csv' % save_prefix,
                torch.stack(err_data).detach().numpy(),
                delimiter=','
            )
