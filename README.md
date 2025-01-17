# Ruth's Gills Pie: A Rule-Based Gillespie Algorithm for Simulating Particle Interactions With a Polymer in Python
The advent of high-throughput Next Generation Sequencing (NGS) technology allowed the assessment of interaction profiles
on a genome-wide scale. Over the years, this has been extended to measure a large variety of different property distribution
over the entire DNA. However, it can be challenging to link these results back to the actual biological processes. Due
to the sheer complexity of possible interactions, it is usually not straightforward to formulate concrete hypotheses
where certain properties are expected to be found. This becomes particularly difficult when it is important to include
the stability of the interaction. For example, an increased protein occupancy can mean many short-lived DNA-protein bindings
or many stable ones.

The software package **GillesPy** is an easy-to-use and general computational framework to simulate sequencing experiments based on 
the Gillespie algorithm [1] and the implementation of user-defined interaction rules, i.e. interaction between molecules.
We implemented a fast parameter estimation using a modified version of Backpropagation based on a differentiable 
approximation of the Gillespie update step. To our knowledge, this is the first holistic simulation and training package
for NGS data which can be applied to most molecular pathways.

Rules are defined by a condition, a causal effect, and the frequency of the event within an arbitrary but fixed time step (e.g. minute). They commonly consist of 
protein-protein or protein-DNA association/dissociation behaviour, but they can equally contain movements along the DNA.
We extended the traditional Gillespie algorithm [1] such that it contains the notion of space to implement DNA-segment 
specific occupation profiles. Proteins as well as DNA can transition between different states to modulate behaviour. 
This allows for example the representation of ubiquitinated proteins and damaged DNA.
Note that the library can be used for simulating entire chromosomes as well as single compartments. We present here the usage
for protein interactions with the DNA. Yet it can be equally applied to any particle interactions along a one-dimensional polymer
(such as ribosomal interactions with an aminoacid chain), where distributional data is available.
          

## Installation
The software requires `python` (`3.6` or higher, tested on `3.9`). If `pip` is installed, run
```commandline
python3.9 -m pip install -r requirements.txt
```
to download and install all necessary packages.

## Spatial Gillespie Algorithm
Chemical reactions (i.e. the user-defined rules) are randomly sampled based on the possible number of interactions given 
the current state of the system (i.e. the simulated cell) and the frequency (which is understood as a surrogate for probability) to occur in a given time step 
independent of the system state. Let's consider a reaction between protein *A* and *B* with each 100 proteins in a given
volume. There are 100 x 100m = 10000 interactions possible between the proteins of species *A* and *B*.
The reaction happens with a frequency of *f*=0.0025. The number of interactions that are expected to be observed
in the system with a given volume within the next time step is therefore 10000 * 0.0025 = 25. Based on this distribution
over all possible reactions, it is possible to sample a reaction and the corresponding time step when we expect to observe it.

The spatial Gillespie algorithm assumes that there is no global knowledge. Every protein or particle interacts 
independently with the DNA molecule. However, we make several important assumptions and simplifications

- The DNA is in a well-mixed solution. Therefore, we do not need to keep track of every protein that is in the nucleus. Note that this assumption could possibly limit the length of DNA that can be reasonably represented due to the possible creation of phases through protein-protein interactions
- As soon as a protein interacts with the DNA, we incorporate the notion of space. Therefore, we randomly sample the position to which the protein binds. The set of possible positions can be restricted by the given rules.
- We enforce that only one protein of the same kind (independent of its state) can associate to the same position on the DNA in a single cell. For example, there can be only one Pol II protein complex at position 35 of chromosome X.
- Proteins of different kinds can bind to the same position.

The notion of space incorporated by directed particle movements in a force field, which makes use of the Smoluchowski equation.
The rules can be simulated for several cells in parallel. Aggregating the occupancy profiles results in the NGS data. In order to limit
the amount of data produced (or necessary for training), we presume that proteins do not interact with each other in the well-mixed solution.
There are no reaction sampled between proteins that are not bound to the DNA or not associating through the reaction. The overall reaction
mechanism which performs the simulation and training is implemented in the `gillespie.py` module.


## States and interactants
All particles must be defined as interactants, i.e. molecules that can participate in a reaction. That includes
the DNA and any other particle, such as proteins, lipids or snoRNAs etc. Whilst the user is free to define any particles
they like, they will be all categorised into: DNA, bound interactant, free interactant. It is clear that proteins will
transition between bound and free, and they therefore change their corresponding group dynamically. The categories
are important for defining the rules with respect to different protein conformations (i.e. bound protein or free protein).
However, it is not expected that the polymer (i.e. the DNA) changes its group. More information about how the categories are
used can be found in Section Rules. The implementation of interactants is defined in the `interactants.py` module.
They can be given names, but the internal representation are numbers to facilitate index-based updates. 
Follow this syntax template

```python
from src.interactants import convert_to_dict
proteins = ['protein A', 'protein B']
dna_segments = ['core promoter', 'tss']
protein_states = ['associated', 'ubiquitinated']
dna_states = ['damaged', 'repaired']

protein_dict = convert_to_dict(proteins)
dna_dict = convert_to_dict(dna_segments)
protein_state_dict = convert_to_dict(protein_states)
dna_state_dict = convert_to_dict(dna_states)

protein_a_id = protein_dict['protein A']
```

Every dictionary contains also a `default` state (id 0) and an `unspecific` state (id -1). `default` represents all undefined
states, whereas `unspecific` can be used to define a rule over all types of proteins or all type of protein states etc (more below
in Section Rules).

Each defined protein, molecule and DNA position can be in different states. It is therefore possible to make the rules
dependent on specific states. For example, elongation of ubiquitinated Pol2 is possible but any other Pol2 is not affected.
It also allows transitioning between different states to simulate a particular process. E.g. damaged DNA can transition
to cleaved, removed, replaced, and repaired.


## The Gillespie DNA
Proteins can associate from the well-mixed solution to the DNA or dissociate from it. The implementation makes it possible
to categorise positions to enforce different DNA-protein interactions (e.g. the TSS or the core promoter). 
Each position can be only assigned to one segment assigned. They are considered to be relatively stable and cannot be changed by a rule.
If a change is required, this must be done by specifically overwriting the positions to be part of a new segment. 
All undefined positions are set to `default`. The DNA is implemented
in the `dna.py` module. The DNA can be created with the following syntax

```python
from src.dna import DNALayout
import torch
dna_size = 1000
# Each segment is defined by the triple (start position, end position, type)
cp_start = 100
cp_end = 300
tss_start = 300
tss_end = 400
device = torch.device('cpu')  # Tensor flow device
dna_specs = [
        (cp_start, cp_end, dna_dict['core promoter']),
        (tss_start, tss_end, dna_dict['tss'])
]
dna = DNALayout(dna_size, dna_specs, device=device)
```

The Gillespie *DNA* implements the same algorithm as described in [1], but it includes the notion of space. Thus, next to the type of
reaction and the reaction time, we sample also a position to and from which a protein is associating/dissociating. 
The set of possible positions can be restricted by the defined rules to a particular class of positions. For example, 
Pol II can only be ubiquitinated at the TSS. 

![Gillespie interactions](figures/paper/gillespy-gateway.png)

## Rules
Rules are a set of reactants (with their respective states) and the corresponding products (with their respective states) together with the reaction 
probability. Per default, it is assumed that this is a reaction frequency per minute. Additionally, it is possible to
define a direction of movement along the DNA, either positive or negative direction (depending on their interpretation).
As rules can be made dependent on several segments, it is possible to limit movement. For example, the elongation of
Pol2 can be restricted to TSS, transcript, and TES.

Rules are implemented in the following syntax

```python
from src.rules import DNA_REACTANT, SPECIES_REACTANT, DNA_SPECIES_REACTANT, DEFAULT, UNSPECIFIC, rule_set_factory
# create a rule set
rule_set, = rule_set_factory(
    protein_dict,
    dna_dict,
    protein_state_dict,
    dna_state_dict,
    n_rule_set=1,   # Number of rule sets to be created. Normally 1, but if a change between rule sets 
                    # is required during simulation, several can be instantiated simultaneously
    dna=dna,  # the DNA instance
    device=device  # Tensorflow device. Either cpu or a specified gpu
)

# Rules are defined by the following pseudo-formal logic notation
# (present reactants) AND (absent reactants) -> (products)
# Each participating interactant (reactant and product) is defined by the triplet (name, state, type),
# where name the name of the molecule (e.g. protein A or tss), state is the state of the molecule (e.g. default or 
# ubiquitinated), and type is the type of particle (i.e. DNA reactant (DNA_REACTANT), 
# bound protein (DNA_SPECIES_REACTANT), or free protein (DNA_REACTANT))
# Note that rules are not defined by the molecule ids but by their names. 
# The value UNSPECIFIC is passed when a rule can be applied to any value of a kind. For example, if the rule is
# unspecific with respect to the state of protein A, it can be applied to any of them. Similarly, this can
# be applied for entire particle types (e.g. a rule can be applied to protein A and B). Note that if the
# product is also defined to be unspecific, it is set to the value of the sampled reactant. If a rule can be applied
# to several state but not to all, they can be defined in brackets within the species definition, e.g. ('protein A', ['associated, 'ubiquitinated'], DNA_SPECIES_REACTANT).

# Each outer bracket defines a DNA region where these rules apply. You can therefore define a rule for chromatin
# changes at different DNA positions at the same time. All DNA regions must occur in reactants and products, otherwise
# an error is raised.

# IMPORTANT: the DEFAULT state for the type DNA_SPECIES_REACTANT represents a free position on the DNA. All 
# association rules should be defined by requiring the presence of a free position for a given particle.

# This rule defines the association of protein A in an ubiquitinated state to the core promoter if protein B is present at the TSS in an
# ubiquitinated state and no other protein A is already present at the core promoter.
rule_set.add_rule(
    # Reactants that must be present for a reaction to occur
    reactants_presence=[
     # Required state at the core promoter
     [
        ('core promoter', DEFAULT, DNA_REACTANT),
        ('protein A', UNSPECIFIC, SPECIES_REACTANT),
        ('protein A', DEFAULT, DNA_SPECIES_REACTANT),
    ],
    # Required state at the TSS
    [
        ('tss', DEFAULT, DNA_REACTANT),
        ('protein B', 'ubiquitinated', DNA_SPECIES_REACTANT)
    ]
    ],
    # Only association when protein A is not already present
    reactants_absence=[[
        ('core promoter', DEFAULT, DNA_REACTANT),
        ('protein A', ['associated', 'ubiquitinated'], DNA_SPECIES_REACTANT)
    ]],
    products=[
     # Define the product state at the core promoter
     [
        ('core promoter' , DEFAULT, DNA_REACTANT),
        ('protein A', 'ubiquitinated', DNA_SPECIES_REACTANT)
     ],
     # Define product state at the tss. This is crucial to be defined, even though there is no change
     [
        ('tss', DEFAULT, DNA_REACTANT),
        ('protein B', 'ubiquitinated', DNA_SPECIES_REACTANT)
     ]
     
    ],
    c=0.1,
    force=0  # Optional
)
```

## Simulation
With all the prior definitions, it is now possible to simulate the NGS data with the parameters provided in the rules.
A GillesPy object can be easily instantiated and executed via
```python
from src.interactants import UNSPECIFIC, InteractantList, DNASpeciesReactant, DNAReactant, DNA_SPECIES_REACTANT, DNA_REACTANT
from src.gillespie import Gillespy
import torch
n_proteins = {'protein A': 200, 'protein B': 100}  # Number proteins available in well-mixed solution
D = 100  # Amount of noise assumed in update positions for particle movements
n_cells = 30  # Number of simulated cells
# Uncertainty about the exact particle position. This window sets all protein states along the DNA within the window
# to the most occurring state. This can be also understood as representing several similar cells at the same time
uncertainty = 20  
smoothing = 100  # Smoothing of the simulated NGS which aims to represent the sonication step
# Define the probed sequencing signal as an output
probing = [('protein A', UNSPECIFIC, DNA_SPECIES_REACTANT), ('protein B', 'ubiquitinated', DNA_SPECIES_REACTANT)]
sequencing_colors = ['tab:blue', 'tab:green']  # Define sequencing colors
sampling_time = 30  # Sampling time in a fixed time unit (normally minutes)
# Number of cells that are updated in a time step. It's recommended to set this value lower than 1 to avoid
# oscillations due to synchronicity
n_samples = int(.3 * sum(n_proteins.values())) 
verbosity = 3  # set verbosity level
proteins = torch.zeros(rule_set.n_species)
# Define number of free available proteins in the well-mixed solution around the DNA
for prot_name, value in n_proteins:
     proteins[protein_dict[prot_name]] = value
gillespy_simulator = Gillespy(
      data=data,  # Can be None for simulations, but are required if you use a dna init callback function
      rules=rule_set,
      D=D,
      proteins=proteins,
      n_cells=n_cells,
      dna=dna,
      uncertainty=uncertainty,
      seq_bias=smoothing,
      device=device
  )

inter_list = []
inter_names = []
for probe in probing:
    if probe[-1] == DNA_SPECIES_REACTANT:
        inter_names.append('%s:%s' % (probe[0], probe[1]))
        probe_type = DNASpeciesReactant
        react = protein_dict[probe[0]]
        state = protein_state_dict[probe[1]]
        n_reactant = len(protein_dict) - 1 # Reduce by unspecific
        n_state = len(protein_state_dict) - 1 # Reduce by unspecific
    elif probe[-1] == DNA_REACTANT:
        inter_names.append('dna:%s' % probe[1])
        probe_type = DNAReactant
        react = dna_dict[UNSPECIFIC]  # Cannot sequence position specific, ie. only core promoter
        state = dna_state_dict[probe[1]]
        n_reactant = len(dna_dict) - 1,  # Reduce by unspecific
        n_state = len(dna_state_dict) - 1  # Reduce by unspecific
    else:
        continue
    inter_list.append(probe_type(
        reactant=react,
        state=state,
        n_reactant=n_reactant,
        n_state=n_state
    ))
    
seq_probing = InteractantList(inter_list=inter_list)
gillespy_simulator.run(
    sampling_time,
    probing=seq_probing,
    labels=inter_names,
    n_samples=n_samples,
    dna_init_callback=None,  # Callable for initial DNA chromatin state. When none, simulation starts from a free polymer
    dna_init_data_idc=None,  # The data indices that are used by the dna init callback
    colors=sequencing_colors,
    verbosity=verbosity,
    save_simulation_result=False,  # Save resulting NGS distribution to a csv
    save_fig=False,  # Save NGS simulation as a animated gif
    save_prefix='',  # Prefix that is added to saved data files as identifier 
    plot_frequency= .1  # Simulated time difference when plots are updated
)
```
## Training
Training can be performed using a modified backpropagation algorithm. It takes distributional data for different
particle species and time points and finds parameter estimates that can recreate the data. Error weights can 
be set species-specific, eg. for `protein A` there is a different contribution to the error `w=10` than for `protein B`
(`w=1`)
It is highly important to note that 
the behaviour does not follow standard machine learning best practices. For example, learning rates and parameters can
differ over several orders of magnitudes, as the number of sampled positions on the DNA where a specific protein is bound
are commonly much smaller than the number of free proteins that can participate during association. We highly recommend
to define reasonable higher and lower bounds for each parameter to avoid exploding values and gradients. Due to the
stochastic sampling, this allows the evaluation of various reaction sequences (exploration) whilst limiting the search space
within which the solution should converge (exploitation). Momentum and weights can drag parameters quickly down, and
we recommend setting them not too high if at all. 

The training allows the definition of protein and DNA callbacks, which a functions which define the initial state
from which onwards the simulation and training should begin after each training iteration. If not defined, the training
will start from an empty DNA polymer. 

Training can be similarly executed as the simulation. However, it is highly important to set the data before training
starts. Replace `gillespy_simulator.run` function with the something similar to the following template and definitions

```python
tp = torch.tensor([15., 25.])  # sequencing time points
# pass one learning rate per rule or one learning rate for all. When set to 0, the parameter is not trained
lr = torch.tensor([.01, 0., 1e-6, 10.])   
# force specific learning rate. Set one per rule or one lr for all. Note that this value must be commonly very large
lr_force = 1e10  
# Set momentum, recommended to be set not too large
momentum = .5
# Set weight decay. Caution should be used, as some implemented pathways are highly sensitive to small values 
decay = 0.
# Number of training iterations
n_epoch = 500 
# Error tolerance when the alogrithm is assumed to have converged. Note that the error is computed over the entire
# DNA, but your rule definitions might not allow protein interactions everywhere. It is unlikely that the error
# becomes very small
tol = 1e-3
# Define parameter specific lower and upper bounds. Also possible to set a single value for all rule
lower_bounds = torch.tensor([1e-3, 1e-2, 1e-6, 1.])
upper_bounds = torch.tensor([1e-1, 1e-2, 1e-4, 50.])
# Upper and lower bounds for force values. Can be defined with a single value for all rules or rule-specific
min_force = 50
max_force = 1000
# Define error weights
error_weight = torch.tensor([10., 1.])
# Update all parameters by (sampling boost * parameter value) when not sufficiently many reactions have been sampled.
sampling_boost = .1
# Apply momentum to simulated trained sequencing data 
seq_momentum = 0.

gillespy_estimator.train(
      probing=seq_probing,
      seq_tp=tp,
      lr=lr,
      lr_force=lr_force,
      decay=decay,
      grad_momentum=momentum,
      n_samples=n_samples,
      max_iter=n_epoch,
      tol=tol,
      use_parameter_stop=False,  # If set, the error tolerance break criterion is applied to the parameter updates
      lower_bound=lower_bounds,
      upper_bound=upper_bounds,
      min_force=min_force,
      max_force=max_force,
      error_weight=error_weight,
      sampling_boost=sampling_boost,
      dna_init_callback=None,  # Callable for initial DNA chromatin state. When none, simulation starts from a free polymer
      dna_init_data_idc=None,  # The data indices that are used by the dna init callback
      colors=sequencing_colors,
      save_fig=False,
      save_prefix='',
      save_params=False,  # Save parameters to csv
      save_error=False,  # Save loss to csv
      seq_momentum=seq_momentum,
      verbosity=verbosity,
  )
```

## Use the Boilerplate
To simplify usage, we allow the dynamic loading of rules and interactant definitions during runtime. The framework
automatically converts everything to the internal syntax. Although the library can be imported into customised scripts
to allow maximal flexibility and behaviour, we highly recommend the usage of dynamic loading using our boilerplates. 
Following in **GillesPy** specific syntax, data, rules, and definition of interactants can be automatically loaded
and converted for both simulation and training. We provide a template in `examples/trainingFileTemplate.py` and many
other examples in the same folder. In a nutshell, all training or simulation files must contain:

- a `get_parameters` function that returns all necessary parameters in a dictionary
- a `get_data` function which returns the necessary distributional data per time point and species. Note that this can be left empty for simulations
- a `get_rules` function that returns the rule definitions, the dna, and all defined interactants.

These files can be automatically loaded by the `gTraining.py` or the `gSimulation.py` script by setting the appropriate
command line parameter to the path of the training or simulation file. For example, you can start a simulation
whose parameters are defined in the file `simulation/yourDefs.py` by
```commandline
python3.8 gSimulation.py --simulation_file=simulation.yourDefs
```
or similarly for the training
```commandline
python3.8 gTraining.py --training_file=simulation.yourDefs
```

Run
```commandline
python3.8 gSimulation.py --help
```
or
```commandline
python3.8 gTraining.py --help
```
for more information about possible command line parameters. Specific parameters that are required by your defined script
can be passed as additional cli parameters with the same naming as string. Bear in mind that this requires the correct conversion
which you need to implement in your script.
See our examples for more information.

## Run the Experiments
Experiments are implemented in `experimentProfiles.py`. Run

```commandline
python3.8 experimentProfiles.py --training_file=examples.repairRules
```
for the gene-specific parameter estimation of the repair rules and

```commandline
python3.8 experimentProfiles.py --training_file=examples.transcriptionGene
```
for the gateway problem.


## Run an Easy Example
We provide several examples in the `example` folder. A very simple pathway of a single
protein associating and dissociating along the DNA is given by `easyExample.py`. Run the simulation via

```commandline
python3.8 gSimulation.py --verbosity=3 --simulation_file=examples.easyExample --smoothing=150 --uncertainty=30 --n_cells=50 --sampling_time=45 --do_train=False
```

Similarly, run the training through

```commandline
python3.8 gTraining.py --verbosity=4 --training_file=examples.easyExample --smoothing=150 --uncertainty=30 --n_cells=50 --n_epoch=500 --tol=0 --seq_momentum=.0 --do_train=True
```

Note that the predicted/trained dissociation parameter is much lower than in the simulation. This reflects the fact that
dissociation is barely sampled, and much lower values could explain a similar distribution. Therefore, weight decay should be used
with caution.

## Example Figures
In the following, we simulate transcription on an imaginary gene using a transcription factor (Rad3 for *S. cerevisiae*), 
and the RNA Polymerase II (Pol2). Shaded areas give the variance along the entire simulated cell culture.
![Cell culture](figures/simulation/transcription_dummy_simulation.gif)

We want to emphasise that Pol 2 is now seemingly everywhere along the transcript. However, the actual number of cells
that contain Pol2 at this particular position might be actual very low. This is exemplified in the second simulation, 
where we plot only a single cell.

![Single cell](figures/simulation/transcription_dummy_single_simulation.gif)


## References
[1] Gillespie, Daniel T.
 "Exact stochastic simulation of coupled chemical reactions." 
 The journal of physical chemistry 81.25 (1977): 2340-2361.  

[2] Erixon, Klaus, and Gunnar Ahnström. 
"Single-strand breaks in DNA during repair of UV-induced damage in normal human and
xeroderma pigmentosum cells as determined by alkaline DNA unwinding and hydroxylapatite chromatography:
effects of hydroxyurea, 5-fluorodeoxyuridine and 1-β-d-arabinofuranosylcytosine on the
kinetics of repair."
Mutation Research/Fundamental and Molecular Mechanisms of Mutagenesis 59.2
(1979): 257-271.
