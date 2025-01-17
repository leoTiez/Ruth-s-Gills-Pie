import numpy as np
import networkx as nx
from src.rules import RuleSet, DNASpeciesReactant, DEFAULT_ID, DNAReactant, SpeciesReactant
import matplotlib.pyplot as plt
from typing import Tuple, List
from pathlib import Path


def create_graph(rule_set: RuleSet) -> nx.DiGraph:
    graph = nx.DiGraph()
    for u, r in enumerate(rule_set):
        exist_force = np.any(r.dmove != 0.)
        free_association = r.is_present_type(SpeciesReactant)
        free_dissociation = r.is_product_type(SpeciesReactant)
        if not exist_force and not free_association and not free_dissociation:
            graph.add_node(u, is_dna_dependent=True)
        else:
            graph.add_node(u, is_dna_dependent=False)
        for reactants in r.reactants_presence:
            parents_l = []
            # don't create reactions for rules that are only dependent on the absence of a molecule at DNA
            if all([
                (isinstance(pr, DNASpeciesReactant) and np.any(pr.state == DEFAULT_ID))
                or isinstance(pr, DNAReactant) for pr in reactants
            ]):
                continue
            for pr in reactants:
                parents_l.append(list(rule_set.get_rules_with_product(
                    product=pr,
                    return_bool=True,
                    return_array=False
                )))
            if len(parents_l) == 0:
                continue
            parents = np.all(parents_l, axis=0)
            for v, is_parent in enumerate(parents):
                if not is_parent or v == u:
                    continue
                graph.add_edge(v, u)
    return graph


def get_parent(graph: nx.DiGraph, rule_num: int) -> Tuple[List[int], List[bool]]:
    parents = []
    is_independent = []
    for p in graph.predecessors(rule_num):
        parents.append(p)
        is_independent.append(not graph.nodes[p]['is_dna_dependent'])
    return parents, is_independent


def find_independent_parent(graph: nx.DiGraph, rule_num: int) -> List[int]:
    q = [rule_num]
    independent_parents = []
    while True:
        if len(q) == 0:
            break
        current_node = q.pop()
        for p in graph.predecessors(current_node):
            if graph.nodes[p]['is_dna_dependent']:
                q.append(p)
            else:
                independent_parents.append(p)

    return independent_parents


def find_independent_child(graph: nx.DiGraph, rule_num: int) -> List[int]:
    q = [rule_num]
    independent_children = []
    while True:
        if len(q) == 0:
            break
        for c in graph.successors(q.pop()):
            if graph.nodes[c]['is_dna_dependent']:
                q.append(c)
            else:
                for p in graph.predecessors(c):
                    if not graph.nodes[p]['is_dna_dependent']:
                        break
                else:
                    independent_children.append(c)
    return independent_children


def get_dna_dependent_rules(graph: nx.DiGraph) -> List[int]:
    return [n for n in graph.nodes if graph.nodes[n]['is_dna_dependent']]


def plot_graph(
        graph: nx.DiGraph,
        node_size=500,
        fig_size: Tuple[int, int] = (8, 7),
        save_fig: bool = False,
        save_prefix: str = ''
):
    try:
        pos = nx.planar_layout(graph)
    except nx.NetworkXException:
        pos = nx.shell_layout(graph)

    dna_dependent_nodes = get_dna_dependent_rules(graph)
    fig = plt.figure(figsize=fig_size)
    nx.draw(graph, pos, ax=plt.gca(), node_size=node_size)
    nx.draw(graph.subgraph(dna_dependent_nodes), pos, ax=plt.gca(), node_color='red', node_size=node_size)
    nx.draw_networkx_labels(graph, pos)
    fig.suptitle('Rule network')
    fig.tight_layout()
    if save_fig:
        Path('figures/rules').mkdir(parents=True, exist_ok=True)
        fig.savefig('figures/rules/%s_rule_network.png' % save_prefix)
    else:
        plt.show()
