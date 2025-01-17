#!/usr/bin/env python3
import os
from typing import List, Dict, Union, Generator
from collections.abc import Iterable as AbstractIterable
from pathlib import Path
import numpy as np
import torch


def load_unknown_cmd_params(unknown_params: List[str]) -> Dict[str, str]:
    kw_args = {}
    for arg in unknown_params:
        if arg.startswith(('-', '--')):
            key, value = arg.split('=')
            key = key.strip('-') if arg.startswith('-') else key.strip('--')
            kw_args[key] = value
    return kw_args


def validate_dir(rel_path=''):
    curr_dir = os.getcwd()
    Path('%s/%s/' % (curr_dir, rel_path)).mkdir(parents=True, exist_ok=True)
    return '%s/%s/' % (curr_dir, rel_path)


def get_img(fig):
    graph = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    graph = graph.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return graph


def flatten_nested(nested_list: AbstractIterable) -> Generator:
    for nl in nested_list:
        if isinstance(nl, AbstractIterable) and not isinstance(nl, (str, bytes)):
            yield from flatten_nested(nl)
        else:
            yield nl


def print_progress(ratio: float, prefix: str = 'Progress', length: int = 100):
    progress = int(ratio * length)
    print(f"\r{prefix}: [{u'█' * progress}{('.' * (length - progress))}] %.3f%%" % (ratio * 100.), end='', flush=True)


def check_cell_state(cell_state: torch.Tensor) -> torch.Tensor:
    shape = cell_state.shape
    if len(shape) < 2:
        raise ValueError('Cell state must have at least two dimensions: position x species.')
    if len(shape) == 2:
        cell_state = cell_state.reshape(1, *shape)
    if len(shape) > 3:
        raise ValueError('Cell state has too many dimensions. Expect: (n_cells: optional) x position x species.')

    return cell_state


def check_train_params(
        n_rules: int,
        parameter: Union[torch.Tensor, float],
        name: str,
        device: Union[torch.device, int] = torch.device('cpu')
) -> Union[torch.Tensor, float]:
    def raise_error(param_name: str):
        raise ValueError('Pass either float %s or pass one %s per rule.' % (param_name, param_name))

    if isinstance(parameter, torch.Tensor):
        if len(parameter) != n_rules:
            raise_error(name)
        parameter = parameter.to(device)
    return parameter



