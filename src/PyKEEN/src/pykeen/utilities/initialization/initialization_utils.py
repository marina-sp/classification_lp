# -*- coding: utf-8 -*-

"""Script for initializing the knowledge graph embedding models."""

from typing import Dict, Mapping

import torch.cuda as cuda
import torch.optim as optim
from pykeen.constants import (
    ADAGRAD_OPTIMIZER_NAME, ADAM_OPTIMIZER_NAME, LEARNING_RATE, OPTMIZER_NAME, SGD_OPTIMIZER_NAME,
    GPU
)

__all__ = [
    'OPTIMIZERS',
    'get_optimizer',
    'get_device_name'
]

OPTIMIZERS: Mapping = {
    SGD_OPTIMIZER_NAME: optim.SGD,
    ADAGRAD_OPTIMIZER_NAME: optim.Adagrad,
    ADAM_OPTIMIZER_NAME: optim.Adam,
}


def get_optimizer(config: Dict, kge_model):
    """Get an optimizer for the given knowledge graph embedding model."""
    optimizer_name = config[OPTMIZER_NAME]
    optimizer_cls = OPTIMIZERS.get(optimizer_name)

    if optimizer_cls is None:
        raise ValueError(f'invalid optimizer name: {optimizer_name}')

    parameters = filter(lambda p: p.requires_grad, kge_model.parameters())

    return optimizer_cls(parameters, lr=config[LEARNING_RATE])


def get_device_name(preferred_device):
    if cuda.is_available() and preferred_device.startswith(GPU):
        if preferred_device == GPU:
            gpu_idx = '0'
        else:
            gpu_idx = preferred_device[3:]
    return 'cuda:' + gpu_idx if preferred_device.startswith(GPU) else 'cpu'
