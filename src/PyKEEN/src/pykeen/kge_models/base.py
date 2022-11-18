# -*- coding: utf-8 -*-

"""Utilities for getting and initializing KGE models."""

from typing import Dict, Optional, Union, List

import numpy as np
import torch
from dataclasses import dataclass
from pykeen.constants import (
    EMBEDDING_DIM, LEARNING_RATE, MARGIN_LOSS, NUM_ENTITIES, NUM_RELATIONS, PREFERRED_DEVICE,
)
from torch import nn

__all__ = [
    'BaseModule',
    'BaseConfig',
    'slice_triples',
]


@dataclass
class BaseConfig:
    """Configuration for KEEN models."""

    device: bool
    margin_loss: Union[float, List]
    number_entities: int
    number_relations: int
    embedding_dimension: int
    corrupt_relations: bool
    do_normalization: bool

    @classmethod
    def from_dict(cls, config: Dict) -> 'BaseConfig':
        """Generate an instance from a dictionary."""
        return cls(
            device=config.get(PREFERRED_DEVICE),
            margin_loss=config[MARGIN_LOSS],
            number_entities=config[NUM_ENTITIES],
            number_relations=config[NUM_RELATIONS],
            embedding_dimension=config[EMBEDDING_DIM],
            corrupt_relations=config.get('corrupt_relations', False),
            do_normalization=config.get('normalize', True)
        )


class BaseModule(nn.Module):
    """A base class for all of the models."""

    margin_ranking_loss_size_average: bool = ...
    entity_embedding_max_norm: Optional[int] = None
    entity_embedding_norm_type: int = 2
    hyper_params = [EMBEDDING_DIM, MARGIN_LOSS, LEARNING_RATE, 'corrupt_relations']
    single_threshold = False

    def __init__(self, config: Union[Dict, BaseConfig]) -> None:
        super().__init__()

        if not isinstance(config, BaseConfig):
            config = BaseConfig.from_dict(config)

        # Device selection
        self.device = torch.device(config.device)

        # Output type (distance or probability)
        self.prob_mode = False
        self.single_pass = False
        self.neg_factor = 1

        # Loss
        self.margin_loss = config.margin_loss
        self.opt = 'SGD'

        if type(self.margin_loss) != list:
            self.criterion = nn.MarginRankingLoss(
                margin=self.margin_loss,
                size_average=False  #  self.margin_ranking_loss_size_average,
            )

        # Entity dimensions
        #: The number of entities in the knowledge graph
        self.num_entities = config.number_entities
        #: The number of unique relation types in the knowledge graph
        self.num_relations = config.number_relations
        #: The dimension of the embeddings to generate
        self.embedding_dim = config.embedding_dimension

        self.entity_embeddings = nn.Embedding(
            self.num_entities,
            self.embedding_dim,
            norm_type=self.entity_embedding_norm_type,
            max_norm=self.entity_embedding_max_norm,
        )

        self.do_normalization = config.do_normalization
        self.relation_thresholds = np.ndarray((self.num_relations,))

    def __init_subclass__(cls, **kwargs):  # noqa: D105
        if not getattr(cls, 'model_name', None):
            raise TypeError('missing model_name class attribute')

    def _get_entity_embeddings(self, entities):
        return self.entity_embeddings(entities).view(-1, self.embedding_dim)

    def _normalize(self):
        return

    def update(self, *args):
        return

def slice_triples(triples):
    """Get the heads, relations, and tails from a matrix of triples."""
    h = triples[:, 0:1]
    r = triples[:, 1:2]
    t = triples[:, 2:3]
    return h, r, t
