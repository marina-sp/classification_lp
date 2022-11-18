# -*- coding: utf-8 -*-

"""Implementation of the TransE model."""

import logging
from typing import Dict

import numpy as np
import torch
import torch.autograd
from dataclasses import dataclass
from pykeen.constants import NORM_FOR_NORMALIZATION_OF_ENTITIES, SCORING_FUNCTION_NORM, TRANS_E_NAME
from pykeen.kge_models.base import BaseModule, slice_triples
from torch import nn

__all__ = [
    'TransE',
    'TransEConfig',
]

log = logging.getLogger(__name__)


@dataclass
class TransEConfig:
    lp_norm: str
    scoring_function_norm: str
    loss_type: str

    @classmethod
    def from_dict(cls, config: Dict) -> 'TransEConfig':
        """Generate an instance from a dictionary."""
        return cls(
            lp_norm=config[NORM_FOR_NORMALIZATION_OF_ENTITIES],
            scoring_function_norm=config[SCORING_FUNCTION_NORM],
            loss_type=config.get('loss_type', 'MRL')
        )


class TransE(BaseModule):
    """An implementation of TransE [borders2013]_.

     This model considers a relation as a translation from the head to the tail entity.

    .. [borders2013] Bordes, A., *et al.* (2013). `Translating embeddings for modeling multi-relational data
                     <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`_
                     . NIPS.

    .. seealso::

       - Alternative implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py
    """

    model_name = TRANS_E_NAME
    margin_ranking_loss_size_average: bool = False
    hyper_params = BaseModule.hyper_params + [SCORING_FUNCTION_NORM, NORM_FOR_NORMALIZATION_OF_ENTITIES]

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        config = TransEConfig.from_dict(config)

        # Embeddings
        self.l_p_norm_entities = config.lp_norm
        self.scoring_fct_norm = config.scoring_function_norm
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.forward = self._forward_mrl if config.loss_type == 'MRL' else self._forward_bpr

        self._initialize()

    def _initialize(self):
        embeddings_init_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )
        nn.init.uniform_(
            self.relation_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )

        norms = torch.norm(self.relation_embeddings.weight, p=2, dim=1).data
        self.relation_embeddings.weight.data = self.relation_embeddings.weight.data.div(
            norms.view(self.num_relations, 1).expand_as(self.relation_embeddings.weight))

    def predict(self, triples):
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def _forward_mrl(self, batch_positives, batch_negatives):
        # Normalize embeddings of entities
        norms = torch.norm(self.entity_embeddings.weight, p=self.l_p_norm_entities, dim=1).data
        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
            norms.view(self.num_entities, 1).expand_as(self.entity_embeddings.weight))

        positive_scores = self._score_triples(batch_positives)
        negative_scores = self._score_triples(batch_negatives)
        loss = self._compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return loss

    def _forward_bpr(self, batch_positives, batch_negatives):
        norms = torch.norm(self.entity_embeddings.weight, p=self.l_p_norm_entities, dim=1).data
        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
            norms.view(self.num_entities, 1).expand_as(self.entity_embeddings.weight))

        pos = self._score_triples(batch_positives)
        neg = self._score_triples(batch_negatives)

        loss = torch.sum(-torch.log(torch.sigmoid(neg - pos)))
        return loss


    def _compute_loss(self, positive_scores, negative_scores):
        y = np.repeat([-1], repeats=positive_scores.shape[0])
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        # Scores for the positive and negative triples
        #positive_scores = torch.tensor(positive_scores, dtype=torch.float, device=self.device)
        #negative_scores = torch.tensor(negative_scores, dtype=torch.float, device=self.device)

        loss = self.criterion(positive_scores, negative_scores, y)
        return loss

    def _score_triples(self, triples):
        head_embeddings, relation_embeddings, tail_embeddings = self._get_triple_embeddings(triples)
        scores = self._compute_scores(head_embeddings, relation_embeddings, tail_embeddings)
        return scores

    def _compute_scores(self, head_embeddings, relation_embeddings, tail_embeddings):
        """Compute the scores based on the head, relation, and tail embeddings.

        :param head_embeddings: embeddings of head entities of dimension batchsize x embedding_dim
        :param relation_embeddings: emebddings of relation embeddings of dimension batchsize x embedding_dim
        :param tail_embeddings: embeddings of tail entities of dimension batchsize x embedding_dim
        :return: Tensor of dimension batch_size containing the scores for each batch element
        """
        # Add the vector element wise
        sum_res = head_embeddings + relation_embeddings - tail_embeddings
        distances = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        return distances

    def _get_triple_embeddings(self, triples):
        heads, relations, tails = slice_triples(triples)
        return (
            self._get_entity_embeddings(heads),
            self._get_relation_embeddings(relations),
            self._get_entity_embeddings(tails),
        )

    def _get_relation_embeddings(self, relations):
        return self.relation_embeddings(relations).view(-1, self.embedding_dim)
