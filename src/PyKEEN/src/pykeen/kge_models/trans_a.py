# -*- coding: utf-8 -*-

"""Implementation of the Region model."""

import logging
from typing import Dict

import numpy as np
import torch
import torch.autograd
from dataclasses import dataclass
from pykeen.constants import NORM_FOR_NORMALIZATION_OF_ENTITIES, RADIUS_INITIAL_VALUE
from pykeen.kge_models.base import BaseModule, slice_triples
from torch import nn

__all__ = [
    'TransA',
    'TransAConfig',
]

log = logging.getLogger(__name__)


@dataclass
class TransAConfig:
    lp_norm: str
    strict_norm: bool
    radius_init: float
    reg_lambda: float
    loss_type: str
    neg_factor: float

    @classmethod
    def from_dict(cls, config: Dict) -> 'RegionConfig':
        """Generate an instance from a dictionary."""
        return cls(
            lp_norm=config[NORM_FOR_NORMALIZATION_OF_ENTITIES],
            strict_norm=config['strict_norm'],
            radius_init=config[RADIUS_INITIAL_VALUE],
            reg_lambda=config['reg_lambda'],
            loss_type=config.get('loss_type', 'MRL'),
            neg_factor=config.get('neg_factor', 1)
        )


class TransA(BaseModule):
    """A modification of TransE [borders2013]_.

     This model considers a relation as a translation from the head to a region, including the tail entity.

    .. [borders2013] Bordes, A., *et al.* (2013). `Translating embeddings for modeling multi-relational data
                     <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`_
                     . NIPS.

    .. seealso::

    """

    model_name = 'TransA'
    margin_ranking_loss_size_average: bool = False
    hyper_params = BaseModule.hyper_params + [
        NORM_FOR_NORMALIZATION_OF_ENTITIES,
        'strict_norm',
        RADIUS_INITIAL_VALUE,
        'reg_lambda',
        'loss_type',
        'neg_factor'
    ]
    single_threshold = True

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        config = TransAConfig.from_dict(config)

        # Embeddings
        self.l_p_norm_entities = config.lp_norm
        self.strict_norm = config.strict_norm
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.region_dim = 3
        self.relation_regions = nn.Embedding(
            self.num_relations,
            self.embedding_dim ** 2)

        self.reg_l = config.reg_lambda
        self.init_radius = config.radius_init
        self.gradient_matrix = False
        self.prob_mode = False

        # TODO: add config parameter and move to base class
        self.loss_type = config.loss_type
        if config.loss_type == 'MRL':
            self.criterion = nn.MarginRankingLoss(
                margin=self.margin_loss,
                reduction='mean'  # self.margin_ranking_loss_size_average
            )
            self.single_pass = False
            self.forward = self._forward_mrl
        elif config.loss_type == 'NLL':
            self.criterion = nn.NLLLoss(
                reduction='sum'  # self.margin_ranking_loss_size_average
            )  # todo: add weights for pos and neg classes
            self.margin_loss = 0
            self.forward = self._forward_nll
            self.single_pass = True
        elif config.loss_type == 'BPR':
            self.criterion = nn.NLLLoss(
                reduction='sum'
            )
            self.forward = self._forward_bpr
            self.single_pass = False

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

        nn.init.constant_(
            self.relation_regions.weight.data,
            0
        )

        #self.relation_regions.weight.data.fill_diagonal_(self.init_radius)
        self.relation_regions.weight.requires_grad = self.gradient_matrix

        norms = torch.norm(self.relation_embeddings.weight, p=2, dim=1).data
        self.relation_embeddings.weight.data = self.relation_embeddings.weight.data.div(
            norms.view(self.num_relations, 1).expand_as(self.relation_embeddings.weight))

    def _normalize(self):
        if self.do_normalization:
            # Normalize embeddings of entities
            if self.strict_norm:
                # normalize to norm=1
                norms = torch.norm(self.entity_embeddings.weight, p=self.l_p_norm_entities, dim=1).data
                self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
                    norms.view(self.num_entities, 1).expand_as(self.entity_embeddings.weight))
            else:
                # normalize to norm<=1
                norms = torch.norm(self.entity_embeddings.weight, p=2, dim=1)
                mask = norms > 1
                self.entity_embeddings.weight.data[mask] = (self.entity_embeddings.weight.data / norms.unsqueeze(1))[mask]

    def update(self, pos_triples, neg_triples):
        # adjust the region matrix to the last embedding updates
        if not self.gradient_matrix:
            def update_r(r):
                pos_r = pos_triples[pos_triples[:, 1] == r]
                head_embeddings, relation_embeddings, tail_embeddings, _ = \
                    self._get_triple_embeddings(pos_r)
                pos_d = torch.abs(head_embeddings + relation_embeddings - tail_embeddings)

                if torch.isnan(head_embeddings + relation_embeddings + tail_embeddings).any():
                    print("bad backprop")
                neg_r = neg_triples[neg_triples[:, 1] == r]
                head_embeddings, relation_embeddings, tail_embeddings, _ = \
                    self._get_triple_embeddings(neg_r)
                if torch.isnan(head_embeddings + relation_embeddings + tail_embeddings).any():
                    print("bad backprop")
                neg_d = torch.abs(head_embeddings + relation_embeddings - tail_embeddings)

                pos_w = torch.sum(torch.bmm(
                    pos_d.view(-1, self.embedding_dim, 1),
                    pos_d.view(-1, 1, self.embedding_dim)
                ), dim=0)
                if torch.isnan(pos_w).any() or torch.isinf(pos_w).any():
                    print("nan in posw")

                neg_w = torch.sum(torch.bmm(
                    neg_d.view(-1, self.embedding_dim, 1),
                    neg_d.view(-1, 1, self.embedding_dim)
                ), dim=0)
                if torch.isnan(neg_w).any() or torch.isinf(neg_w).any():
                    print("nan in negw")

                new_matrix = (- pos_w + neg_w).view(-1)

                if (torch.abs(new_matrix) > 1000).any():
                    print("large weights after update in relation {}".format(r))
                    # print(new_matrix)
                    pass
                if (torch.isnan(new_matrix)).any():
                    print("nan weights after update")
                    # print(new_matrix)
                    pass
                # set non-diagonal values to zero
                # new_matrix[[i != j for i in range(50) for j in range(50)]] = 0.

                # make non-negative
                new_matrix[new_matrix < 0] = 0.

                if (new_matrix == 0).all():
                    pass
                    #print("no weights at all")
                # new_matrix = torch.sigmoid(new_matrix)
                self.relation_regions.weight.data[r] = new_matrix

            map(update_r, np.array(range(self.num_relations)))

    def predict(self, triples):
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def get_reg_loss(self):
        rels = self.relation_embeddings(torch.tensor(range(self.num_relations), device=self.device))
        ents = self.entity_embeddings(torch.tensor(range(self.num_entities), device=self.device))
        return (torch.pow(rels.view(-1), 2).sum() + torch.pow(ents.view(-1), 2).sum())

    def _score_triples(self, triples):
        head_embeddings, relation_embeddings, tail_embeddings, relation_regions = \
            self._get_triple_embeddings(triples)

        if self.gradient_matrix:
            m_x = (head_embeddings + relation_embeddings - tail_embeddings).unsqueeze(-1)
        else:
            m_x = torch.abs(head_embeddings + relation_embeddings - tail_embeddings).unsqueeze(-1)

        if torch.isnan(head_embeddings).any():
            print("nan in heads")
        if torch.isnan(tail_embeddings).any():
            print("nan in tails")
        if torch.isnan(relation_embeddings).any():
            print("nan in relations embedidngs")
        if torch.isnan(relation_regions).any():
            print("nan in regions")
            return

        dists = torch.matmul(
            torch.matmul(m_x.transpose(-1, -2), relation_regions),
            m_x).squeeze(-1)

        return dists

    def _forward_mrl(self, batch_positives, batch_negatives):
        self._normalize()

        pos = self._score_triples(batch_positives)
        neg = self._score_triples(batch_negatives)

        #loss = torch.sum(
        #    torch.max(pos - neg + self.margin_loss, torch.tensor(0.0, device=self.device))
        #)/ len(pos)


        loss = self.criterion(pos, neg, torch.tensor([-1.0], device=self.device))

        loss = loss + self.reg_l * self.get_reg_loss()
        return loss

    def _forward_bpr(self, batch_positives, batch_negatives):
        self._normalize()
        pos = self._score_triples(batch_positives)
        neg = self._score_triples(batch_negatives)
        loss = torch.sum(-torch.log(torch.sigmoid(neg - pos) + 1e-15))

        loss = loss + self.reg_l * len(pos) * self.get_reg_loss()
        return loss

    def _get_triple_embeddings(self, triples):
        heads, relations, tails = slice_triples(triples)
        return (
            self._get_entity_embeddings(heads),
            self._get_relation_embeddings(relations),
            self._get_entity_embeddings(tails),
            self._get_relation_regions(relations)
        )

    def _get_relation_embeddings(self, relations):
        return self.relation_embeddings(relations).view(-1, self.embedding_dim)

    def _get_relation_regions(self, relations):
        if self.region_dim < 4:
            if self.region_dim == 1:
                values = self.relation_regions(relations).view(-1, 1, 1)
            elif self.region_dim == 2:
                values = self.relation_regions(relations).view(-1, self.embedding_dim, 1)
            else:
                values = self.relation_regions(relations).view(-1, self.embedding_dim, self.embedding_dim)
                return values

            sigmas = torch.log(1 + torch.exp(values))
            matrix = (sigmas * torch.eye(self.embedding_dim, device=self.device))
        else:
            bs = len(relations)
            emb = self.embedding_dim

            # get embedding values
            values = self.relation_regions(relations)

            # fill triangular matrix
            mask = torch.ones(emb, emb, device=self.device).tril() == 1
            triang_index = torch.stack([mask for _ in range(bs)])
            triang_matrix = torch.zeros(bs, emb, emb, device=self.device)
            triang_matrix[triang_index] = values.view(-1)

            # positive diagonal
            mask = torch.eye(emb, emb, device=self.device) == 1
            diag_index = torch.stack([mask for _ in range(bs)])
            triang_matrix[diag_index] = torch.log(1 + torch.exp(triang_matrix[diag_index]))

            # get region matrix
            matrix = triang_matrix.bmm(triang_matrix.transpose(-2, -1))

        return matrix
