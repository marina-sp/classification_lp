# -*- coding: utf-8 -*-

"""Implementation of the Region model."""

import logging
from typing import Dict
import pickle as pkl

import numpy as np
import torch
import torch.autograd
from dataclasses import dataclass
from pykeen.constants import NORM_FOR_NORMALIZATION_OF_ENTITIES, REGION_NAME, RADIUS_INITIAL_VALUE
from pykeen.kge_models.base import BaseModule, slice_triples
from torch import nn

__all__ = [
    'Region',
    'RegionConfig',
]

log = logging.getLogger(__name__)


@dataclass
class RegionConfig:
    lp_norm: str
    strict_norm: bool
    radius_init: float
    reg_lambda: float
    loss_type: str
    neg_factor: float
    region_type: str
    gradient_matrix: bool
    conv_score: bool

    @classmethod
    def from_dict(cls, config: Dict) -> 'RegionConfig':
        """Generate an instance from a dictionary."""
        return cls(
            lp_norm=config[NORM_FOR_NORMALIZATION_OF_ENTITIES],
            strict_norm=config['strict_norm'],
            radius_init=config[RADIUS_INITIAL_VALUE],
            reg_lambda=config['reg_lambda'],
            loss_type=config.get('loss_type', 'MRL'),
            neg_factor=config.get('neg_factor', 1),
            region_type=config.get('region_type', 'sphere'),
            gradient_matrix=config.get('gradient_matrix', True),
            conv_score=config.get('conv_score', False)
        )


class Region(BaseModule):
    """A modification of TransE [borders2013]_.

     This model considers a relation as a translation from the head to a region, including the tail entity.

    .. [borders2013] Bordes, A., *et al.* (2013). `Translating embeddings for modeling multi-relational data
                     <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`_
                     . NIPS.

    .. seealso::

    """

    model_name = REGION_NAME
    margin_ranking_loss_size_average: bool = False
    hyper_params = BaseModule.hyper_params + [
        NORM_FOR_NORMALIZATION_OF_ENTITIES,
        'strict_norm',
        RADIUS_INITIAL_VALUE,
        'reg_lambda',
        'loss_type',
        'neg_factor',
        'region_type',
        'gradient_matrix',
        'conv_score'
    ]
    single_threshold = True

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        config = RegionConfig.from_dict(config)

        # Embeddings
        self.l_p_norm_entities = config.lp_norm
        self.strict_norm = config.strict_norm
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim).double()
        self.entity_embeddings = self.entity_embeddings.double()
        if config.region_type == 'sphere':
            self.region_dim = 1
            self.relation_regions = nn.Embedding(
                self.num_relations,
                1).double()
        elif config.region_type == 'ellipse':
            self.region_dim = 2
            self.relation_regions = nn.Embedding(
                self.num_relations,
                self.embedding_dim).double()
        elif config.region_type.startswith('outer'):
            self.region_dim = 5
            rang = int(config.region_type[5:])
            self.relation_regions = nn.Embedding(
                self.num_relations,
                self.embedding_dim * rang).double()
        elif config.region_type == 'full':
            self.region_dim = 3
            self.relation_regions = nn.Embedding(
                self.num_relations,
                self.embedding_dim ** 2).double()
        else:
            self.region_dim = 4
            self.relation_regions = nn.Embedding(
                self.num_relations,
                self.embedding_dim * (self.embedding_dim - 1) // 2 + self.embedding_dim).double()

        self.reg_l = config.reg_lambda
        self.init_radius = config.radius_init
        self.conv_score = config.conv_score

        if self.conv_score:
            self.opt = 'Adam'
            self.dropout = nn.Dropout(0.2)
            self.conv_layer = nn.Conv1d(2, 10, kernel_size=5).double()
            self.linear = nn.Linear((self.embedding_dim - 5 + 1)*10, self.embedding_dim).double()

        # whether to train region matrix via backprop (or to derive it)
        self.gradient_matrix = False if self.region_dim == 3 else True


        # TODO: add config parameter and move to base class
        self.loss_type = config.loss_type
        if config.loss_type == 'MRL':
            self.criterion = nn.MarginRankingLoss(
                margin=self.margin_loss,
                reduction='sum'  # self.margin_ranking_loss_size_average
            )
            self.single_pass = False
            self.forward = self._forward_mrl
            self.prob_mode = True
        elif config.loss_type == 'NLL':
            self.criterion = nn.NLLLoss(
                reduction='sum'  # self.margin_ranking_loss_size_average
            )  # todo: add weights for pos and neg classes
            self.margin_loss = 0
            self.forward = self._forward_nll
            self.single_pass = True
            self.prob_mode = True
        elif config.loss_type == 'BPR':
            self.criterion = nn.NLLLoss(
                reduction='sum'
            )
            self.forward = self._forward_bpr
            self.single_pass = False
            self.prob_mode = False

        self._initialize()
        print(self.region_dim)

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
            self.init_radius
        )
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

            # check region embeddings:
            # values = self.relation_regions(torch.tensor(range(self.num_relations), device=self.device))
            # #values = torch.sigmoid(values)
            # bs = len(values)
            # matrix = torch.bmm(values.view(bs, self.embedding_dim, -1), values.view(bs, -1, self.embedding_dim))
            # print('Symmetry: ', (matrix == matrix.transpose(-1, -2)).all())
            # idx = (matrix != matrix.transpose(-1,-2)).any(dim=-1).any(dim=-1)
            # print(matrix[idx])
            # print(values[idx])

    def update(self, pos_triples, neg_triples):
        # adjust the region matrix to the last embedding updates
        if not self.gradient_matrix:
            for r in range(self.num_relations):
                pos_r = pos_triples[pos_triples[:, 1] == r]
                head_embeddings, relation_embeddings, tail_embeddings, _ = \
                    self._get_triple_embeddings(pos_r)
                #pos_d = torch.abs(head_embeddings + relation_embeddings - tail_embeddings)
                pos_d = (head_embeddings + relation_embeddings - tail_embeddings)

                neg_r = neg_triples[neg_triples[:, 1] == r]
                head_embeddings, relation_embeddings, tail_embeddings, _ = \
                    self._get_triple_embeddings(neg_r)
                #neg_d = torch.abs(head_embeddings + relation_embeddings - tail_embeddings)
                neg_d = (head_embeddings + relation_embeddings - tail_embeddings)

                posm = torch.sum(torch.bmm(
                            pos_d.view(-1, self.embedding_dim, 1),
                            pos_d.view(-1, 1, self.embedding_dim)
                        ), dim=0)

                try:
                    posm.view(self.embedding_dim, -1).cholesky()
                except:
                    print("non psd positive correlations!")


                negm = torch.sum(torch.bmm(
                            neg_d.view(-1, self.embedding_dim, 1),
                            neg_d.view(-1, 1, self.embedding_dim)
                        ), dim=0)

                try:
                    negm.view(self.embedding_dim, -1).cholesky()
                except:
                    print("non psd negative correlations!")

                new_matrix = (
                        - torch.sum(torch.bmm(
                            pos_d.view(-1, self.embedding_dim, 1),
                            pos_d.view(-1, 1, self.embedding_dim)
                        ), dim=0)
                        + torch.sum(torch.bmm(
                            neg_d.view(-1, self.embedding_dim, 1),
                            neg_d.view(-1, 1, self.embedding_dim)
                        ), dim=0)
                ).view(-1)

                try:
                    new_matrix.view(self.embedding_dim, -1).cholesky()
                except:
                    print("non psd!")

                if torch.isinf(new_matrix).any():
                    print("inf weights after update")
                if torch.isinf(new_matrix).any():
                    print("nan weights after update")
                if (torch.abs(new_matrix) > 100).any():
                    pass
                    # print("large weights after update")
                    # print(new_matrix)

                # set non-diagonal values to zero
                # new_matrix[[i != j for i in range(50) for j in range(50)]] = 0.

                # make non-negative
                new_matrix[new_matrix < 0] = 0.

                # new_matrix = torch.sigmoid(new_matrix)
                self.relation_regions.weight.data[r] = new_matrix


    def predict(self, triples):
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def get_reg_loss(self, batch_size):
        embeddings = self.relation_regions(torch.tensor(range(self.num_relations), device=self.device))
        return (
            self.reg_l * batch_size / self.embedding_dim / self.num_relations *
            torch.norm(embeddings.reshape(-1), 2)
        )

    def _score_triples(self, triples):
        head_embeddings, relation_embeddings, tail_embeddings, region_embeddings = \
            self._get_triple_embeddings(triples)

        if self.conv_score:
            conv_out = self.conv_layer(
                self.dropout(
                    torch.stack([head_embeddings, relation_embeddings], dim=1)
                )
            )
            translated = self.linear(
                self.dropout(
                    conv_out.view((len(head_embeddings), -1))
                )
            )
            m_x = (translated - tail_embeddings)
        elif self.gradient_matrix:
            m_x = (head_embeddings + relation_embeddings - tail_embeddings)
        else:
            m_x = torch.abs(head_embeddings + relation_embeddings - tail_embeddings)
        m_x.unsqueeze_(-1)

        if torch.isnan(head_embeddings).any():
            print("nan in heads")
        if torch.isnan(tail_embeddings).any():
            print("nan in tails")
        if torch.isnan(relation_embeddings).any():
            print("nan in relations embeiddngs")
        if torch.isnan(region_embeddings).any():
            print("nan in regions")
            return

        dists = torch.bmm(
            torch.bmm(m_x.transpose(-1, -2), region_embeddings),
            m_x).squeeze(-1)

        if not self.prob_mode:  # bpr case
            return dists

        if (dists == 0).any():
            log.debug("zero distances in loss computation")
        # if (dists < 0).any():
        #     idx = dists < 0
        #     log.debug("negative distances in loss computation: %d"%sum(idx))
        #     log.debug(str(dists[idx][:10]))
        #     eigs, _ = region_embeddings[idx.squeeze()].symeig()
        #     int_idx = idx.squeeze().nonzero().type(torch.long)
        #     log.debug(str(self.relation_regions(int_idx)))
        #     log.debug(str(region_embeddings[idx.squeeze()]))
        #     pkl.dump((region_embeddings[idx.squeeze()].detach().cpu().numpy(), self.relation_regions(int_idx).detach().cpu().numpy()), open('logs/wrong_vector.pkl','wb'))
        #     if idx.sum():
        #         exit(1)
        #     #log.debug(str(eigs.tolist()))
        #     log.debug("negative eigenvalues: %d"%(eigs < 0).sum())
        #
        #     log.debug("critically negative: %d"%(dists < -1).sum())
        #     if (dists < -1).sum():
        #         exit(1)

        # TODO: try other activation, like tanh instead of logistic
        scores = 1.0 / (1 + dists)
        # print("Dist values: ", dists[:10])
        # print("Probs values: ", probs[:10])
        # print("M x: ", m_x)
        return scores

    def _forward_mrl(self, batch_positives, batch_negatives):
        self._normalize()

        pos = self._score_triples(batch_positives)
        positive_loss = - torch.log(pos)

        neg = self._score_triples(batch_negatives)
        if (pos < 0).any() or (neg < 0).any():
            log.debug("%d negative input to log function from pos"%(len(pos[pos < 0])))
            log.debug("%d negative input to log function from neg"%(len(neg[neg < 0])))

            log.debug(str(pos[pos < 0][:5]))
            log.debug(str(neg[neg < 0][:5]))
        if (pos == 0).any():
            log.debug("zero input from pos to log function")
        if (neg == 0).any():
            log.debug("zero input from neg to log function")
        negative_loss = - torch.log(neg)

        # loss
        y = np.repeat([-1], repeats=positive_loss.shape[0])
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        loss = torch.sum(
            (positive_loss - negative_loss + self.margin_loss).clamp_min_(0)
        )  # loss = self.criterion(positive_scores, negative_scores, y)

        if self.reg_l:
            loss = loss + self.get_reg_loss(len(positive_loss))

        return loss

    def _forward_bpr(self, batch_positives, batch_negatives):
        self._normalize()
        pos = self._score_triples(batch_positives)
        neg = self._score_triples(batch_negatives)

        # if torch.isnan(neg).any():
        #     print("nan in neg dists")
        # if torch.isnan(pos).any():
        #     print("nan in pos dists")
        # if torch.isnan(neg - pos).any():
        #     print("nan in difference dists")
        # if torch.isnan(torch.sigmoid(neg - pos)).any():
        #     print("nan in sigmoid")
        # if torch.isnan(-torch.log(torch.sigmoid(neg - pos))).any():
        #     print("nan in logarithms")
        #
        # if torch.isinf(neg).any():
        #     print("isinf in neg dists")
        # if torch.isinf(pos).any():
        #     print("isinf in pos dists")
        # if torch.isinf(neg - pos).any():
        #     print("isinf in difference dists")
        # if torch.isinf(torch.sigmoid(neg - pos)).any():
        #     print("isinf in sigmoid")
        # if torch.isinf(-torch.log(torch.sigmoid(neg - pos))).any():
        #     print("isinf in logarithms")

        loss = torch.sum(-torch.log(torch.sigmoid(neg - pos) + 1e-15))
        # if torch.isnan(loss).any():
        #     print("nan in losses")
        # if torch.isinf(loss).any():
        #     print("isinf in losses")
        # if (torch.sigmoid(neg - pos) < 1e-45).any():
        #     print("too small sigmoid output")

        if self.reg_l:
            loss = loss + self.get_reg_loss(len(neg))
        return loss

    def _forward_nll(self, batch, targets):
        self._normalize()
        scores_1d = self._score_triples(batch)

        pos_mask = torch.tensor((targets == 1), dtype=torch.float, device=self.device).unsqueeze(1)
        scores_pos = scores_1d * pos_mask + (1 - scores_1d) * (1 - pos_mask)
        targets = torch.tensor(np.zeros(targets.shape[0]), dtype=torch.long, device=self.device)
        loss = self.criterion(torch.log(scores_pos), targets)

        # TODO: regularization once or for every element
        # todo: if here -- then once a batch -- normalized to batch size?
        if self.reg_l:
            loss = loss + self.get_reg_loss(len(targets))
        #print("reg loss: %0.2f"%loss)
        #print(positive_scores, negative_scores, loss, loss_)
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
        bs = len(relations)
        if self.region_dim < 3:
            if self.region_dim == 1:
                values = self.relation_regions(relations).view(-1, 1, 1)
            elif self.region_dim == 2:
                values = self.relation_regions(relations).view(-1, self.embedding_dim, 1)
            sigmas = torch.log(1 + torch.exp(values))
            matrix = (sigmas * torch.eye(self.embedding_dim, device=self.device))
        elif self.region_dim == 3:
            matrix = self.relation_regions(relations).view(-1, self.embedding_dim, self.embedding_dim)
        elif self.region_dim == 5:
            values = self.relation_regions(relations).view(bs, self.embedding_dim, 1)
            # x = 30
            # values = (values * 10.0**x).round() / 10.0**x
            # #values = torch.sigmoid(values) - 0.5
            matrix = torch.bmm(values, values.transpose(-2, -1))
            matrix += self.embedding_dim * torch.eye(self.embedding_dim, device=self.device).unsqueeze(0)
        else:
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

        # def isPSD(A, tol=1e-8):
        #     E = np.linalg.eigvalsh(A)
        #     return np.all(E > -tol)
        #
        # for i,m in enumerate(matrix):
        #     if not isPSD(m.view(self.embedding_dim, -1).detach().cpu()):
        #
        #         print("non psd matrix!")
        #         print(m)
        #         print(values[i])
        #         import pickle
        #         pickle.dump(m.cpu(), open('nonpsd.pkl', 'wb'))
        #         pickle.dump(values[i].cpu(), open('vec.pkl', 'wb'))
        #         exit(1)

        return matrix
