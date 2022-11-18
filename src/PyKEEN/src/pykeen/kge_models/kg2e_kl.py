# -*- coding: utf-8 -*-

"""Implementation of the TransE model."""

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.autograd
from torch.autograd import Function
from torch import nn
from torch.distributions import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal

from pykeen.kge_models.base import BaseModule, slice_triples

__all__ = [
    'KG2E',
    'KG2EConfig',
]

log = logging.getLogger(__name__)


@dataclass
class KG2EConfig:
    energy_function: str
    c_min: float
    c_max: float

    @classmethod
    def from_dict(cls, config: Dict) -> 'KG2EConfig':
        """Generate an instance from a dictionary."""
        return cls(
            energy_function=config['energy_function'],  # todo: pkc
            c_min=config['covariance_min'],
            c_max=config['covariance_max']
        )


class KG2E(BaseModule):
    """An implementation of KG2E Shizhu He 2015.

     This model considers a relation as a translation from the head to the tail entity.

    .. [borders2013] Bordes, A., *et al.* (2013). `Translating embeddings for modeling multi-relational data
                     <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`_
                     . NIPS.

    .. seealso::

       - Original implementation in C++: http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/code/cikm15_he_code.zip
    """

    model_name = 'kg2e'
    margin_ranking_loss_size_average: bool = False
    hyper_params = BaseModule.hyper_params + ['energy_function', 'covariance_max', 'covariance_min']

    def __init__(self, config: Dict) -> None:
        self.margin = config['margin_loss']
        super().__init__(config)
        config = KG2EConfig.from_dict(config)

        # Embeddings
        self.l_p_norm_entities = 2
        self.ent_mean = nn.Embedding(self.num_entities, self.embedding_dim)
        self.ent_vari = nn.Embedding(self.num_entities, self.embedding_dim)
        self.rel_mean = nn.Embedding(self.num_relations, self.embedding_dim)
        self.rel_vari = nn.Embedding(self.num_relations, self.embedding_dim)

        self.c_min = torch.tensor(config.c_min, dtype=torch.float, device=self.device)
        self.c_max = torch.tensor(config.c_max, dtype=torch.float, device=self.device)

        if config.energy_function == 'KL':
            self._compute_scores = KLDivergenceFunction.apply  #self._kl_divergence
        elif config.energy_function == 'EL':
            raise NotImplementedError('expected likelihood not implemented yet')
            self._compute_scores = self._exp_likelihood
        else:
            raise AttributeError('Unknown energy function.')

        self._initialize()

    def _initialize(self):
        embeddings_init_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(
            self.ent_mean.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )
        nn.init.uniform_(
            self.rel_mean.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )

        nn.init.uniform_(
            self.ent_vari.weight.data,
            a=self.c_min,
            b=self.c_max,
        )

        nn.init.uniform_(
            self.rel_vari.weight.data,
            a=self.c_min,
            b=self.c_max,
        )

    def _normalize(self):
        # Normalize means
        norms = torch.norm(self.ent_mean.weight, p=2, dim=1)
        mask = norms > 1
        self.ent_mean.weight.data[mask] = (self.ent_mean.weight.data / norms.unsqueeze(1))[mask]
        norms = torch.norm(self.rel_mean.weight, p=2, dim=1)
        mask = norms > 1
        self.rel_mean.weight.data[mask] = (self.rel_mean.weight.data / norms.unsqueeze(1))[mask]

        # Check covariance ranges
        self.ent_vari.weight.data.clamp_(min=self.c_min, max=self.c_max)
        self.rel_vari.weight.data.clamp_(min=self.c_min, max=self.c_max)

    def predict(self, triples):
        #self._normalize()
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def forward(self, batch_positives, batch_negatives):
        #self._normalize()

        positive_scores = self._score_triples(batch_positives)
        negative_scores = self._score_triples(batch_negatives)
        loss = self._compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return loss

    def _compute_loss(self, positive_scores, negative_scores):
        y = np.repeat([-1], repeats=positive_scores.shape[0])
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        # Scores for the positive and negative triples
        #positive_scores = torch.tensor(positive_scores, dtype=torch.float, device=self.device)
        #negative_scores = torch.tensor(negative_scores, dtype=torch.float, device=self.device)

        ## new
        loss = torch.max(torch.tensor(0., device=self.device),
                         self.margin + positive_scores - negative_scores).sum()

        ## todo: test order of pos and neg
        #loss = self.criterion(positive_scores, negative_scores, y)
        return loss

    def _score_triples(self, triples):  # compute energy
        head_embeddings, relation_embeddings, tail_embeddings = self._get_triple_embeddings(triples)
        h_mean, h_vari = head_embeddings
        r_mean, r_vari = relation_embeddings
        t_mean, t_vari = tail_embeddings
        scores = self._compute_scores(h_mean, h_vari, r_mean, r_vari, t_mean, t_vari)
        return scores


    # rewrite as PyTorch function class
    def _kl_divergence(self, h_mean, h_vari, r_mean, r_vari, t_mean, t_vari):
        """Compute the KL-based energy of a triple based on the head, relation, and tail embeddings.

        :param head_embeddings: embeddings of head entities of dimension batchsize x embedding_dim
        :param relation_embeddings: emebddings of relation embeddings of dimension batchsize x embedding_dim
        :param tail_embeddings: embeddings of tail entities of dimension batchsize x embedding_dim
        :return: Tensor of dimension batch_size containing the scores for each batch element
        """

        batch_size = len(h_mean)

        # entity distribution
        mean = h_mean - t_mean
        vari = h_vari + t_vari
        cov = (torch.eye(self.embedding_dim, device=self.device) *
               vari.view(
                   batch_size, -1, 1, self.embedding_dim
               )
               ).squeeze(1)
        entity_dist = MultivariateNormal(mean, cov)

        # relation distribution
        r_cov = (torch.eye(self.embedding_dim, device=self.device) *
                 r_vari.view(
                     batch_size, -1, 1, self.embedding_dim
                 )
                 ).squeeze(1)

        # todo: find batch based distribution and divergence calculation
        relation_dist = MultivariateNormal(r_mean, r_cov)

        try:
            distances = kl_divergence(entity_dist, relation_dist)
        except:
            print(mean, vari, r_mean, r_vari)
            distances = torch.randn((batch_size,))

        # define parameter updates
        #r_vari_inv = r_vari.inverse()
        #delta = r_vari_inv * (r_mean + t_mean - h_mean)

        #r_mean.grad.data = delta

        return distances

    def _exp_likelihood(self, head_embeddings, relation_embeddings, tail_embeddings):
        """Compute the EL-based energy of a triple based on the head, relation, and tail embeddings.

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
        return (
            self.rel_mean(relations).view(-1, self.embedding_dim),
            self.rel_vari(relations).view(-1, self.embedding_dim)
        )

    def _get_entity_embeddings(self, entities):
        return (
            self.ent_mean(entities).view(-1, self.embedding_dim),
            self.ent_vari(entities).view(-1, self.embedding_dim)
        )


class KLDivergenceFunction(Function):
    @staticmethod
    def forward(ctx, h_mean, h_vari, r_mean, r_vari, t_mean, t_vari):
        """Compute the KL-based energy of a triple based on the head, relation, and tail embeddings.

        :param head_embeddings: embeddings of head entities of dimension batchsize x embedding_dim
        :param relation_embeddings: emebddings of relation embeddings of dimension batchsize x embedding_dim
        :param tail_embeddings: embeddings of tail entities of dimension batchsize x embedding_dim
        :return: Tensor of dimension batch_size containing the scores for each batch element
        """
        dim = h_mean.shape[-1]

        # entity distribution
        e_mean = h_mean - t_mean
        e_vari = h_vari + t_vari

        ctx.save_for_backward(e_mean, e_vari, r_mean, r_vari)

        # define supporting parts
        #print('r vari == 0', (r_vari == 0).any())
        r_vari_inv = 1 / r_vari
        det_r = r_vari.double().prod(-1)
        det_e = e_vari.double().prod(-1)
        #print('e_vari', (e_vari == float('inf')).any(), (e_vari == - float('inf')).any())
        #print('det_E', (e_vari[det_e == float('inf')]))
        trace = (r_vari_inv * e_vari).sum(-1)
        means = (e_mean - r_mean)

        distances = trace

        distances += (means * r_vari_inv).unsqueeze(-1).transpose(-1, -2).bmm(means.unsqueeze(-1)).squeeze(-1).squeeze(
            -1)

        distances -= torch.log(det_e / det_r).float() + dim
        #print('det_E, det_R', det_e[distances == - float('inf')], det_r[distances == - float('inf')], torch.log(det_e / det_r)[distances == - float('inf')])
        #print('dist == - inf', (distances == - float('inf')).any())

        distances /= 2
        return distances

    @staticmethod
    def backward(ctx, grad_output):
        e_mean, e_cov, r_mean, r_cov = ctx.saved_tensors

        # define supporting parts
        e_cov_inv = 1 / e_cov
        r_cov_inv = 1 / r_cov
        delta = r_cov_inv * (r_mean - e_mean)

        # define relative derivates
        dE_dcovR = 1 / 2 * (- (r_cov_inv * e_cov * r_cov_inv) - delta * delta + r_cov_inv)
        dE_dcovE = 1 / 2 * (r_cov_inv - e_cov_inv)

        grad_output = grad_output.unsqueeze(-1)
        return (
            grad_output * (-delta),  # head mean
            grad_output * dE_dcovE,  # head cov
            grad_output * delta,  # rel  mean
            grad_output * dE_dcovR,  # rel  cov
            grad_output * delta,  # tail mean
            grad_output * dE_dcovE,  # tail cov
        )
