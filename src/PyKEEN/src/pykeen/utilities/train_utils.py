# -*- coding: utf-8 -*-

"""Utilities for training KGE models."""

import logging
import os, sys
import timeit
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
import pykeen.constants as pkc
import torch
import torch.optim as optim
from pykeen.kge_models import ConvE
from pykeen.utilities.evaluation_utils.metrics_computations import compute_metric_results
from sklearn.utils import shuffle
from torch.nn import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
sys.path.append('/mounts/Users/student/speranskaya/work/kge/region-kge/')
#from optimizer import Lookahead
from Lookahead import Lookahead # https://github.com/mgrankin/over9000/blob/master/train.py

__all__ = [
    'train_kge_model',
]

log = logging.getLogger(__name__)


def _split_list_in_batches(input_list, batch_size):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]


def train_kge_model(
        kge_model: Module,
        all_entities,
        all_relations,
        learning_rate,
        num_epochs,
        batch_size,
        test_batch_size,
        train_triples,
        train_types,
        val_triples,
        val_types,
        neg_val_triples,
        device,
        model_dir,
        es_metric=pkc.MEAN_RANK,
        neg_factor=1,
        single_pass=False,
        seed: Optional[int] = None,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[Module, List[float], Optional[List[float]]]:
    """Train the model."""
    return _train_basic_model(
        kge_model=kge_model,
        all_entities=all_entities,
        all_relations=all_relations,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        train_triples=train_triples,
        train_types=train_types,
        val_triples=val_triples,
        val_types=val_types,
        neg_val_triples=neg_val_triples,
        es_metric=es_metric,
        device=device,
        seed=seed,
        neg_factor=neg_factor,
        tqdm_kwargs=tqdm_kwargs,
        model_dir=model_dir
    )

def _train_basic_model(
        kge_model: Module,
        all_entities,
        all_relations,
        learning_rate,
        num_epochs,
        batch_size,
        test_batch_size,
        train_triples,
        train_types,
        val_triples,
        val_types,
        neg_val_triples,
        neg_factor,
        es_metric,
        device,
        model_dir,
        seed: Optional[int] = None,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[Module, List[float], List[float]]:
    """"""
    if seed is not None:
        np.random.seed(seed=seed)

    kge_model = kge_model.to(device)
    if kge_model.single_pass:
        batch_size = batch_size // (neg_factor + 1)
    else:
        batch_size = batch_size // neg_factor

    if kge_model.opt == 'SGD':
        base_optimizer = optim.SGD(kge_model.parameters(), lr=learning_rate)
    else:
        base_optimizer = optim.Adam(kge_model.parameters(), lr=learning_rate)
    optimizer = base_optimizer
    #optimizer = Lookahead(base_optimizer, k=10)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, min_lr=0.00001, patience=50, verbose=True)

    pid=os.getpid()
    log.debug(f'****running model on {device}, pid {pid}****')

    loss_per_epoch = []
    valloss_per_epoch = []
    metric_per_epoch = []

    num_train_triples = train_triples.shape[0]
    num_entities = all_entities.shape[0]

    if train_types is not None:
        train_data = np.concatenate([train_triples, train_types], axis=1)
        val_data = np.concatenate([val_triples, val_types], axis=1)
    else:
        train_data = train_triples
        val_data = val_triples

    #pos_data = torch.tensor(train_data[:, :3], device=device)
    #neg_data = torch.empty((0, 3), device=device).type_as(pos_data)

    # shuffle validation
    num_val_triples = len(val_data)
    indices = np.arange(num_val_triples)
    np.random.shuffle(indices)
    val_data = val_data[indices]
    val_idx = _split_list_in_batches(input_list=np.arange(len(val_data)), batch_size=test_batch_size)
    all_corrupted_subj = np.random.choice(np.arange(0, num_entities), size=num_val_triples)
    all_corrupted_obj  = np.random.choice(np.arange(0, num_entities), size=num_val_triples)

    start_training = timeit.default_timer()

    _tqdm_kwargs = dict(desc='Training epoch')
    if tqdm_kwargs:
        _tqdm_kwargs.update(tqdm_kwargs)

    waited = 0
    last_saved = -1
    last_dump = -100
    metric_name = 'MRR' if es_metric == pkc.MEAN_RANK else 'acc'

    for k in range(num_epochs):

        # log.debug("%f %f %f"%(kge_model.relation_regions.weight.max(),kge_model.relation_embeddings.weight.max(),kge_model.entity_embeddings.weight.max()))
        # log.debug("%f %f %f" % (kge_model.relation_regions.weight.min(), kge_model.relation_embeddings.weight.min(),
        #                         kge_model.entity_embeddings.weight.min()))
        # log.debug("%f %f %f" % (kge_model.relation_regions.weight.abs().min(), kge_model.relation_embeddings.weight.abs().min(),
        #                         kge_model.entity_embeddings.weight.abs().min()))
        kge_model.train()
        start = timeit.default_timer()

        # update non-gradient parameters
        #kge_model.update(pos_data, neg_data)

        indices = np.arange(num_train_triples)
        np.random.shuffle(indices)
        train_data = train_data[indices]
        train_batches = _split_list_in_batches(input_list=train_data, batch_size=batch_size)

        pos_data = torch.tensor(train_data[:, :3], device=device)
        neg_data = torch.empty((0, 3), device=device).type_as(pos_data)

        current_epoch_loss = 0.
        current_train_size, current_val_size = 0, 0

        for i, pos_batch in enumerate(train_batches):
            # log.info("batch %3d epoch %3d " % (i, k))
            pos_batch = pos_batch[:, 0:3]

            if neg_factor != 1:
                pos_batch = np.concatenate([pos_batch for _ in range(neg_factor)])

            current_batch_size = len(pos_batch)
            batch_subjs = pos_batch[:, 0:1]
            batch_relations = pos_batch[:, 1:2]
            batch_objs = pos_batch[:, 2:3]
            train_types = pos_batch[:, 3:]

            # TODO: config parameter corrupt relation
            if all_relations is not None:
                num_subj_corrupt = len(pos_batch) // 4
                num_obj_corrupt = len(pos_batch) // 4
                num_rel_corrupt = len(pos_batch) - num_obj_corrupt - num_subj_corrupt

                corrupted_relations_indices = np.random.choice(np.arange(0, len(all_relations)), size=num_rel_corrupt)
                corrupted_relations = np.reshape(all_relations[corrupted_relations_indices], newshape=(-1, 1))

                # assure that the corrupted relation is different from the initial triple
                corrupted_relations[corrupted_relations == batch_relations[-num_rel_corrupt]] += 1
                corrupted_relations[corrupted_relations >= len(all_relations)] //= len(all_relations)
                relation_based_corrupted_triples = np.concatenate(
                    [batch_subjs[-num_rel_corrupt:], corrupted_relations, batch_objs[-num_rel_corrupt:]], axis=1)

            else:
                num_subj_corrupt = len(pos_batch) // 2
                num_obj_corrupt = len(pos_batch) - num_subj_corrupt
                num_rel_corrupt = 0

            # todo: filter correct triples from neg batch
            corrupted_subj_indices = np.random.choice(np.arange(0, num_entities), size=num_subj_corrupt)
            corrupted_subjects = np.reshape(all_entities[corrupted_subj_indices], newshape=(-1, 1))
            subject_based_corrupted_triples = np.concatenate(
                [corrupted_subjects, batch_relations[:num_subj_corrupt], batch_objs[:num_subj_corrupt]], axis=1)

            corrupted_obj_indices = np.random.choice(np.arange(0, num_entities), size=num_obj_corrupt)
            corrupted_objects = np.reshape(all_entities[corrupted_obj_indices], newshape=(-1, 1))
            object_based_corrupted_triples = np.concatenate(
                [batch_subjs[num_subj_corrupt: -num_rel_corrupt if num_rel_corrupt else None],
                 batch_relations[num_subj_corrupt: -num_rel_corrupt if num_rel_corrupt else None],
                 corrupted_objects], axis=1)

            neg_batch = np.concatenate([subject_based_corrupted_triples, object_based_corrupted_triples], axis=0)
            if all_relations is not None:
                neg_batch = np.concatenate([neg_batch, relation_based_corrupted_triples], axis=0)
            neg_batch = torch.tensor(neg_batch, dtype=torch.long, device=device)

            #pos_batch = np.concatenate([pos_batch, pos_batch], axis=0)
            pos_batch = torch.tensor(pos_batch, dtype=torch.long, device=device)

            neg_data = torch.cat((neg_data, neg_batch))

            # Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old instance
            optimizer.zero_grad()

            #with torch.autograd.profiler.profile(enabled=() use_cuda= (device.type == 'cuda')) as prof:
            if kge_model.single_pass:
                batch = torch.cat((pos_batch, neg_batch))
                current_train_size += len(pos_batch) + len(neg_batch)
                target = torch.tensor([1]*len(pos_batch) + [0]*len(neg_batch), dtype=torch.float, device=device)
                loss = kge_model(batch, target)
            else:
                current_train_size += len(pos_batch)
                loss = kge_model(pos_batch, neg_batch)

            current_epoch_loss += loss.item()

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(kge_model.parameters(), 1)
            optimizer.step()
            kge_model._normalize()

            #with open("profiler.out", "w") as f:
            #    f.write(str(prof))

        kge_model.update(pos_data[:neg_data.shape[0]], neg_data)

        #log.debug('start evaluation')
        with torch.no_grad():

            # validation loss
            current_epoch_valloss = 0.

            for i, pos_idx in enumerate(val_idx):
                pos_batch = val_data[pos_idx]
                batch_subjs = pos_batch[:, 0:1]
                batch_relations = pos_batch[:, 1:2]
                batch_objs = pos_batch[:, 2:3]
                val_types = pos_batch[:, 3:]
                pos_batch = pos_batch[:, 0:3]

                num_subj_corrupt = len(pos_batch) // 2
                num_obj_corrupt = len(pos_batch) - num_subj_corrupt
                pos_batch = torch.tensor(pos_batch, dtype=torch.long, device=device)

                corrupted_subj_indices = pos_idx[:num_subj_corrupt]
                corrupted_subjects = np.reshape(all_corrupted_subj[corrupted_subj_indices], newshape=(-1, 1))
                subject_based_corrupted_triples = np.concatenate(
                    [corrupted_subjects, batch_relations[:num_subj_corrupt], batch_objs[:num_subj_corrupt]], axis=1)

                corrupted_obj_indices = pos_idx[:num_obj_corrupt]
                corrupted_objects = np.reshape(all_corrupted_obj[corrupted_obj_indices], newshape=(-1, 1))
                object_based_corrupted_triples = np.concatenate(
                    [batch_subjs[-num_obj_corrupt:], batch_relations[-num_obj_corrupt:], corrupted_objects], axis=1)

                neg_batch = np.concatenate([subject_based_corrupted_triples, object_based_corrupted_triples], axis=0)
                neg_batch = torch.tensor(neg_batch, dtype=torch.long, device=device)

                if kge_model.single_pass:
                    batch = torch.cat((pos_batch, neg_batch))
                    current_val_size += len(batch)
                    target = torch.tensor([1] * len(pos_batch) + [0] * len(neg_batch), dtype=torch.float,
                                          device=device)
                    loss = kge_model(batch, target)
                else:
                    current_val_size += len(pos_batch)
                    loss = kge_model(pos_batch, neg_batch)

                current_epoch_valloss += loss.item()

            stop = timeit.default_timer()

            #current_epoch_loss /= current_train_size
            #current_epoch_valloss /= current_val_size

            if k == 0 or ((k+1) % 5 == 0) or (k >= 50 and current_epoch_valloss < min(valloss_per_epoch)):
                # validation metric
                results = compute_metric_results(
                    metrics=[es_metric if es_metric != 'custom' else pkc.MEAN_RANK],
                    all_entities=all_entities,
                    kg_embedding_model=kge_model,
                    mapped_train_triples=train_triples,
                    mapped_pos_test_triples=val_triples,
                    mapped_neg_test_triples=neg_val_triples,
                    batch_size=test_batch_size,
                    device=device,
                    filter_neg_triples=False,
                    threshold_search=True
                )
                if es_metric == pkc.MEAN_RANK:
                    current_epoch_metric = results.mean_rank
                elif es_metric == pkc.HITS_AT_K:
                    current_epoch_metric = results.hits_at_k[10]
                elif es_metric == 'custom':
                    rank = results.mean_rank
                    hits = results.hits_at_k[10]
                    current_epoch_metric = (2 * rank * hits) / (rank + hits)
                else:
                    current_epoch_metric = results.accuracy
                metric_per_epoch.append(current_epoch_metric)
                log.debug("\ncurrent metric: {}  ---- rank/hits@10: {}".format(current_epoch_metric,
                                                                               (
                                                                               results.mean_rank, results.hits_at_k[10])
                                                                               if results.hits_at_k is not None else None))

                if current_epoch_metric > last_saved:
                    waited = 0
                    if model_dir:
                        # Save trained model
                        torch.save(
                            kge_model.state_dict(),
                            os.path.join(model_dir, 'best_model.pkl'),
                        )
                        # last_dump = k
                        log.debug('Saving the following model to disk:')
                    last_saved = current_epoch_metric
            waited += 1

            #scheduler.step(current_epoch_metric)
            #print('new normalized {} {}'.format(
            #    kge_model.entity_embeddings.weight.data.norm(2, dim=1).min(),
            #    kge_model.entity_embeddings.weight.data.norm(2, dim=1).max())
            #)

        # Track epoch loss
        loss_per_epoch.append(current_epoch_loss)
        valloss_per_epoch.append(current_epoch_valloss)
        log.info("Epoch {:3d} / {:4d} ({:3.1f}s):  {:0.3f} ---- {:0.3f} ---- {:0.2f} ---- {:3d}"
                 .format(k+1, num_epochs, stop - start, loss_per_epoch[-1], valloss_per_epoch[-1], metric_per_epoch[-1], waited))

        if waited >= 1000:
            break


    stop_training = timeit.default_timer()
    log.debug("training took %.2fs seconds", stop_training - start_training)

    # load back the best model
    if model_dir and num_epochs:
        kge_model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pkl')))

    return kge_model, loss_per_epoch, valloss_per_epoch, metric_per_epoch


def _train_conv_e_model(
        kge_model: ConvE,
        all_entities,
        learning_rate,
        num_epochs,
        batch_size,
        pos_triples, device,
        seed: Optional[int] = None) -> Tuple[ConvE, List[float]]:
    """"""
    if seed is not None:
        np.random.seed(seed=seed)

    kge_model = kge_model.to(device)

    optimizer = optim.Adam(kge_model.parameters(), lr=learning_rate)

    loss_per_epoch: List[float] = []

    log.info('****Run Model On %s****' % str(device).upper())

    num_pos_triples = pos_triples.shape[0]
    num_entities = all_entities.shape[0]

    start_training = timeit.default_timer()

    for epoch in range(num_epochs):
        indices = np.arange(num_pos_triples)
        np.random.shuffle(indices)
        pos_triples = pos_triples[indices]
        num_positives = batch_size // 2
        # TODO: Make sure that batch = num_pos + num_negs
        # num_negatives = batch_size - num_positives

        pos_batches = _split_list_in_batches(input_list=pos_triples, batch_size=num_positives)
        current_epoch_loss = 0.

        for i in range(len(pos_batches)):
            # TODO: Remove original subject and object from entity set
            pos_batch = pos_batches[i]
            current_batch_size = len(pos_batch)
            batch_subjs = pos_batch[:, 0:1]
            batch_relations = pos_batch[:, 1:2]
            batch_objs = pos_batch[:, 2:3]

            num_subj_corrupt = len(pos_batch) // 2
            num_obj_corrupt = len(pos_batch) - num_subj_corrupt
            pos_batch = torch.tensor(pos_batch, dtype=torch.long, device=device)

            corrupted_subj_indices = np.random.choice(np.arange(0, num_entities), size=num_subj_corrupt)
            corrupted_subjects = np.reshape(all_entities[corrupted_subj_indices], newshape=(-1, 1))
            subject_based_corrupted_triples = np.concatenate(
                [corrupted_subjects, batch_relations[:num_subj_corrupt], batch_objs[:num_subj_corrupt]], axis=1)

            corrupted_obj_indices = np.random.choice(np.arange(0, num_entities), size=num_obj_corrupt)
            corrupted_objects = np.reshape(all_entities[corrupted_obj_indices], newshape=(-1, 1))

            object_based_corrupted_triples = np.concatenate(
                [batch_subjs[num_subj_corrupt:], batch_relations[num_subj_corrupt:], corrupted_objects], axis=1)

            neg_batch = np.concatenate([subject_based_corrupted_triples, object_based_corrupted_triples], axis=0)

            neg_batch = torch.tensor(neg_batch, dtype=torch.long, device=device)

            batch = np.concatenate([pos_batch, neg_batch], axis=0)
            positive_labels = np.ones(shape=current_batch_size)
            negative_labels = np.zeros(shape=current_batch_size)
            labels = np.concatenate([positive_labels, negative_labels], axis=0)

            batch, labels = shuffle(batch, labels, random_state=seed)

            batch = torch.tensor(batch, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.float)

            # Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            optimizer.zero_grad()
            loss = kge_model(batch, labels)
            current_epoch_loss += (loss.item() * current_batch_size)

            loss.backward()
            optimizer.step()

        # log.info("Epoch %s took %s seconds \n" % (str(epoch), str(round(stop - start))))
        # Track epoch loss
        loss_per_epoch.append(current_epoch_loss / len(pos_triples))

    stop_training = timeit.default_timer()
    log.info("Training took %s seconds \n" % (str(round(stop_training - start_training))))

    return kge_model, loss_per_epoch
