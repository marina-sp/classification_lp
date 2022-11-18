# -*- coding: utf-8 -*-

"""Script to compute mean rank and hits@k."""

import logging
import timeit
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np
import pykeen.constants as pkc
import torch
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm_notebook

log = logging.getLogger(__name__)

DEFAULT_HITS_AT_K = [1, 3, 5, 10]


def _hash_triples(triples: Iterable[Hashable]) -> int:
    """Hash a list of triples."""
    return hash(tuple(triples))


def update_hits_at_k_(
        hits_at_k_values: Dict[int, List[float]],
        rank_of_positive: int
) -> None:
    """Update the Hits@K dictionary for two values."""
    for k, values in hits_at_k_values.items():
        if rank_of_positive < k:
            values.append(1.0)
        else:
            values.append(0.0)

def _create_corrupted_triples(triple, all_entities, device):
    candidate_entities_subject_based = np.delete(arr=all_entities, obj=triple[0:1])
    candidate_entities_subject_based = np.reshape(candidate_entities_subject_based, newshape=(-1, 1))
    candidate_entities_object_based = np.delete(arr=all_entities, obj=triple[2:3])
    candidate_entities_object_based = np.reshape(candidate_entities_object_based, newshape=(-1, 1))

    # Extract current test tuple: Either (subject,predicate) or (predicate,object)
    tuple_subject_based = np.reshape(a=triple[1:3], newshape=(1, 2))
    tuple_object_based = np.reshape(a=triple[0:2], newshape=(1, 2))

    # Copy current test tuple
    tuples_subject_based = np.repeat(a=tuple_subject_based,
                                     repeats=candidate_entities_subject_based.shape[0],
                                     axis=0)
    tuples_object_based = np.repeat(a=tuple_object_based,
                                    repeats=candidate_entities_object_based.shape[0],
                                    axis=0)

    corrupted_subject_based = np.concatenate([
        candidate_entities_subject_based,
        tuples_subject_based], axis=1)

    corrupted_object_based = np.concatenate([
        tuples_object_based,
        candidate_entities_object_based], axis=1)

    return corrupted_subject_based, corrupted_object_based


def _filter_corrupted_triples(
        corrupted,
        all_pos_triples_hashed,
):
    # TODO: Check - split to subj/obj
    corrupted_hashed = np.apply_along_axis(_hash_triples, 1, corrupted)
    mask = np.in1d(corrupted_hashed, all_pos_triples_hashed, invert=True)
    mask = np.where(mask)[0]

    if mask.size == 0:
        raise Exception("User selected filtered metric computation, but all corrupted triples exists"
                        "also as positive triples.")

    return corrupted[mask]


def _compute_filtered_rank(
        kg_embedding_model,
        score_of_positive,
        corrupted,
        batch_size,
        device,
        all_pos_triples_hashed,
) -> Tuple[int, int]:
    """

    :param kg_embedding_model:
    :param score_of_positive:
    :param corrupted_subject_based:
    :param corrupted_object_based:
    :param device:
    :param all_pos_triples_hashed:
    """
    filtered_corrupted = _filter_corrupted_triples(
        corrupted=corrupted,
        all_pos_triples_hashed=all_pos_triples_hashed)

    return _compute_rank_(
        kg_embedding_model=kg_embedding_model,
        score_of_positive=score_of_positive,
        corrupted=filtered_corrupted,
        batch_size=batch_size,
        device=device,
        all_pos_triples_hashed=all_pos_triples_hashed,
    )


def _compute_rank(
        kg_embedding_model,
        score_of_positive,
        corrupted_subject_based,
        corrupted_object_based,
        batch_size,
        device,
        all_pos_triples_hashed=None,
) -> Tuple[int, int]:
    """

    :param kg_embedding_model:
    :param score_of_positive:
    :param corrupted_subject_based:
    :param corrupted_object_based:
    :param device:
    :param all_pos_triples_hashed: This parameter isn't used but is necessary for compatability
    """
    corrupted_scores = np.ndarray((len(corrupted_subject_based)+len(corrupted_object_based),1))
    #print("num of corrupted: ", len(corrupted_scores), corrupted_scores)
    for i, batch in enumerate(_split_list_in_batches(
            torch.cat([corrupted_subject_based, corrupted_object_based], dim=0),
            batch_size)):
        corrupted_scores[i * batch_size: (i + 1) * batch_size] = kg_embedding_model.predict(batch).reshape(-1,1)

    scores_of_corrupted_subjects = corrupted_scores[:len(corrupted_subject_based)]
    scores_of_corrupted_objects = corrupted_scores[len(corrupted_subject_based):]

    scores_subject_based = np.append(arr=scores_of_corrupted_subjects, values=score_of_positive)
    index_of_pos_subject_based = scores_subject_based.size - 1

    scores_object_based = np.append(arr=scores_of_corrupted_objects, values=score_of_positive)
    index_of_pos_object_based = scores_object_based.size - 1

    _, sorted_score_indices_subject_based = torch.sort(torch.tensor(scores_subject_based, dtype=torch.float),
                                                       descending=kg_embedding_model.prob_mode)
    sorted_score_indices_subject_based = sorted_score_indices_subject_based.cpu().numpy()

    _, sorted_score_indices_object_based = torch.sort(torch.tensor(scores_object_based, dtype=torch.float),
                                                      descending=kg_embedding_model.prob_mode)
    sorted_score_indices_object_based = sorted_score_indices_object_based.cpu().numpy()

    # Get index of first occurrence that fulfills the condition
    rank_of_positive_subject_based = np.where(sorted_score_indices_subject_based == index_of_pos_subject_based)[0][0]
    rank_of_positive_object_based = np.where(sorted_score_indices_object_based == index_of_pos_object_based)[0][0]

    return (
        rank_of_positive_subject_based,
        rank_of_positive_object_based,
    )

def _compute_rank_(
        kg_embedding_model,
        score_of_positive,
        corrupted,
        batch_size,
        device,
        all_pos_triples_hashed=None,
) -> Tuple[int, int]:
    """

    :param kg_embedding_model:
    :param score_of_positive:
    :param corrupted_subject_based:
    :param corrupted_object_based:
    :param device:
    :param all_pos_triples_hashed: This parameter isn't used but is necessary for compatability
    """
    corrupted = torch.tensor(corrupted, dtype=torch.long, device=device)

    corrupted_scores = np.ndarray((len(corrupted), 1))
    #print("num of corrupted: ", len(corrupted_scores), corrupted_scores)
    for i, batch in enumerate(_split_list_in_batches(corrupted, batch_size)):
        corrupted_scores[i * batch_size: (i + 1) * batch_size] = kg_embedding_model.predict(batch).reshape(-1,1)

    scores = np.append(arr=corrupted_scores, values=score_of_positive)
    index_of_pos = scores.size - 1

    _, sorted_score_indices = torch.sort(torch.tensor(scores, dtype=torch.float),
                                         descending=kg_embedding_model.prob_mode)
    sorted_score_indices = sorted_score_indices.cpu().numpy()

    # Get index of first occurrence that fulfills the condition
    rank_of_positive = np.where(sorted_score_indices == index_of_pos)[0][0]

    return (
        rank_of_positive
    )

def _split_list_in_batches(input_list, batch_size):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

@dataclass
class MetricResults:
    """Results from computing metrics."""

    mean_rank: float
    hits_at_k: Dict[int, float]
    precision: float
    recall: float
    accuracy: float
    fscore: float


def compute_metric_results(
        metrics,
        all_entities,
        kg_embedding_model,
        mapped_train_triples,
        mapped_pos_test_triples,
        mapped_neg_test_triples,
        batch_size,
        device,
        threshold_search=False,
        filter_neg_triples=False,
        return_subj_obj=False,
        single_threshold=None,
        ks: Optional[List[int]] = None
) -> MetricResults:
    """Compute the metric results.

    :param metrics:
    :param all_entities:
    :param kg_embedding_model:
    :param mapped_train_triples:
    :param mapped_pos_test_triples:
    :param mapped_neg_test_triples:
    :param device:
    :param filter_neg_triples:
    :param ks:
    :return:
    """
    start = timeit.default_timer()

    kg_embedding_model = kg_embedding_model.eval()
    kg_embedding_model = kg_embedding_model.to(device)

    results = MetricResults(
        mean_rank=None,
        hits_at_k=None,
        precision=None,
        recall=None,
        accuracy=None,
        fscore=None
    )

    # todo: names to constants
    if pkc.MEAN_RANK in metrics or pkc.HITS_AT_K in metrics:
        subj_ranks: List[int] = []
        obj_ranks: List[int] = []
        subj_hits = {
            k: []
            for k in (ks or DEFAULT_HITS_AT_K)
        }
        obj_hits = {
            k: []
            for k in (ks or DEFAULT_HITS_AT_K)
        }

        all_pos_triples = np.concatenate([mapped_train_triples, mapped_pos_test_triples], axis=0)
        all_pos_triples_hashed = np.apply_along_axis(_hash_triples, 1, all_pos_triples)

        compute_rank_fct: Callable[..., Tuple[int, int]] = (
            _compute_filtered_rank
            if filter_neg_triples else
            _compute_rank_
        )

        all_scores = np.ndarray((len(mapped_pos_test_triples), 1))

        all_triples = torch.tensor(mapped_pos_test_triples, dtype=torch.long, device=device)

        # predict all triples
        for i, batch in enumerate(_split_list_in_batches(all_triples, batch_size)):
            predictions = kg_embedding_model.predict(batch).reshape(-1, 1)
            #print(predictions.shape)
            all_scores[i*batch_size: (i+1)*batch_size] = predictions

        # Corrupt triples
        success = tqdm_notebook(total=len(mapped_pos_test_triples)) if len(mapped_pos_test_triples) > 1000 else None
        for i, pos_triple in enumerate(mapped_pos_test_triples):
            corrupted_subject_based, corrupted_object_based = _create_corrupted_triples(
                triple=pos_triple,
                all_entities=all_entities,
                device=device,
            )

            rank_of_positive_subject_based = compute_rank_fct(
                kg_embedding_model=kg_embedding_model,
                score_of_positive=all_scores[i],
                corrupted=corrupted_subject_based,
                batch_size=batch_size,
                device=device,
                all_pos_triples_hashed=all_pos_triples_hashed,
            )

            rank_of_positive_object_based = compute_rank_fct(
                kg_embedding_model=kg_embedding_model,
                score_of_positive=all_scores[i],
                corrupted=corrupted_object_based,
                batch_size=batch_size,
                device=device,
                all_pos_triples_hashed=all_pos_triples_hashed,
            )

            subj_ranks.append(1 / (rank_of_positive_subject_based+1))
            obj_ranks.append(1 / (rank_of_positive_object_based+1))

            # Compute hits@k for k in {1,3,5,10}
            update_hits_at_k_(
                subj_hits,
                rank_of_positive=rank_of_positive_subject_based
            )

            update_hits_at_k_(
                obj_hits,
                rank_of_positive=rank_of_positive_object_based,
            )

            if success:
                success.update(1)

        results.mean_rank = float(np.mean(subj_ranks + obj_ranks))


        results.hits_at_k = {
            k: np.mean(subj_hits[k] + obj_hits[k])
            for k in DEFAULT_HITS_AT_K
        }

        if return_subj_obj:
            subj_mean = float(np.mean(subj_ranks))
            obj_mean = float(np.mean(obj_ranks))

            subj_hits = {
                k: np.mean(values)
                for k, values in subj_hits.items()
            }

            obj_hits = {
                k: np.mean(values)
                for k, values in obj_hits.items()
            }


    if pkc.TRIPLE_PREDICTION in metrics:
        if single_threshold is None:
            single_threshold = kg_embedding_model.single_threshold

        if mapped_neg_test_triples is None:
            log.info("No negative test triples specified for the triple prediciton task.")
            all_test_triples = mapped_pos_test_triples
        else:
            all_test_triples = np.concatenate([mapped_pos_test_triples, mapped_neg_test_triples])

        y_true = [1]*len(mapped_pos_test_triples) + [0]*len(mapped_neg_test_triples)
        y_true = np.array(y_true)

        # predict all triples
        eval_scores = np.ndarray((len(all_test_triples), 1))
        all_test_triples = torch.tensor(all_test_triples, dtype=torch.long, device=device)

        for i, batch in enumerate(_split_list_in_batches(all_test_triples, batch_size)):
            predictions = kg_embedding_model.predict(batch).reshape(-1, 1)
            eval_scores[i * batch_size: (i + 1) * batch_size] = predictions

        # shuffle
        n_objects = len(all_test_triples)
        shuffle_idx = torch.randperm(n_objects)
        y_true = y_true[shuffle_idx]
        eval_scores = eval_scores[shuffle_idx]
        all_test_triples = all_test_triples.cpu().detach().numpy()[shuffle_idx]

        if threshold_search:
            #search_scores = eval_scores[:n_objects // 2]
            #y_search = y_true[:n_objects // 2]
            #search_relations = all_test_triples[:n_objects // 2, 1]
            #eval_scores = eval_scores[n_objects // 2:]
            #y_true = y_true[n_objects // 2:]
            #eval_relations = all_test_triples[n_objects // 2:, 1]

            search_scores = eval_scores
            y_search = y_true
            search_relations = all_test_triples[:, 1]

            if single_threshold:
                kg_embedding_model.relation_thresholds[:] = find_threshold_for_data(
                    search_scores,
                    y_search,
                    kg_embedding_model.prob_mode)
            else:
                kg_embedding_model.relation_thresholds[:] = search_scores.mean()
                for r_id in set(search_relations):
                    kg_embedding_model.relation_thresholds[r_id] = find_threshold_for_data(
                        search_scores[search_relations == r_id],
                        y_search[search_relations == r_id],
                        kg_embedding_model.prob_mode
                    )
        else:
            eval_relations = all_test_triples[:, 1]

            y_pred = predict_with_thresholds(eval_scores, eval_relations,
                                             kg_embedding_model.relation_thresholds, kg_embedding_model.prob_mode)
            results.precision = precision_score(y_true, y_pred)
            results.recall = recall_score(y_true, y_pred)
            results.accuracy = accuracy_score(y_true, y_pred)
            results.fscore = f1_score(y_true, y_pred)

            log.info('Best accuracy {:0.2f} achieved on threshold: {:0.2f}'.format(results.accuracy, kg_embedding_model.relation_thresholds.mean()))

    stop = timeit.default_timer()
    log.debug("Evaluation took %.2fs seconds", stop - start)

    if return_subj_obj:
        return results, subj_mean, obj_mean, subj_hits, obj_hits
    else:
        return results

def find_threshold_for_data(scores, y_true, prob_mode):
    best_acc = -1
    step = (max(scores) * 1.1 - min(scores)) / 100
    for th in np.arange(min(scores), max(scores) + step, step):
        y = 1 * (scores >= th) if prob_mode else 1 * (scores <= th)
        acc = accuracy_score(y_true, y)
        if acc > best_acc:
            best_acc = acc
            threshold = th
    return threshold

def predict_with_thresholds(scores, relations, thresholds, prob_mode):
    y_pred = np.ndarray((len(scores), 1), dtype=int)
    for r_id in set(relations):
        mask = relations == r_id
        if mask.any():
            th = thresholds[r_id]
            y_pred[mask] = 1 * (scores[mask] >= th) if prob_mode else 1 * (scores[mask] <= th)
    return y_pred
