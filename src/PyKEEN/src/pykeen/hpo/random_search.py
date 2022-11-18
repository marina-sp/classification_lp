# -*- coding: utf-8 -*-

"""A hyper-parameter optimizer that uses random search."""

from collections import OrderedDict
import os

import numpy as np
import random
import torch
from torch.nn import Module
from tqdm import trange
from typing import Any, Dict, List, Optional, Union

import pykeen.constants as pkc
from pykeen.hpo.utils import HPOptimizer, HPOptimizerResult
from pykeen.kge_models import get_kge_model
from pykeen.utilities.evaluation_utils.metrics_computations import compute_metric_results
from pykeen.utilities.train_utils import train_kge_model

__all__ = [
    'RandomSearch',
]


class RandomSearch(HPOptimizer):
    """A hyper-parameter optimizer that uses random search."""

    def _sample_conv_e_params(self, hyperparams_dict) -> Dict[str, Any]:
        kg_model_config = OrderedDict()
        # Sample params which are dependent on each other

        hyperparams_dict = hyperparams_dict.copy()
        embedding_dimensions = hyperparams_dict[pkc.EMBEDDING_DIM]
        sampled_index = random.choice(range(len(embedding_dimensions)))
        kg_model_config[pkc.EMBEDDING_DIM] = hyperparams_dict[pkc.EMBEDDING_DIM][sampled_index]
        kg_model_config[pkc.CONV_E_HEIGHT] = hyperparams_dict[pkc.CONV_E_HEIGHT][sampled_index]
        kg_model_config[pkc.CONV_E_WIDTH] = hyperparams_dict[pkc.CONV_E_WIDTH][sampled_index]
        kg_model_config[pkc.CONV_E_KERNEL_HEIGHT] = hyperparams_dict[pkc.CONV_E_KERNEL_HEIGHT][sampled_index]
        kg_model_config[pkc.CONV_E_KERNEL_WIDTH] = hyperparams_dict[pkc.CONV_E_KERNEL_WIDTH][sampled_index]

        del hyperparams_dict[pkc.EMBEDDING_DIM]
        del hyperparams_dict[pkc.CONV_E_HEIGHT]
        del hyperparams_dict[pkc.CONV_E_WIDTH]
        del hyperparams_dict[pkc.CONV_E_KERNEL_HEIGHT]
        del hyperparams_dict[pkc.CONV_E_KERNEL_WIDTH]

        kg_model_config.update(self._sample_parameter_value(hyperparams_dict))

        return kg_model_config

    def optimize_hyperparams(
            self,
            mapped_train_triples,
            train_types,
            mapped_pos_test_triples,
            test_types,
            mapped_neg_test_triples,
            entity_to_id,
            rel_to_id,
            config,
            device,
            seed,
            model_dir: Optional[int] = None
    ) -> HPOptimizerResult:
        if seed is not None:
            torch.manual_seed(config[pkc.SEED])

        trained_kge_models: List[Module] = []
        epoch_losses: List[List[float]] = []
        epoch_val_losses: List[List[float]] = []
        epoch_metrics: List[List[float]] = []
        evaluation_criterion: List[float] = []
        entity_to_ids: List[Dict[int, str]] = []
        rel_to_ids: List[Dict[int, str]] = []
        models_params: List[Dict] = []
        eval_summaries: List = []
        search_summary: Dict[str, Dict[Union[str, int, float], List[float]]] = {}

        config = config.copy()
        max_iters = config[pkc.NUM_OF_HPO_ITERS]

        sample_fct = (
            self._sample_conv_e_params
            if config[pkc.KG_EMBEDDING_MODEL_NAME] == pkc.CONV_E_NAME else
            self._sample_parameter_value
        )
        es_metric = config.get('es_metric', 'custom')

        for _ in trange(max_iters, desc='HPO Iteration'):
            # Sample hyper-params
            kge_model_config: Dict[str, Any] = sample_fct(config)
            kge_model_config[pkc.NUM_ENTITIES]: int = len(entity_to_id)
            kge_model_config[pkc.NUM_RELATIONS]: int = len(rel_to_id)
            kge_model_config[pkc.SEED]: int = seed

            print(kge_model_config)
            # Configure defined model
            kge_model: Module = get_kge_model(config=kge_model_config)

            models_params.append(kge_model_config)
            entity_to_ids.append(entity_to_id)
            rel_to_ids.append(rel_to_id)

            all_entities = np.array(list(entity_to_id.values()))
            all_relations = np.array(list(rel_to_id.values()))

            try:
                trained_kge_model, epoch_loss, val_loss, epoch_metric = train_kge_model(
                    kge_model=kge_model,
                    all_entities=all_entities,
                    all_relations=all_relations if kge_model_config.get('corrupt_relations') else None,
                    learning_rate=kge_model_config[pkc.LEARNING_RATE],
                    num_epochs=kge_model_config[pkc.NUM_EPOCHS],
                    batch_size=kge_model_config[pkc.BATCH_SIZE],
                    test_batch_size=kge_model_config.get(pkc.TEST_BATCH_SIZE, kge_model_config[pkc.BATCH_SIZE]),
                    train_triples=mapped_train_triples,
                    train_types=train_types,
                    val_triples=mapped_pos_test_triples,
                    val_types=test_types,
                    neg_val_triples=mapped_neg_test_triples,
                    es_metric=kge_model_config.get('es_metric', 'custom'),
                    seed=seed,
                    device=device,
                    neg_factor=kge_model_config.get('neg_factor', 1),
                    single_pass=kge_model_config.get('single_pass', False),
                    tqdm_kwargs=dict(leave=False),
                    model_dir=os.path.join(os.path.dirname(model_dir), '_temp')
                )


                # Evaluate trained model
                metric_results = compute_metric_results(
                    metrics=[pkc.MEAN_RANK],
                    all_entities=all_entities,
                    kg_embedding_model=trained_kge_model,
                    mapped_train_triples=mapped_train_triples,
                    mapped_pos_test_triples=mapped_pos_test_triples,
                    mapped_neg_test_triples=mapped_neg_test_triples,
                    batch_size=kge_model_config['test_batch_size'],
                    device=device,
                    threshold_search=True,
                    filter_neg_triples=False
                )

                # TODO: Define HPO metric
                eval_summaries.append(metric_results)
                print(metric_results)

                trained_kge_models.append(trained_kge_model)
                epoch_losses.append(epoch_loss)
                epoch_val_losses.append(val_loss)
                epoch_metrics.append(epoch_metric)

                if es_metric == pkc.MEAN_RANK:
                    evaluation_value = metric_results.mean_rank
                elif es_metric == pkc.HITS_AT_K:
                    evaluation_value = metric_results.hits_at_k[10]
                elif es_metric == 'custom':
                    rank = metric_results.mean_rank
                    hits = metric_results.hits_at_k[10]
                    evaluation_value = (2 * rank * hits) / (rank + hits)
                else:
                    evaluation_value = metric_results.accuracy
                evaluation_criterion.append(evaluation_value)

                # save scores for parameters
                for param_name in kge_model.hyper_params:
                    search_summary.\
                        setdefault(param_name, {}).\
                        setdefault(kge_model_config[param_name], []).\
                        append(evaluation_value)

            except:
                #print(str(err))
                print("Model training failed.")

        index_of_max = int(np.argmax(a=evaluation_criterion))

        return (
            trained_kge_models[index_of_max],
            epoch_losses[index_of_max],
            epoch_val_losses[index_of_max],
            epoch_metrics[index_of_max],
            entity_to_ids[index_of_max],
            rel_to_ids[index_of_max],
            eval_summaries[index_of_max],
            models_params[index_of_max],
            search_summary,
        )

    @classmethod
    def run(cls,
            mapped_train_triples: np.ndarray,
            train_types: np.ndarray,
            mapped_pos_test_triples: np.ndarray,
            test_types: np.ndarray,
            mapped_neg_test_triples: np.ndarray,
            entity_to_id: Dict[int, str],
            rel_to_id: Dict[int, str],
            config: Dict,
            device,
            seed,
            model_dir) -> HPOptimizerResult:
        return cls().optimize_hyperparams(
            mapped_train_triples=mapped_train_triples,
            train_types=train_types,
            mapped_pos_test_triples=mapped_pos_test_triples,
            test_types=test_types,
            mapped_neg_test_triples=mapped_neg_test_triples,
            entity_to_id=entity_to_id,
            rel_to_id=rel_to_id,
            config=config,
            device=device,
            seed=seed,
            model_dir=model_dir
        )
