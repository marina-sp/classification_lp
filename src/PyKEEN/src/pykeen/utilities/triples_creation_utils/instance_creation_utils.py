# -*- coding: utf-8 -*-

import logging
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    'create_mapped_triples',
    'create_mappings',
]

log = logging.getLogger(__name__)


def create_mapped_triples(
        triples: np.ndarray,
        entity_label_to_id: Optional[Dict[str, int]] = None,
        relation_label_to_id: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int], np.ndarray]:
    """"""
    if entity_label_to_id is None or relation_label_to_id is None:
        entity_label_to_id, relation_label_to_id = create_mappings(triples)

    subject_column = np.vectorize(entity_label_to_id.get)(triples[:, 0:1])
    relation_column = np.vectorize(relation_label_to_id.get)(triples[:, 1:2])
    object_column = np.vectorize(entity_label_to_id.get)(triples[:, 2:3])
    relation_type_column = np.array(
        [[1] if rel=='<instance_of>' else [2] if rel=='<subclass_of>' else [0]
         for rel in triples[:, 1:2]])
    triples_of_ids = np.concatenate([subject_column, relation_column, object_column, relation_type_column], axis=1)

    triples_of_ids = np.array(triples_of_ids, dtype=np.long)
    # Note: Unique changes the order
    triples_of_ids = np.unique(ar=triples_of_ids, axis=0)
    types_of_triples = triples_of_ids[:, 3:4]
    triples_of_ids = triples_of_ids[:, 0:3]
    return triples_of_ids, entity_label_to_id, relation_label_to_id, types_of_triples


def create_mappings(triples: np.ndarray) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Assign entities and relations numerical indices."""
    entities = np.unique(np.ndarray.flatten(np.concatenate([triples[:, 0:1], triples[:, 2:3]])))
    entity_label_to_id = {
        entity_label: entity_id
        for entity_id, entity_label in enumerate(entities)
    }

    relations = np.unique(np.ndarray.flatten(triples[:, 1:2]).tolist())
    relation_label_to_id = {
        relation_label: relation_id
        for relation_id, relation_label in enumerate(relations)
    }

    return entity_label_to_id, relation_label_to_id
