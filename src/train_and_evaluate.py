import argparse
import json
import logging
import os
from datetime import datetime

import pykeen
import pykeen.constants as pkc

# LOAD SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str)
parser.add_argument('-o', '--outputdir', type=str)
parser.add_argument('-d', '--dataset', type=str)
args = parser.parse_args()

with open(args.config) as f:
    config = json.load(f)

if args.outputdir is not None:
    output_directory = args.outputdir
else:
    time = datetime.now()
    stamp = "{:04d}{:02d}{:02d}-{:02d}{:02d}".format(time.year, time.month, time.day, time.hour, time.minute)
    if config['kg_embedding_model_name'].lower() == 'region':
        output_directory = 'models/{}_{}_{}_dim{}_rad{}_regL{}_lr{}_loss-{}_margin{}_negfactor{}_epochs{}_esmetric-{}_bs{}'.format(
            args.dataset,
            config['kg_embedding_model_name'].lower() + config.get('region_type', ''),
            stamp,
            config['embedding_dim'], config['init_radius'],
            str(config['reg_lambda']).replace('.', ''),
            str(config['learning_rate']).replace('.', ''),
            config['loss_type'],
            config['margin_loss'],
            config.get('neg_factor', 1),
            config['num_epochs'],
            config['es_metric'],
            config['batch_size']
        )
    else:
        output_directory = 'models/{}_{}_dim{}'.format(config['kg_embedding_model_name'].lower(), stamp, config['embedding_dim'])
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('pykeen').setLevel(logging.DEBUG)

# TRAIN MODEL
results = pykeen.run(
    config=config,
    output_directory=output_directory,
)

# EVALUATE MODEL
model = results.trained_model
pipeline = results.pipeline
metrics = [pkc.MEAN_RANK, pkc.HITS_AT_K]  # [pkc.MEAN_RANK, pkc.HITS_AT_K, pkc.TRIPLE_PREDICTION]

# val_results = pipeline.evaluate(
#     model,
#     config['test_set_path'].replace('_200', ''),
#     # neg_test_path = '../../../data/fb15k-237/valid_neg.tsv',
#     metrics=metrics,
#     filter_neg_triples=False,
#     threshold_search=True,
#     single_threshold=False
# )
#
# json.dump(val_results['eval_summary'], open(os.path.join(output_directory, 'valid_evaluation_summary.json'), "w"))

val_results = pipeline.evaluate(
    model,
    config['test_set_path'].replace('_200', ''),
    # neg_test_path = '../../../data/fb15k-237/valid_neg.tsv',
    metrics=metrics,
    filter_neg_triples=True,
    threshold_search=True,
    single_threshold=False
)

json.dump(val_results['eval_summary'],
          open(os.path.join(output_directory, 'valid_evaluation_summary_filtered.json'), "w"))
