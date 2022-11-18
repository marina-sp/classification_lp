# Project repo for the paper [Ranking vs. Classifying: Measuring Knowledge Base Completion Quality](https://arxiv.org/abs/2102.06145)
The backbone of the model training code is an open-source framework [PyKeen](https://github.com/pykeen/pykeen). The underlying framework version is 1.9.0 as seen in [current Readme at the time of code development](https://github.com/marina-sp/classification_lp/blob/master/src/PyKEEN/README.rst).

## FB14k-QAQ dataset for query-based evaluation of link prediction in the classification setting
The dataset is availabe under `FB14k-QAQ/data`.

The `train.txt` file contains the training split of the triples in a standard tab-separated format with three columns for the triples.

The evaluation files contain triples grouped into queries, s.t. the answers for the specific positions (subj or obj) are represented together as a list. E.g. the `valid_obj.txt` file contains three tab-separated columns with queries for the object position, where the column order is "query entity" - "relation" - "set of answer entities". For a `*_subj.txt` file the order is reversed: "set of answer entities" comes first, followed by "relation"  and "query entity". 
Note, that the sets can be empty due to the structure of the dataset. 

The script used for the creation you can find under `FB14k-QAQ`.

## Model training
Custom modifications to the framework have been created for the [implementation of the Region model](https://github.com/marina-sp/classification_lp/blob/master/src/PyKEEN/src/pykeen/kge_models/region.py) (TransE variation introduced in our paper). 

The majority of the model parameters and training option are specified in a separate config file (see example [here](https://github.com/marina-sp/classification_lp/blob/master/src/configs/region.config)).
There is also an examplary [training script](https://github.com/marina-sp/classification_lp/blob/master/src/train_and_evaluate.py which outlines how to train and evaluate a single model. There is also a [jupyter notebook](https://github.com/marina-sp/classification_lp/blob/master/src/PyKEEN/notebooks/train_and_evaluate/Train%20and%20Evaluate%20Region%20on%20FB15K-217.ipynb) that shows how to train or load a single model, and perform multiple evaluations on it. The jupyter notebook also contains a full config example. 

**Note on examplary scripts**. Please be aware that the provided scripts take data from the auxillary directory specified in the configs, and not from the root 'FB14k-QAQ' directory with the most recent dataset version. 

**Note on general code structure**. Specific model implementations as well as their config classes inherit from the [Base model class](https://github.com/marina-sp/classification_lp/blob/master/src/PyKEEN/src/pykeen/kge_models/base.py). The model class contains main and universal attributes used across different model types, such as *thresholds* that are introduced at the Base class level.


The high-level training process is regulated in the Pipeline class, which maps the config parameters to the actual structure of the training process. Main functions of interest are 'run' and ['evaluate'](https://github.com/marina-sp/classification_lp/blob/be27bb3bc827ff4fe5710c730dea44f528b1a269/src/PyKEEN/src/pykeen/utilities/pipeline.py#L210).


## Evaluation scripts 
As opposed to the commonly used ranking metris, the key evaluation features relevant to the classification settings are (with the links to the relevant files provided for each of the key changes):

1. corresponding dataset construction and loading to provide the foundation for classification (s. above)

2. integrated pipeline for F1 score calculation.

[metrics_computation.py#L358](https://github.com/marina-sp/classification_lp/blob/master/src/PyKEEN/src/pykeen/utilities/evaluation_utils/metrics_computations.py#L358) marks the point where the additional "triple prediction" evaluation happens if the corresponding evaluation type is specified in the [training script](https://github.com/marina-sp/classification_lp/blob/master/src/train_and_evaluate.py#L54).
  
3. [threshold tuning](https://github.com/marina-sp/classification_lp/blob/master/src/PyKEEN/src/pykeen/utilities/evaluation_utils/metrics_computations.py#L398)
 for the binary decision. There are two possibilities to do so:
-  search for a one joint threshold to be used for decisions for all classes (the cut is made at the same score for triples with different relations)
-  separate thresholds must be found for each relation (i.e. threshold search happens multiple times, one time for each triple group exhibiting one relation)
The model attribute 'single_threshold' is the fallback value for the choice of tuning type if nothing explicit was provided as an argument ('True' for the first single aka joint option). See [Region model class](https://github.com/marina-sp/classification_lp/blob/master/src/PyKEEN/src/pykeen/kge_models/region.py#L79) as an example of model level instantiation and [training script](https://github.com/marina-sp/classification_lp/blob/master/src/train_and_evaluate.py#L68) for passing the arguments to the evaluation function directly. The type of threshold tuning can not be specified through the configuration file. 

Applying thresholds is an obligatory step for the classification evaluation. If the tuning is disabled, default thresholds can be applied if set in the model initialisation.


