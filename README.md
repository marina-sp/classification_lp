## FB14k-QAQ dataset for query-based evaluation of link prediction in the classification setting
The dataset is availabe under `FB14k-QAQ/data`.

The `train.txt` file contains the training split of the triples in a standard tab-separated format with three columns for the triples.

The evaluation files contain triples grouped into queries, s.t. the answers for the specific positions (subj or obj) are represented together as a list. E.g. the `valid_obj.txt` file contains three tab-separated columns with queries for the object position, where the column order is "query entity" - "relation" - "set of answer entities". For a `*_subj.txt` file the order is reversed: "set of answer entities" comes first, followed by "relation"  and "query entity". 
Note, that the sets can be empty due to the structure of the dataset. 

The script used for the creation you can find under `FB14k-QAQ`.

## Evaluation scripts 
TBD

## Model training
TBD
