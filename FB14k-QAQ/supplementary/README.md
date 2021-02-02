
The files here help assign the queries from validation and test files to one of the set *C*, *I* or *F*. *I* is 0, *C* is 1. If your query is not among these files, it means it is fake.

The `grouping_answers_subj.csv` file contains queries from `data/*_subj.txt`. First column contains the query entity (object in this case), followed by the query relation and an integer in the third and last column. The interger is always 0 or 1 and indicates the query set.

Example:
Following lines are available in the `grouping_answers_subj.csv` file (line numbers are not preserved):

```
[1] /m/04cr6qv,/base/popstra/location/vacationers./base/popstra/vacation_choice/vacationer,0
[2] /m/04cr6qv,/music/genre/artists,1
[3] /m/04cr6qv,/music/instrument/instrumentalists,0
[4] /m/04cr6qv,/music/record_label/artist,0
```

If you then look into the `data/valid_subj.txt` file, and search for the query entity, you will find these corresponding lines:

```
[952]	/music/record_label/artist	/m/04cr6qv
[4745]  /m/04cr6qv ...ommitted for readability...  /people/person/gender	/m/05zppz
[12393] 	/location/statistical_region/gdp_nominal./measurement_unit/dated_money_value/currency	/m/04cr6qv
```
For 952 query it can be derived from line[4] of the supplementary file, that it has lost at least one answer during entity removal process.
For query 4745 the entity of interest is in the answer position. To find out the type for this query, we would have to look for a line starting with `/m/05zppz,/people/person/gender` in the  `grouping_answers_subj.csv` file.
For query 123393 there is no relevant record in the grouping file, meaning the query is a fake one.

We did not find queries ocurring in lines[1,2,3] in the validation file. Since the query grouping files are shared between all evaluation queries, they come from the test split and can be found there.
