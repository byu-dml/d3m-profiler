Install requirements.txt

Download desired sent2vec weights [here](https://github.com/epfml/sent2vec#downloading-sent2vec-pre-trained-models)

Collect raw data with `python build_table.py </path/to/datasets/dir>`

* Builds a csv of column metadata and column types

Embed raw data with `python3 embed.py </path/to/sent2vec_weights.bin>`

Run models with `python3 evaluate_models.py [<number of cores to use>]`

* Set `use_small_data` manually in the code to switch between small and large data
* Predictions saved to `results[_small]/predictions_<model_name>.csv`

Score predictions with `python3 score_results.py results[_small]/<filename>.csv`
