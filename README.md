# Installation

To install:

- Clone repo
- Run the following (preferably in a virualenv): `python -m pip install -e path/to/repo/` 

Install requirements.txt

Download desired sent2vec weights [here](https://github.com/epfml/sent2vec#downloading-sent2vec-pre-trained-models). `example.py` uses the model called "sent2vec_wiki_unigrams".

Collect raw data with `python build_table.py </path/to/datasets/dir>`

* Builds a csv of column metadata and column types

Embed raw data with `python3 embed.py </path/to/sent2vec_weights.bin>`

Run models with `python3 evaluate_models.py [<number of cores to use>]`

* Set `use_small_data` manually in the code to switch between small and large data
* Predictions saved to `results[_small]/predictions_<model_name>.csv`

Score predictions with `python3 score_results.py results[_small]/<filename>.csv`

## Creating a `conda` environment
To create a `conda` virtual environment for running `example.py`, run the following commands:
```
conda create --name example python=3.6
conda activate example
pip install -r requirements.txt
```
