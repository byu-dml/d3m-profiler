# Installation

To install:

- Clone repo
- Run the following (preferably in a virualenv): `python -m pip install -e path/to/repo/` 

Install requirements.txt

Download desired sent2vec weights [here](https://github.com/epfml/sent2vec#downloading-sent2vec-pre-trained-models)

Collect raw data with `python build_table.py </path/to/datasets/dir>`

* Builds a csv of column metadata and column types

Embed raw data with `python3 embed.py </path/to/sent2vec_weights.bin>`

Run models with `python3 evaluate_models.py [<number of cores to use>]`

* Set `use_small_data` manually in the code to switch between small and large data
* Predictions saved to `results[_small]/predictions_<model_name>.csv`

Score predictions with `python3 score_results.py results[_small]/<filename>.csv`

# Creating a virtual environment
```
conda create --name quick-model cudatoolkit=9.0 tensorflow tensorflow-gpu cycler joblib kiwisolver matplotlib numpy pandas pyparsing python-dateutil pytz scikit-learn scipy six
pip install imbalanced-learn==0.6.2 sentence-transformers
```