
Run the command `python build_table.py [/complete/path/to/datasets/repo]` to build the CSV of annotations and column types

The embedding of the text in this CSV requires the `sent2vec` package, which is not in PyPI. Clone [the sent2vec git repo](https://github.com/epfml/sent2vec), navigate into the directory, and run `pip install .` to install the package. Then the command `py embed.py` (run inside this repository) will create a CSV of the embedded data.

A pre-trained sent2vec model can be downloaded from their repo's landing page on GitHub and must be named `pretrained_embedding_model.bin`, in this directory, when `embed.py` is run.
