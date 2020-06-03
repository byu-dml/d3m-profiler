# Installation

To install:

- Clone repo
- Run the following (preferably in a virtualenv): `python -m pip install -e path/to/repo/` 

Install requirements.txt

Download desired sent2vec weights [here](https://github.com/epfml/sent2vec#downloading-sent2vec-pre-trained-models)

Collect raw data with `python build_table.py </path/to/datasets/dir>`

* Builds a csv of column metadata and column types

Embed raw data with `python3 embed.py </path/to/sent2vec_weights.bin>`

Run models with `python3 evaluate_models.py [<number of cores to use>]`

* Set `use_small_data` manually in the code to switch between small and large data
* Predictions saved to `results[_small]/predictions_<model_name>.csv`

Score predictions with `python3 score_results.py results[_small]/<filename>.csv`



# Running the baseline with `simon`
Make sure that the `simon` submodule is initialized. Run this command while inside the `d3m-profiler` directory.
```
git submodule update --init --recursive
```
Create a new `conda` environment and run the baseline script. See [below](#installing-conda-on-linux-via-command-line) for instructions on how to install `conda`.
```
conda create --name simon cudatoolkit=9.0 tensorflow tensorflow-gpu faker keras scikit-learn python-dateutil pandas scipy matplotlib h5py numpy
conda activate simon
python simon.py
```
**Note**: When creating the `conda` environment, we just specify the version for `cudatoolkit` since `simon` requires CUDA 9.0. `conda` will automatically figure out which version of the other packages to download to maintain compatibility.

# Installing `conda` on Linux via command line
Navigate to a directory where you want to save the `conda` installer.

Go to the (Anaconda distribution website)[https://www.anaconda.com/products/individual#Downloads] and copy the link for the 64-bit (x86) installer. Download the installer with `curl`.

```
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
```
**Note** replace the link with the one you copied.

Run the installer script.
```
bash Anaconda3-2020.02-Linux-x86_64.sh
```
**Note** replace the script name with the version of the one you downloaded.

Follow the prompts. Activate the installation by sourcing your shell's resource file, for example `source ~/.bashrc` or `source ~/.zshrc`.
