import gzip
import os
import shutil
from pathlib import Path
from d3m_profiler.build_table import build_table


def create_metadata_file(dataset_dir, filename):
    if '.csv' not in Path(filename).suffixes:
        raise ValueError('Filename must have the .csv extension')
    filename_stem = Path(Path(filename).stem).stem if Path(filename).suffix == '.gz' else Path(filename).stem
    build_table(dataset_dir, filename_stem)
    if Path(filename).suffix == '.gz':
        compress_output(filename_stem)


def compress_output(filename_stem):
    if not os.path.isfile(f'{filename_stem}.csv'):
        raise ValueError(f'The file "{filename_stem}.csv" cannot be compressed because it does not exist')
    with open(f'{filename_stem}.csv', 'rb') as f_in:
        with gzip.open(f'{filename_stem}.csv.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(f'{filename_stem}.csv')
