import json
import logging
import os
import sys
import typing
import pandas as pd
from pandas.errors import ParserError, EmptyDataError


logger = logging.getLogger(__name__)


def get_datasets(datasets_dir: str) -> typing.Dict[str, str]:
    if datasets_dir is None:
        raise ValueError("Datasets directory has to be provided.")

    datasets: typing.Dict[str, str] = {}

    for dirpath, dirnames, filenames in os.walk(datasets_dir, followlinks=True):
        if 'datasetDoc.json' in filenames:
            # Do not traverse further (to not parse "datasetDoc.json" or "problemDoc.json" if they
            # exists in raw data filename).
            dirnames[:] = []

            dataset_path = os.path.join(os.path.abspath(dirpath), 'datasetDoc.json')

            try:
                with open(dataset_path, 'r', encoding='utf8') as dataset_file:
                    dataset_doc = json.load(dataset_file)

                dataset_id = dataset_doc['about']['datasetID']

                if (
                    dataset_id[-5:] == '_TEST' or
                    dataset_id[-6:] == '_TRAIN' or
                    dataset_id[-6:] == '_SCORE' or
                    dataset_id[-21:] == '_MIN_METADATA_dataset'
                ):
                    continue

                if dataset_id in datasets:
                    logger.warning(
                        "Duplicate dataset ID '%(dataset_id)s': '%(old_dataset)s' and '%(dataset)s'", {
                            'dataset_id': dataset_id,
                            'dataset': dataset_path,
                            'old_dataset': datasets[dataset_id],
                        },
                    )
                else:
                    datasets[dataset_id] = dataset_path

            except (ValueError, KeyError):
                logger.exception(
                    "Unable to read dataset '%(dataset)s'.", {
                        'dataset': dataset_path,
                    },
                )

    return datasets


def resource_generator(datasets):
    for dataset_id, dataset_doc_path in datasets.items():
        with open(dataset_doc_path, 'r') as dataset_doc:
            dataset = json.load(dataset_doc)
            d_name = human_readable_ify(dataset['about']['datasetName'])
            d_description = dataset['about'].get('description', '')

            for resource in dataset['dataResources']:
                if 'columns' not in resource:
                    logger.warning(
                        "No columns found in a data resource. Dataset ID: '%(d_id)s'; Resource ID: '%(r_id)s'", {
                            'd_id': dataset_id,
                            'r_id': resource['resID']
                        }
                    )
                    continue
                yield resource, d_name, d_description, dataset_doc_path


def extract_columns(datasets):
    output = []
    for resource, d_name, d_description, dataset_doc_path in resource_generator(datasets):
        output.extend(get_metadata_from_resource(resource, d_name, d_description))
    return pd.DataFrame(output)


def extract_data_values(datasets, max_cells, max_len):
    output = []
    for resource, d_name, d_description, dataset_doc_path in resource_generator(datasets):
        data = get_data_from_resource(resource, dataset_doc_path, d_name, max_cells=max_cells, max_len=max_len)
        if data is None:
            continue
        output.extend(data)
    return pd.DataFrame(output)


def get_metadata_from_resource(resource, dataset_name, dataset_description):
    metadata = []
    for column in resource['columns']:
        metadata.append({
            'datasetName': dataset_name,
            'description': dataset_description,
            'colName': human_readable_ify(column['colName']),
            'colType': column['colType']
        })
    return metadata


def open_dataset(resource, dataset_doc_path, max_cells):
    if resource['resPath'][-4:] == '.csv':
        try:
            data = pd.read_csv(os.path.join(os.path.dirname(dataset_doc_path), resource['resPath']))
        except (EmptyDataError, ParserError, ValueError):
            logger.warning(f'Could not open dataset with path {dataset_doc_path}')
            return None
    else:
        values = 0
        tables = []
        for entry in os.scandir(os.path.join(os.path.dirname(dataset_doc_path), resource['resPath'])):
            tables.append(pd.read_csv(entry.path))
            values += len(tables[-1])
            if values >= max_cells:
                break
        data = pd.concat(tables, ignore_index=True)
    return data


def get_data_from_resource(resource, dataset_doc_path, dataset_name, max_cells=100, max_len=20):
    data = open_dataset(resource, dataset_doc_path, max_cells)
    if data is None:
        return None
    extracted_data = []
    for column in resource['columns']:
        values = list(data[column['colName']].values)
        if len(values) > max_cells:
            values = [str(v)[:max_len] for v in values[:max_cells]]
        else:
            values = [str(v)[:max_len] for v in values] + ['' for i in range(max_cells - len(values))]

        extracted_data.append({
            'values': values,
            'colType': column['colType'],
            'datasetName': dataset_name
        })
    return extracted_data


def human_readable_ify(in_str: str) -> str:
    out_str = in_str[0]
    for i in range(1, len(in_str)):
        if in_str[i] in ['_', '-']:
            out_str += ' '
        elif in_str[i].isupper() and in_str[i-1] not in [' ', '_', '-'] and in_str[i-1].islower():
            out_str += ' '+in_str[i]
        elif in_str[i].isdigit() and in_str[i-1] not in [' ', '_', '-']:
            out_str += ' '+in_str[i]
        else:
            out_str += in_str[i]
    return out_str


def build_table(dataset_path, output_filename='data', logfile_path='/dev/null'):
    logging.basicConfig(filename=logfile_path, level=logging.DEBUG)
    data = extract_columns(get_datasets(dataset_path))
    data.to_csv(output_filename, index=False)


if __name__ == '__main__':
    build_table(sys.argv[1])
