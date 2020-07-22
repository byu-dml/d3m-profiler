import logging
import sys
import re
from enum import Enum
import pandas as pd
from d3m.container.dataset import get_dataset, SEMANTIC_TYPES_TO_D3M_COLUMN_TYPES, SEMANTIC_TYPES_TO_D3M_ROLES
from d3m.utils import get_datasets_and_problems
from d3m.metadata import base as metadata_base


class TableKeys(Enum):
    DATASET_NAME = 'datasetName'
    DESCRIPTION = 'description'
    COLUMN_NAME = 'colName'
    COLUMN_TYPE = 'colType'
    COLUMN_DATA = 'data'


logger = logging.getLogger(__name__)


def get_datasets(datasets_dir: str):
    dataset_docs, problem_docs = get_datasets_and_problems(datasets_dir)
    return {key: value for key, value in dataset_docs.items() if not key.endswith(('_SCORE', '_TEST', '_TRAIN'))}


def build_table(datasets_dir, include_data=True, max_cells=100, max_len=20, write_path=None, logfile_path='/dev/null'):
    logging.basicConfig(filename=logfile_path, level=logging.DEBUG)
    output = []
    for dataset_id, dataset_doc_path in get_datasets(datasets_dir).items():
        dataset = get_dataset(dataset_doc_path)
        metadata = dataset.metadata.query(())
        for resource in dataset.keys():
            resource_metadata = dataset.metadata.query((resource, metadata_base.ALL_ELEMENTS)).get('dimension')
            data = dataset.get(resource)[:max_cells]
            data.apply(lambda x: x.str.slice(0, max_len))

            for column in range(resource_metadata['length']):
                raw_column_metadata = dataset.metadata.query((resource, metadata_base.ALL_ELEMENTS, column))
                column_data = {
                    TableKeys.DATASET_NAME.value: metadata['name'],
                    TableKeys.DESCRIPTION.value: metadata.get('description', ''),
                    TableKeys.COLUMN_NAME.value: raw_column_metadata['name'],
                    TableKeys.COLUMN_TYPE.value: get_semantic_column_type(raw_column_metadata['semantic_types']),
                }
                if include_data:
                    column_data[TableKeys.COLUMN_DATA.value] = list(data[raw_column_metadata['name']])
                output.append(column_data)
    if write_path:
        pd.DataFrame(output).to_pickle(write_path)
        return None
    return pd.DataFrame(output)


def get_semantic_column_type(semantic_types: tuple):
    for semantic_type in semantic_types:
        if semantic_type in SEMANTIC_TYPES_TO_D3M_COLUMN_TYPES:
            return SEMANTIC_TYPES_TO_D3M_COLUMN_TYPES[semantic_type]
    for semantic_type in semantic_types:
        if semantic_type in SEMANTIC_TYPES_TO_D3M_ROLES:
            return SEMANTIC_TYPES_TO_D3M_ROLES[semantic_type]
    return 'unknown'


def human_readable_ify(text: str) -> str:
    # replace '_' with a space
    text = re.sub('_', ' ', text)

    # insert a space before a number if the text ends in a number
    # does NOT insert another space if there is already whitespace before the number (or another number)
    # eg: LL0 Test Data0   --> LL0 Test Data 0
    #     LL0 Test Data 23 --> LL0 Test Data 23  [ no change ]
    text = re.sub('((?<![\s\d])\d+$)', ' \\1', text)
    return text


if __name__ == '__main__':
    build_table(sys.argv[1])
