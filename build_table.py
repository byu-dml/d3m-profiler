import sys, os, typing, json, logging, csv


DATA_PATH = './data.csv'


LOG_FILENAME = '/dev/null'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_datasets(datasets_dir: str) -> typing.Dict[str, str]:
    if datasets_dir is None:
        raise exceptions.InvalidArgumentValueError("Datasets directory has to be provided.")

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


def build_table(dataset_path):

    datasets = get_datasets(dataset_path)

    output_headers = ['datasetName', 'description', 'colName', 'colType']
    output = []
    for dataset_id, path in datasets.items():
        with open(path, 'r') as dataset_doc:
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

                for column in resource['columns']:
                    c_name = human_readable_ify(column['colName'])
                    c_type = column['colType']

                    output.append({
                        'datasetName': d_name,
                        'description': d_description,
                        'colName': c_name,
                        'colType': c_type
                    })

    with open(DATA_PATH, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_headers)

        writer.writeheader()
        for row in output:
            writer.writerow(row)


if __name__ == '__main__':
    build_table(sys.argv[1])
