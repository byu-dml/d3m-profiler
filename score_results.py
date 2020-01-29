import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score


if __name__ == '__main__':
    results_path = sys.argv[1]
    with open(results_path, 'r') as f:
        results = pd.read_csv(f)

    y = results['colType']
    y_hat = results['colType_predicted']
    y_labels = y.unique()

    print('y percentages:')
    print(y.value_counts(normalize=True))
    print()
    print('Accuracy:', accuracy_score(y, y_hat))
    print('F1 Macro:', f1_score(y, y_hat, average='macro'))
    print('F1 Micro:', f1_score(y, y_hat, average='micro'))

    conf_mat_normalized = confusion_matrix(y, y_hat, labels=y_labels, normalize='true')
    disp = ConfusionMatrixDisplay(conf_mat_normalized, y_labels)
    disp.plot(xticks_rotation='vertical')
    plt.show()
    plt.close()

    conf_mat = confusion_matrix(y, y_hat, labels=y_labels)
    disp = ConfusionMatrixDisplay(conf_mat, y_labels)
    disp.plot(xticks_rotation='vertical')
    plt.show()
    plt.close()
