import sys, os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score

def score_results(results_path: str):
    with open(results_path, 'r') as f:
        results = pd.read_csv(f)

    y = results['colType']
    y_hat = results['colType_predicted']
    y_labels = y.unique()

    with open(os.path.splitext(results_path)[0]+'_scores.txt', 'w') as f:
        f.write('y percentages:\n')
        f.write(str(y.value_counts(normalize=True))+'\n')
        f.write('\n')
        f.write('Accuracy:' + str(accuracy_score(y, y_hat)) + '\n')
        f.write('F1 Macro:' + str(f1_score(y, y_hat, average='macro')) + '\n')
        f.write('F1 Micro:' + str(f1_score(y, y_hat, average='micro')) + '\n')

    conf_mat_normalized = confusion_matrix(y, y_hat, labels=y_labels, normalize='true')
    disp = ConfusionMatrixDisplay(conf_mat_normalized, y_labels)
    disp.plot(xticks_rotation='vertical')
    plt.savefig(os.path.splitext(results_path)[0]+'_ncm.png')
    plt.show()
    plt.close()

    conf_mat = confusion_matrix(y, y_hat, labels=y_labels)
    disp = ConfusionMatrixDisplay(conf_mat, y_labels)
    disp.plot(xticks_rotation='vertical')
    plt.savefig(os.path.splitext(results_path)[0]+'_cm.png')
    plt.show()
    plt.close()