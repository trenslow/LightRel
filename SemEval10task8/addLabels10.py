import itertools
import sys
from parameters10 import *


def read_labels(file):
    labs = {}
    with open(file) as lbs:
        for i, lab in enumerate(lbs):
            labs[str(i)] = lab.strip()
    return labs


if __name__ == '__main__':
    s, c, e = [''.join(char for char in arg if char != '.') for arg in sys.argv[1:]]
    lib_linear_params = 's' + s + 'c' + c + 'e' + e
    path_to_predictions = path_to_model_folder + lib_linear_params + '_predictions.txt'
    path_to_labels = path_to_feat_folder + 'labels.txt'
    path_to_full_test = 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
    labels = read_labels(path_to_labels)

    with open(path_to_predictions) as predicts:
        lines = []
        for i, line in enumerate(predicts):
            p = line.strip()
            lines.append([i, labels[p]])
    with open(path_to_model_folder + lib_linear_params + '_predictions_with_labels.txt', 'w+') as out:
        for line in lines:
            out.write('\t'.join(str(el) for el in line) + '\n')

    with open(path_to_full_test) as test, open('answer_key10.txt', 'w+') as key:
        correct_labels = itertools.islice(test, 1, None, 4)
        for i, label in enumerate(correct_labels):
            key.write(str(i) + '\t' + label)