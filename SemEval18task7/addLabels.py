import ast
import sys
from parameters18 import *


def read_labels(file):
    labs = {}
    with open(file) as lbf:
        for i, lab in enumerate(lbf, start=1):
            labs[str(i)] = lab.strip()
    return labs


def read_record(file):
    with open(file) as f:
        return [ast.literal_eval(line) for line in f]


if __name__ == '__main__':
    which_set = '' if sys.argv[1] == '0' else sys.argv[1]
    path_to_predictions = path_to_model_folder + 'predictions.txt'
    path_to_labels = path_to_feat_folder + 'labels.txt'
    path_to_test = path_to_feat_folder + 'record_test' + which_set + '.txt'
    labels = read_labels(path_to_labels)
    test_record = read_record(path_to_test)
    relation_list = [(rec[2:5]) for rec in test_record]  # list of entities in order of test case

    with open('answer_key18.txt', 'w+') as key:
        for rec in relation_list:
            e1, e2, label = rec
            label_split = label.split()
            if len(label_split) == 2:
                key.write(label_split[0] + '(' + e1 + ',' + e2 + ',REVERSE)' + '\n')
            else:
                key.write(label_split[0] + '(' + e1 + ',' + e2 + ')' + '\n')

    with open(path_to_predictions) as predicts, open(path_to_model_folder + 'predictions_with_labels.txt', 'w+') as out:
        for i, pred in enumerate(predicts):
            e1, e2, label = relation_list[i]
            label_split = label.split()
            actual = labels[pred.strip()]
            if len(label_split) == 2:  # mistake; should have been len(actual) to get direction of predictions
               out.write(actual.split()[0] + '(' + e1 + ',' + e2 + ',REVERSE)' + '\n')
            else:
               out.write(actual.split()[0] + '(' + e1 + ',' + e2 + ')' + '\n')
