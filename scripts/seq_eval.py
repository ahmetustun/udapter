import argparse
from seqeval.metrics import classification_report
from seqeval.metrics import accuracy_score


def read_conllu(file, column):
    fin = open(file)
    sentences = []
    sentence = []
    for line in fin:
        if line.startswith('#'):
            continue
        if line is None or line == '\n':
            sentences.append(sentence)
            sentence = []
        else:
            columns = line.rstrip().split('\t')
            if not '.' in columns[0]:
                sentence.append(line.rstrip().split('\t')[column])
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences


parser = argparse.ArgumentParser()
parser.add_argument("--gold_file", type=str)
parser.add_argument("--pred_file", type=str)
parser.add_argument("--column", type=int, default=5)
args = parser.parse_args()

y_true = read_conllu(args.gold_file, args.column)
y_pred = read_conllu(args.pred_file, args.column)

assert len([item for sublist in y_true for item in sublist]) == len([item for sublist in y_pred for item in sublist])

print(classification_report(y_true, y_pred, digits=4))
print(accuracy_score(y_true, y_pred))
