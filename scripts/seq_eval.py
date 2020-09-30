import argparse
from seqeval.metrics import classification_report
from seqeval.metrics import accuracy_score

from collections import defaultdict  # available in Python 2.5 and newer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
parser.add_argument("--out_plot", type=str)
parser.add_argument("--column", type=int, default=5)
args = parser.parse_args()

y_true = read_conllu(args.gold_file, args.column)
y_pred = read_conllu(args.pred_file, args.column)

flat_y_true = [item for sublist in y_true for item in sublist]
flat_y_pred = [item for sublist in y_pred for item in sublist]

assert len(flat_y_true) == len(flat_y_pred)

print(classification_report(y_true, y_pred, digits=4))
print(accuracy_score(y_true, y_pred))

# Creates a confusion matrix
label_count = defaultdict(int)
for label in flat_y_true:
    label_count[label] += 1

labels = []
for l,c in label_count.items():
    if c > 20:
        labels.append(l)

cm = confusion_matrix(flat_y_true, flat_y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(50, 50))
sns.heatmap(cm_df, annot=True, cmap="YlGnBu")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(args.out_plot, bbox_inches='tight')
plt.close()

