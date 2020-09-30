import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--column", type=int)
args = parser.parse_args()

for file in os.listdir(args.data_dir):
    cmd = 'python scripts/seq_eval.py '
    if 'test' in file:
        cmd += ' --gold_file ' + os.path.join(args.data_dir, file)
        cmd += ' --pred_file ' + os.path.join(args.model, file)
        cmd += ' --out_plot ' + os.path.join(args.model, file) + '.eval.png'
        cmd += ' --column ' + str(args.column)
        cmd += ' > ' + os.path.join(args.model, file) + '.eval.json'
        print(cmd)
