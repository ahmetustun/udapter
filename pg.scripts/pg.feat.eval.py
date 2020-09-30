import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--data_dir", type=str)
args = parser.parse_args()

for file in os.listdir(args.data_dir):
    cmd = 'python scripts/evaluate_feats.py '
    if 'test' in file:
        cmd += ' --reference ' + os.path.join(args.data_dir, file)
        cmd += ' --output ' + os.path.join(args.model, file)
        cmd += ' > ' + os.path.join(args.model, file) + '.feat.json'
        print(cmd)
