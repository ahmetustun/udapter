import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, help="The path containing all UD treebanks")
args = parser.parse_args()

fin = open(args.file)
lines = []
lang = None
for line in fin:
    if line != '\n' and not line.startswith('#'):
        form = line.rstrip('\n').split('\t')
        if lang is None:
            lang = form[10]
            lines.append(line)
        elif lang == form[10]:
            lines.append(line)
        elif lang != form[10]:
            fout = open(args.file +'_'+ lang, 'w')
            for l in lines:
                fout.write(l)
            fout.close()
            lang = form[10]
            lines = []
            lines.append(line)
    else:
        lines.append(line)

fout = open(args.file +'_'+ lang, 'w')
for l in lines:
    fout.write(l)
fout.close()
