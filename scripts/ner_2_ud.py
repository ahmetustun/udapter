import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default="data/ner", type=str)
args = parser.parse_args()

languages = ['koi', 'olo', 'mr', 'mdf', 'sa', 'tl', 'yo', 'am', 'bm', 'be', 'br', 'bxr', 'zh-yue', 'myv', 'fo', 'kk', 'gn', 'ta', 'te', 'hsb', 'cy']

dirs = os.listdir(args.dataset_dir)
for dir in dirs:
    file_dir = os.path.join(args.dataset_dir,dir)
    files = os.listdir(file_dir)
    for f in files:
        with open(os.path.join(file_dir, f+'.conllu'), 'w') as write:
            with open(os.path.join(file_dir, f), 'r') as read:
                for line in read:
                    idx = 0
                    if line != '\n' and not line.startswith('#'):
                        form = line.rstrip('\n').split('\t')
                        lang,word = form[0].split(':',1)
                        tag = form[1]
                        anno = [idx, word, '_', tag]
                        anno += ['99'] * 6
                        anno += [lang]
                        write.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(*anno))
                    else:
                        write.write(line)
