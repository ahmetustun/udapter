"""
Concatenates all treebanks together
"""

import os
import shutil
import logging
import argparse

from udapter import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("output_dir", type=str, help="The path to output the concatenated files")
parser.add_argument("--dataset_dir", default="data/ud-treebanks-v2.3", type=str,
                    help="The path containing all UD treebanks")
parser.add_argument("--treebanks", default=[], type=str, nargs="+",
                    help="Specify a list of treebanks to use; leave blank to default to all treebanks available")
parser.add_argument("--add_lang_id", dest="lang_id", default=False, action="store_true",
                    help="Add language id for each token in treebanks")

args = parser.parse_args()

treebanks = util.get_ud_treebank_files(args.dataset_dir, args.treebanks)
train, dev, test = list(zip(*[treebanks[k] for k in treebanks]))

for treebank, name in zip([train, dev, test], ["train.conllu", "dev.conllu", "test.conllu"]):
    with open(os.path.join(args.output_dir, name), 'w') as write:
        for t in treebank:
            if not t:
                continue
            with open(t, 'r') as read:
                if not args.lang_id:
                    shutil.copyfileobj(read, write)
                else:
                    for line in read:
                        if line != '\n' and not line.startswith('#'):
                            form = line.rstrip('\n').split('\t')
                            #form += ['treebank=' + os.path.basename(t).split('-')[0]]
                            form += [os.path.basename(t).split('-')[0].split('_')[0]]
                            write.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(*form))
                        else:
                            write.write(line)
        write.close()
