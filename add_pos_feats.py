"""
Add POS to FEATS string
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

parser.add_argument("--input_dir", type=str)

args = parser.parse_args()

for treebank in ["train.conllu", "dev.conllu", "test.conllu"]:
    with open(os.path.join(args.input_dir,treebank)) as read:
        with open(os.path.join(args.input_dir, treebank+'.p2f'), 'w') as write:
            for line in read:
                if line != '\n' and not line.startswith('#'):
                    form = line.rstrip('\n').split('\t')
                    if form[5] == '_':
                        form[5] = 'Pos='+form[3]
                    else:
                        form[5] = 'Pos='+form[3]+'|'+form[3]
                    write.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\\n'.format(*form))
                else:
                    write.write(line)
            write.close()
