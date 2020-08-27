"""
Extracts a adapter weights archive from an existing model
"""

import logging
import argparse

from udapter import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("archive_dir", type=str, help="The directory where model.tar.gz resides")
parser.add_argument("--adapters_dir", default=None, type=str, help="The directory where adapter.bin.x resides")
parser.add_argument("--output_path", default=None, type=str, help="The path for output pytorch model")

args = parser.parse_args()

bert_config = "adapter_bert/configs/adapter-bert.json"
util.load_adapter_weights(args.archive_dir, args.adapters_dir, bert_config, args.output_path)
