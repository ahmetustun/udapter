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
parser.add_argument("--adapter_name", default=None, type=str, help="The name to output the file")

args = parser.parse_args()

util.archive_adapter_weights(args.archive_dir, args.adapter_name)