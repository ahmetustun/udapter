"""
Extracts a language MLP weights archive from an existing model
"""

import logging
import argparse

from udapter import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("archive_dir", type=str, help="The directory where model.tar.gz resides")

args = parser.parse_args()

util.archive_language_MLP(args.archive_dir)
