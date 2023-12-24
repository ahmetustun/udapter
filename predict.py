"""
Predict conllu files given a trained model
"""

import os
import shutil
import logging
import argparse
import tarfile
from pathlib import Path

from allennlp.common import Params
from allennlp.common.util import import_submodules
from allennlp.models.archival import archive_model

from udapter import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("archive", type=str, help="The archive file")
parser.add_argument("input_file", type=str, help="The input file to predict")
parser.add_argument("pred_file", type=str, help="The output prediction file")
parser.add_argument("--eval_file", default=None, type=str,
                    help="If set, evaluate the prediction and store it in the given file")
parser.add_argument("--device", default=0, type=int, help="CUDA device number; set to -1 for CPU")
parser.add_argument("--batch_size", default=1, type=int, help="The size of each prediction batch")
parser.add_argument("--lazy", action="store_true", help="Lazy load dataset")

args = parser.parse_args()

import_submodules("udapter")

archive_dir = Path(args.archive).resolve().parent

if not os.path.isfile(archive_dir / "weights.th"):
    with tarfile.open(args.archive) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, archive_dir)

config_file = archive_dir / "config.json"

overrides = {}
if args.device is not None:
    overrides["trainer"] = {"cuda_device": args.device}
if args.lazy:
    overrides["dataset_reader"] = {"lazy": args.lazy}
configs = [Params(overrides), Params.from_file(config_file)]
params = util.merge_configs(configs)

if not args.eval_file:
    util.predict_model_with_archive("udapter_predictor", params, archive_dir, args.input_file, args.pred_file,
                                    batch_size=args.batch_size)
else:
    util.predict_and_evaluate_model_with_archive("udapter_predictor", params, archive_dir, args.input_file,
                                                 args.pred_file, args.eval_file, batch_size=args.batch_size)
