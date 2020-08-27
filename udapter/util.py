"""
A collection of handy utilities
"""
import atexit
import tempfile
from typing import List, Tuple, Dict, Any

import os
import glob
import json
import logging
import tarfile
import traceback

import torch

from allennlp.nn import util
from allennlp.common.checks import ConfigurationError
from allennlp.common import Params
from allennlp.common.params import with_fallback, unflatten, parse_overrides
from allennlp.commands.make_vocab import make_vocab_from_params
from allennlp.commands.predict import _PredictManager
from allennlp.common.checks import check_for_gpu
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.models.archival import load_archive, archive_model, _cleanup_archive_dir, Archive
from allennlp.models.model import remove_pretrained_embedding_params
from allennlp.predictors.predictor import Predictor
from allennlp.common.file_utils import cached_path

from udapter.dataset_readers.conll18_ud_eval import evaluate, load_conllu_file, UDError

VOCAB_CONFIG_PATH = "config/create_vocab.json"

logger = logging.getLogger(__name__)


def merge_configs(configs: List[Params]) -> Params:
    """
    Merges a list of configurations together, with items with duplicate keys closer to the front of the list
    overriding any keys of items closer to the rear.
    :param configs: a list of AllenNLP Params
    :return: a single merged Params object
    """
    while len(configs) > 1:
        overrides, config = configs[-2:]
        configs = configs[:-2]

        if "udify_replace" in overrides:
            replacements = [replace.split(".") for replace in overrides.pop("udify_replace")]
            for replace in replacements:
                obj = config
                try:
                    for key in replace[:-1]:
                        obj = obj[key]
                except KeyError:
                    raise ConfigurationError(f"Config does not have key {key}")
                obj.pop(replace[-1])

        configs.append(Params(with_fallback(preferred=overrides.params, fallback=config.params)))

    return configs[0]


def cache_vocab(params: Params, vocab_config_path: str = None):
    """
    Caches the vocabulary given in the Params to the filesystem. Useful for large datasets that are run repeatedly.
    :param params: the AllenNLP Params
    :param vocab_config_path: an optional config path for constructing the vocab
    """
    if "vocabulary" not in params or "directory_path" not in params["vocabulary"]:
        return

    vocab_path = params["vocabulary"]["directory_path"]

    if os.path.exists(vocab_path):
        if os.listdir(vocab_path):
            return

        # Remove empty vocabulary directory to make AllenNLP happy
        try:
            os.rmdir(vocab_path)
        except OSError:
            pass

    vocab_config_path = vocab_config_path if vocab_config_path else VOCAB_CONFIG_PATH

    params = merge_configs([params, Params.from_file(vocab_config_path)])
    params["vocabulary"].pop("directory_path", None)
    make_vocab_from_params(params, os.path.split(vocab_path)[0])


def get_ud_treebank_files(dataset_dir: str, treebanks: List[str] = None) -> Dict[str, Tuple[str, str, str]]:
    """
    Retrieves all treebank data paths in the given directory.
    :param dataset_dir: the directory where all treebank directories are stored
    :param treebanks: if not None or empty, retrieve just the subset of treebanks listed here
    :return: a dictionary mapping a treebank name to a list of train, dev, and test conllu files
    """
    datasets = {}
    treebanks = os.listdir(dataset_dir) if not treebanks else treebanks
    for treebank in treebanks:
        treebank_path = os.path.join(dataset_dir, treebank)
        conllu_files = [file for file in sorted(os.listdir(treebank_path)) if file.endswith(".conllu")]

        train_file = [file for file in conllu_files if file.endswith("train.conllu")]
        dev_file = [file for file in conllu_files if file.endswith("dev.conllu")]
        test_file = [file for file in conllu_files if file.endswith("test.conllu")]

        train_file = os.path.join(treebank_path, train_file[0]) if train_file else None
        dev_file = os.path.join(treebank_path, dev_file[0]) if dev_file else None
        test_file = os.path.join(treebank_path, test_file[0]) if test_file else None

        datasets[treebank] = (train_file, dev_file, test_file)
    return datasets


def get_ud_treebank_names(dataset_dir: str) -> List[Tuple[str, str]]:
    """
    Retrieves all treebank names from the given directory.
    :param dataset_dir: the directory where all treebank directories are stored
    :return: a list of long and short treebank names
    """
    treebanks = os.listdir(dataset_dir)
    short_names = []

    for treebank in treebanks:
        treebank_path = os.path.join(dataset_dir, treebank)
        conllu_files = [file for file in sorted(os.listdir(treebank_path)) if file.endswith(".conllu")]

        test_file = [file for file in conllu_files if file.endswith("test.conllu")]
        test_file = test_file[0].split("-")[0] if test_file else None

        short_names.append(test_file)

    treebanks = ["_".join(treebank.split("_")[1:]) for treebank in treebanks]

    return list(zip(treebanks, short_names))


def predict_model_with_archive(predictor: str, params: Params, archive: str,
                               input_file: str, output_file: str, batch_size: int = 1):
    cuda_device = params["trainer"]["cuda_device"]

    check_for_gpu(cuda_device)
    archive = load_archive(archive,
                           cuda_device=cuda_device)

    predictor = Predictor.from_archive(archive, predictor)

    manager = _PredictManager(predictor,
                              input_file,
                              output_file,
                              batch_size,
                              print_to_console=False,
                              has_dataset_reader=True)
    manager.run()


def predict_and_evaluate_model_with_archive(predictor: str, params: Params, archive: str, gold_file: str,
                               pred_file: str, output_file: str, segment_file: str = None, batch_size: int = 1):
    if not gold_file or not os.path.isfile(gold_file):
        logger.warning(f"No file exists for {gold_file}")
        return

    segment_file = segment_file if segment_file else gold_file
    predict_model_with_archive(predictor, params, archive, segment_file, pred_file, batch_size)

    try:
        evaluation = evaluate(load_conllu_file(gold_file), load_conllu_file(pred_file))
        save_metrics(evaluation, output_file)
    except UDError:
        logger.warning(f"Failed to evaluate {pred_file}")
        traceback.print_exc()


def predict_model(predictor: str, params: Params, archive_dir: str,
                  input_file: str, output_file: str, batch_size: int = 1):
    """
    Predict output annotations from the given model and input file and produce an output file.
    :param predictor: the type of predictor to use, e.g., "udify_predictor"
    :param params: the Params of the model
    :param archive_dir: the saved model archive
    :param input_file: the input file to predict
    :param output_file: the output file to save
    :param batch_size: the batch size, set this higher to speed up GPU inference
    """
    archive = os.path.join(archive_dir, "model.tar.gz")
    predict_model_with_archive(predictor, params, archive, input_file, output_file, batch_size)


def predict_and_evaluate_model(predictor: str, params: Params, archive_dir: str, gold_file: str,
                               pred_file: str, output_file: str, segment_file: str = None, batch_size: int = 1):
    """
    Predict output annotations from the given model and input file and evaluate the model.
    :param predictor: the type of predictor to use, e.g., "udify_predictor"
    :param params: the Params of the model
    :param archive_dir: the saved model archive
    :param gold_file: the file with gold annotations
    :param pred_file: the input file to predict
    :param output_file: the output file to save
    :param segment_file: an optional file separate gold file that can be evaluated,
    useful if it has alternate segmentation
    :param batch_size: the batch size, set this higher to speed up GPU inference
    """
    archive = os.path.join(archive_dir, "model.tar.gz")
    predict_and_evaluate_model_with_archive(predictor, params, archive, gold_file,
                                            pred_file, output_file, segment_file, batch_size)


def save_metrics(evaluation: Dict[str, Any], output_file: str):
    """
    Saves CoNLL 2018 evaluation as a JSON file.
    :param evaluation: the evaluation dict calculated by the CoNLL 2018 evaluation script
    :param output_file: the output file to save
    """
    evaluation_dict = {k: v.__dict__ for k, v in evaluation.items()}

    with open(output_file, "w") as f:
        json.dump(evaluation_dict, f, indent=4)

    logger.info("Metric     | Correct   |      Gold | Predicted | Aligned")
    logger.info("-----------+-----------+-----------+-----------+-----------")
    for metric in ["Tokens", "Sentences", "Words", "UPOS", "XPOS", "UFeats",
                   "AllTags", "Lemmas", "UAS", "LAS", "CLAS", "MLAS", "BLEX"]:
        logger.info("{:11}|{:10.2f} |{:10.2f} |{:10.2f} |{}".format(
                    metric,
                    100 * evaluation[metric].precision,
                    100 * evaluation[metric].recall,
                    100 * evaluation[metric].f1,
                    "{:10.2f}".format(100 * evaluation[metric].aligned_accuracy)
                    if evaluation[metric].aligned_accuracy is not None else ""))


def cleanup_training(serialization_dir: str, keep_archive: bool = False, keep_weights: bool = False):
    """
    Removes files generated from training.
    :param serialization_dir: the directory to clean
    :param keep_archive: whether to keep a copy of the model archive
    :param keep_weights: whether to keep copies of the intermediate model checkpoints
    """
    if not keep_weights:
        for file in glob.glob(os.path.join(serialization_dir, "*.th")):
            os.remove(file)
    if not keep_archive:
        os.remove(os.path.join(serialization_dir, "model.tar.gz"))


def archive_bert_model(serialization_dir: str, config_file: str, output_file: str = None):
    """
    Extracts BERT parameters from the given model and saves them to an archive.
    :param serialization_dir: the directory containing the saved model archive
    :param config_file: the configuration file of the model archive
    :param output_file: the output BERT archive name to save
    """
    archive = load_archive(os.path.join(serialization_dir, "model.tar.gz"))

    model = archive.model
    model.eval()

    try:
        bert_model = model.text_field_embedder.token_embedder_bert.model
    except AttributeError:
        logger.warning(f"Could not find the BERT model inside the archive {serialization_dir}")
        traceback.print_exc()
        return

    weights_file = os.path.join(serialization_dir, "pytorch_model.bin")
    torch.save(bert_model.state_dict(), weights_file)

    if not output_file:
        output_file = os.path.join(serialization_dir, "bert-finetune.tar.gz")

    with tarfile.open(output_file, 'w:gz') as archive:
        archive.add(config_file, arcname="bert_config.json")
        archive.add(weights_file, arcname="pytorch_model.bin")

    os.remove(weights_file)


def archive_adapter_weights(serialization_dir: str, adapter_name: str):
    archive = load_archive(os.path.join(serialization_dir, "model.tar.gz"))

    model = archive.model
    model.eval()
    weights_file = os.path.join(serialization_dir, adapter_name)
    adapters_w = dict()
    for i, layer in enumerate(model.text_field_embedder.token_embedder_bert.bert_model.encoder.layer):
        try:
            adapters_w['attention_adapter_'+str(i)] = layer.attention.output.adapter.state_dict()
            adapters_w['output_adapter_' + str(i)] = layer.output.adapter.state_dict()
        except AttributeError:
            logger.warning(f"Could not find the adapter model inside the archive {serialization_dir}")
            traceback.print_exc()
            return

    torch.save(adapters_w, weights_file)


def archive_language_MLP(serialization_dir: str):

    archive = load_archive(os.path.join(serialization_dir, "model.tar.gz"))

    model = archive.model
    model.eval()

    try:
        # language_mlp = model.text_field_embedder.token_embedder_bert.language_embedder
        language_mlp = model.language_embedder

        # TODO: add configs in correct format to the archive
        config = language_mlp.config
    except AttributeError:
        logger.warning(f"Could not find the Language MLP inside the archive {serialization_dir}")
        traceback.print_exc()
        return

    weights_file = os.path.join(serialization_dir, "language_mlp.bin")
    torch.save(language_mlp.state_dict(), weights_file)


def load_adapter_weights(serialization_dir: str, adapters_dir: str, config_file: str, output_file: str=None):
    archive = _load_archive(os.path.join(serialization_dir, "model.tar.gz"), adapters_dir)

    model = archive.model
    model.eval()

    try:
        bert_model = model.text_field_embedder.token_embedder_bert.bert_model
    except AttributeError:
        logger.warning(f"Could not find the BERT model inside the archive {serialization_dir}")
        traceback.print_exc()
        return

    weights_file = os.path.join(serialization_dir, "adapter_model.bin")
    torch.save(bert_model.state_dict(), weights_file)

    if not output_file:
        output_file = os.path.join(serialization_dir, "bert-finetune-adapter.tar.gz")

    with tarfile.open(output_file, 'w:gz') as archive:
        archive.add(config_file, arcname="bert_config.json")
        archive.add(weights_file, arcname="adapter_model.bin")

    os.remove(weights_file)


def _load_archive(archive_file: str,
                  adapters_dir: str,
                 cuda_device: int = -1,
                 overrides: str = "",
                 weights_file: str = None):
    """
    Instantiates an Archive from an archived `tar.gz` file.

    Parameters
    ----------
    archive_file: ``str``
        The archive file to load the model from.
    weights_file: ``str``, optional (default = None)
        The weights file to use.  If unspecified, weights.th in the archive_file will be used.
    cuda_device: ``int``, optional (default = -1)
        If `cuda_device` is >= 0, the model will be loaded onto the
        corresponding GPU. Otherwise it will be loaded onto the CPU.
    overrides: ``str``, optional (default = "")
        JSON overrides to apply to the unarchived ``Params`` object.
    """

    # redirect to the cache, if necessary
    resolved_archive_file = cached_path(archive_file)

    if resolved_archive_file == archive_file:
        logger.info(f"loading archive file {archive_file}")
    else:
        logger.info(f"loading archive file {archive_file} from cache at {resolved_archive_file}")

    if os.path.isdir(resolved_archive_file):
        serialization_dir = resolved_archive_file
    else:
        # Extract archive to temp dir
        tempdir = tempfile.mkdtemp()
        logger.info(f"extracting archive file {resolved_archive_file} to temp dir {tempdir}")
        with tarfile.open(resolved_archive_file, 'r:gz') as archive:
            archive.extractall(tempdir)
        # Postpone cleanup until exit in case the unarchived contents are needed outside
        # this function.
        atexit.register(_cleanup_archive_dir, tempdir)

        serialization_dir = tempdir

    # Check for supplemental files in archive
    fta_filename = os.path.join(serialization_dir, "files_to_archive.json")
    if os.path.exists(fta_filename):
        with open(fta_filename, 'r') as fta_file:
            files_to_archive = json.loads(fta_file.read())

        # Add these replacements to overrides
        replacements_dict: Dict[str, Any] = {}
        for key, original_filename in files_to_archive.items():
            replacement_filename = os.path.join(serialization_dir, f"fta/{key}")
            if os.path.exists(replacement_filename):
                replacements_dict[key] = replacement_filename
            else:
                logger.warning(f"Archived file {replacement_filename} not found! At train time "
                               f"this file was located at {original_filename}. This may be "
                               "because you are loading a serialization directory. Attempting to "
                               "load the file from its train-time location.")

        overrides_dict = parse_overrides(overrides)
        combined_dict = with_fallback(preferred=overrides_dict, fallback=unflatten(replacements_dict))
        overrides = json.dumps(combined_dict)

    # Load config
    config = Params.from_file(os.path.join(serialization_dir, "config.json"), overrides)
    config.loading_from_archive = True

    if weights_file:
        weights_path = weights_file
    else:
        weights_path = os.path.join(serialization_dir, "weights.th")
        # Fallback for serialization directories.
        if not os.path.exists(weights_path):
            weights_path = os.path.join(serialization_dir, "best.th")


    # Instantiate model. Use a duplicate of the config, as it will get consumed.
    model = _load(config.duplicate(),
                  adapters_dir=adapters_dir,
                  weights_file=weights_path,
                  serialization_dir=serialization_dir,
                  cuda_device=cuda_device)

    return Archive(model=model, config=config)


def _load(config: Params,
          adapters_dir: str,
          serialization_dir: str,
          weights_file: str = None,
          cuda_device: int = -1) -> 'Model':
    """
    Instantiates an already-trained model, based on the experiment
    configuration and some optional overrides.
    """
    weights_file = weights_file or os.path.join(serialization_dir, "best.th")

    # Load vocabulary from file
    vocab_dir = os.path.join(serialization_dir, 'vocabulary')
    # If the config specifies a vocabulary subclass, we need to use it.
    vocab_params = config.get("vocabulary", Params({}))
    vocab_choice = vocab_params.pop_choice("type", Vocabulary.list_available(), True)
    vocab = Vocabulary.by_name(vocab_choice).from_files(vocab_dir)

    model_params = config.get('model')

    # The experiment config tells us how to _train_ a model, including where to get pre-trained
    # embeddings from.  We're now _loading_ the model, so those embeddings will already be
    # stored in our weights.  We don't need any pretrained weight file anymore, and we don't
    # want the code to look for it, so we remove it from the parameters here.
    remove_pretrained_embedding_params(model_params)
    model = Model.from_params(vocab=vocab, params=model_params)

    # If vocab+embedding extension was done, the model initialized from from_params
    # and one defined by state dict in weights_file might not have same embedding shapes.
    # Eg. when model embedder module was transferred along with vocab extension, the
    # initialized embedding weight shape would be smaller than one in the state_dict.
    # So calling model embedding extension is required before load_state_dict.
    # If vocab and model embeddings are in sync, following would be just a no-op.
    model.extend_embedder_vocab()

    # model_state = torch.load(weights_file, map_location=util.device_mapping(cuda_device))
    # model.load_state_dict(model_state, strict=False)

    for file in os.listdir(adapters_dir):
        logger.info(f"{file} is loading..")

    # loop over the adapters folder and load weights into a dictionary
    for i, layer in enumerate(model.text_field_embedder.token_embedder_bert.bert_model.encoder.layer):
        try:
            for j, (file, attention_adapter, output_attention) in enumerate(zip(os.listdir(adapters_dir), layer.attention.output.adapter, layer.output.adapter)):
                adapter_state = torch.load(os.path.join(adapters_dir, file))
                attention_adapter.load_state_dict(adapter_state['attention_adapter_' + str(i)])
                output_attention.load_state_dict(adapter_state['output_adapter_' + str(i)])
        except AttributeError:
            logger.warning(f"Could not find the adapter model inside the archive {adapters_dir}")
            traceback.print_exc()
            return

    # Force model to cpu or gpu, as appropriate, to make sure that the embeddings are
    # in sync with the weights
    if cuda_device >= 0:
        model.cuda(cuda_device)
    else:
        model.cpu()

    return model
