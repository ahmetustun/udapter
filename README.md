# UDapter

The project is built on [UDify](https://github.com/Hyperparticle/udify) using [AllenNLP](https://allennlp.org/) and [Huggingface Transformers](https://github.com/huggingface/transformers).

## Getting Started

Install the Python packages in `requirements.txt`. 
```bash
pip install -r ./requirements.txt
```

Download the UD corpus by running the script

```bash
bash ./scripts/download_ud_data.sh
```

or alternatively download the data from [universaldependencies.org](https://universaldependencies.org/) and extract 
into `data/ud-treebanks-v2.3/`, then run `scripts/concat_ud_data.sh` to generate the multilingual UD dataset with language ids.


### Training the Model

Before training, make sure the dataset is downloaded and extracted into the `data` directory and the multilingual 
dataset is generated with `scripts/concat_ud_data.sh`. To indicate training and zero-shot languages use `languages/in-langs` and `languages\oov-langs` respectively. To train the multilingual model, 
run the command

```bash
python train.py --config config/ud/udapter/udapter-test.json --name udapter
```

### Viewing Model Performance

One can view how well the models are performing by running TensorBoard

```bash
tensorboard --logdir logs
```

This should show the currently trained model as well as any other previously trained models. The model will be stored 
in a folder specified by the `--name` parameter as well as a date stamp, e.g., `logs/udapter/2020.05.01_00.00.01`.


## Predicting Universal Dependencies from a Trained Model

To predict UD annotations, one can supply the path to the trained model and an input `conllu`-formatted file with a language id as the last column. To split concatenated treebanks with language id, use `scripts/split_file_by_lang.py`. For prediction: 

```bash
python predict.py <archive> <input.conllu> <output.conllu> [--eval_file results.json]
```

## Configuration Options

1. One can specify the type of device to run on. For a single GPU, use the flag `--device 0`, or `--device -1` for CPU.
2. To skip waiting for the dataset to be fully loaded into memory, use the flag `--lazy`. 
Note that the dataset won't be shuffled.
3. Resume an existing training run with `--resume <archive_dir>`.
4. Specify a config file with `--config <config_file>`.