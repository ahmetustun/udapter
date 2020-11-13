# UDapter

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

UDapter is a multilingual dependency parser that uses "contextual" adapters together with language-typology features for language-specific adaptation. This repository includes the code for "[UDapter: Language Adaptation for Truly Universal Dependency Parsing](https://arxiv.org/abs/2004.14327)" 

[![UDify Model Architecture](docs/model.png)](https://arxiv.org/pdf/1904.02099.pdf)

This project is built on [UDify](https://github.com/Hyperparticle/udify) using [AllenNLP](https://allennlp.org/) and [Huggingface Transformers](https://github.com/huggingface/transformers). The code is tested on Python v3.6 

## Getting Started

Install the Python packages in `requirements.txt`. 
```bash
pip install -r ./requirements.txt
```

After downloading the UD corpus from [universaldependencies.org](https://universaldependencies.org/), please run `scripts/concat_ud_data.sh` with `--add_lang_id` to generate the multilingual UD dataset with language ids.

### Training the Model

Before training, make sure the dataset is downloaded and extracted into the `data` directory and the multilingual 
dataset is generated with `scripts/concat_ud_data.sh`. To indicate training and zero-shot languages use `languages/in-langs` and `languages\oov-langs` respectively. To train the multilingual model, 
run the command

```bash
python train.py --config config/ud/multilingual/udapter-test.json --name udapter
```

### Predicting Universal Dependencies from a Trained Model

To predict UD annotations, one can supply the path to the trained model and an input `conllu`-formatted file with a language id as the last column. To split concatenated treebanks with language id, use `scripts/split_file_by_lang.py`. For prediction: 

```bash
python predict.py <archive> <input.conllu> <output.conllu> [--eval_file results.json]
```

## Citing This Research

If you use UDify for your research, please cite this work as:

```latex
@inproceedings{ustun-etal-2020-udapter,
    title = "{UD}apter: Language Adaptation for Truly {U}niversal {D}ependency Parsing",
    author = {{\"U}st{\"u}n, Ahmet  and
      Bisazza, Arianna  and
      Bouma, Gosse  and
      van Noord, Gertjan},
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.180",
    pages = "2302--2315",
    abstract = "Recent advances in multilingual dependency parsing have brought the idea of a truly universal parser closer to reality. However, cross-language interference and restrained model capacity remain major obstacles. To address this, we propose a novel multilingual task adaptation approach based on contextual parameter generation and adapter modules. This approach enables to learn adapters via language embeddings while sharing model parameters across languages. It also allows for an easy but effective integration of existing linguistic typology features into the parsing network. The resulting parser, UDapter, outperforms strong monolingual and multilingual baselines on the majority of both high-resource and low-resource (zero-shot) languages, showing the success of the proposed adaptation approach. Our in-depth analyses show that soft parameter sharing via typological features is key to this success.",
}
