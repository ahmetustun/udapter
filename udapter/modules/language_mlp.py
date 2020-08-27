import random
import json

import lang2vec.lang2vec as l2v

from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F


class LanguageMLP(nn.Module):
    def __init__(self, config, in_language_list: List[str], oov_language_list: List[str], letter_codes: str):
        super(LanguageMLP, self).__init__()

        self.config = config
        self.do_onehot = config.one_hot
        self.in_language_list = in_language_list
        self.oov_language_list = oov_language_list
        self.letter_codes = letter_codes

        l2v.LETTER_CODES_FILE = letter_codes
        l2v.LETTER_CODES = json.load(open(letter_codes))

        self.l2v_cache = dict()
        self._cache_language_features()

        nl_project = config.nl_project
        in_features = len(self.in_language_list) + 1 + config.num_language_features if self.do_onehot else config.num_language_features
        self.nonlinear_project = nn.Linear(in_features, nl_project)
        self.down_project = nn.Linear(nl_project, config.low_rank_dim)
        self.activation = F.relu
        self.dropout = nn.Dropout(config.language_emb_dropout)

    def forward(self, lang_ids):

        lang_vector = self._encode_language_ids(lang_ids, self.do_onehot)

        lang_emb = self.nonlinear_project(torch.tensor(lang_vector).to(lang_ids.device))
        lang_emb = self.activation(lang_emb)
        lang_emb = self.down_project(lang_emb)
        lang_emb = self.dropout(lang_emb)
        return lang_emb

    def _encode_language_ids(self, language_id: int, do_onehot: bool = False) -> List[int]:

        # language one-hot vector
        # 0th id for UNK language
        # drop language_id
        one_hot = [0 for i in range(len(self.in_language_list) + 1)]
        if (random.random() < self.config.language_drop_rate) and self.training:
            one_hot[0] = 1
        elif language_id >= 1000:
            one_hot[0] = 1
        else:
            one_hot[language_id + 1] = 1

        # feature vector from lang2vec cache
        features = self.l2v_cache[self.in_language_list[language_id] if language_id < 1000 else self.oov_language_list[language_id-1000]]

        return features if not do_onehot else one_hot + features

    def _cache_language_features(self):

        features = dict()
        for lang in self.in_language_list + self.oov_language_list:
            features[lang] = l2v.get_features(l2v.LETTER_CODES[lang], self.config.language_features)[l2v.LETTER_CODES[lang]]
        self.l2v_cache = features
