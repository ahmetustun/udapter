import random
import json

import lang2vec.lang2vec as l2v

from typing import List

import numpy as np

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

        self.up_project = nn.Linear(config.low_rank_dim, in_features)
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='mean')

        self.update_embs = True
        self.lang_emb = None
        self.loss = None
        self.predicted_lang_vector = None
        self.masked_lang_vector = None

    def forward(self, lang_ids):

        lang_vector = self._encode_language_ids(lang_ids)
        unmasked_lang_vector = torch.tensor(lang_vector).to(lang_ids.device)

        # replace 0's with -1's
        lang_vector = [-1.0 if f == 0.0 else f for f in lang_vector]
        lang_vector = torch.tensor(lang_vector).to(lang_ids.device)

        # mask typological feature vector
        mask, masked_indexes = None, None
        if self.config.typo_mask:
            mask, masked_indexes = self._mask_features(len(lang_vector), self.config.typo_mask_ratio)
            lang_vector = (~mask).float() * lang_vector

        lang_emb = self.nonlinear_project(lang_vector)
        lang_emb = self.activation(lang_emb)
        lang_emb = self.down_project(lang_emb)
        lang_emb = self.dropout(lang_emb)

        loss, predicted = None, None
        if self.config.typo_mask:
            loss, predicted = self._decode_feats(lang_emb, masked_indexes, unmasked_lang_vector)

        return lang_emb, (loss, lang_vector, predicted)

    def _decode_feats(self, lang_emb, masked_indexes, lang_vector):
        hidden = self.activation(self.up_project(lang_emb))

        # TODO: check if this is problematics
        loss = self.loss_fct(hidden[masked_indexes], lang_vector[masked_indexes])
        loss = loss * self.config.typo_loss_weight

        predicted = F.sigmoid(hidden)
        predicted = torch.where(predicted > 0.5, torch.ones_like(predicted), torch.zeros_like(predicted))

        return loss, predicted

    def get_language_emb(self, lang_ids, update=False):

        if self.update_embs:
            self.lang_emb, (self.loss, self.masked_lang_vector, self.predicted_lang_vector) = self.forward(lang_ids)
            self.update_embs = False

        if update:
            self.update_embs = True

        return self.lang_emb, (self.loss, self.masked_lang_vector, self.predicted_lang_vector)

    def _encode_language_ids(self, language_id: int) -> List[int]:

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

        features = features if not self.do_onehot else one_hot + features

        return features

    def _cache_language_features(self):

        features = dict()
        for lang in self.in_language_list + self.oov_language_list:
            features[lang] = l2v.get_features(l2v.LETTER_CODES[lang], self.config.language_features)[l2v.LETTER_CODES[lang]]
        self.l2v_cache = features

    def _mask_features(self, feats_len, masking_ratio):
        # TODO: use typological heuristics for masking
        mask = torch.rand(feats_len) > (1 - masking_ratio)
        masked_indexes = torch.flatten(torch.nonzero(mask))

        return mask, masked_indexes


# FROM: https://github.com/karpathy/pytorch-made/blob/master/made.py
class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.transpose(mask, 0, 1))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)
