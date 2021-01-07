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

        self.l2v_cache_avg = dict()
        self.l2v_cache_missing = dict()
        self._cache_language_features()

        nl_project = config.nl_project
        in_features = len(self.in_language_list) + 1 + config.num_language_features if self.do_onehot else config.num_language_features

        # number of geo feats = 299
        self.num_geo_feats = 299 if 'geo' in self.config.language_features else 0

        self.nonlinear_project = nn.Linear(in_features, nl_project)
        self.down_project = nn.Linear(nl_project, config.low_rank_dim)
        self.activation = F.relu
        self.dropout = nn.Dropout(config.language_emb_dropout)

        self.up_project = nn.Linear(config.low_rank_dim, in_features)
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='mean')

        self.update_embs = True
        self.lang_emb = None
        self.loss = None
        self.accuracy = None

    def forward(self, lang_ids):

        lang_vector, missing_feats = self._encode_language_ids(lang_ids)

        if self.training:
            unmasked_lang_vector = torch.tensor(lang_vector).to(lang_ids.device)

        # mask typological feature vector
        mask, masked_indexes = torch.ones(len(lang_vector)), None
        if self.config.typo_mask or self.config.typo_missing_out or self.config.eval_with_knn:
            mask, masked_indexes = self._mask_features(missing_feats[:len(lang_vector)-self.num_geo_feats], self.config.typo_mask_ratio)
            mask = torch.cat((mask, torch.zeros(self.num_geo_feats).byte()))

        # replace 0's with -1's
        lang_vector = [-1.0 if (f == 0.0 or f == '--') else f for f in lang_vector]
        lang_vector = torch.tensor(lang_vector).to(lang_ids.device)
        if (not self.training) and self.config.eval_with_knn:
            lang_vector[masked_indexes] = lang_vector[masked_indexes]/2
        else:
            lang_vector = (~mask).float().to(lang_ids.device) * lang_vector

        lang_emb = self.nonlinear_project(lang_vector)
        lang_emb = self.activation(lang_emb)
        lang_emb = self.down_project(lang_emb)
        lang_emb = self.dropout(lang_emb)

        loss, binary_acc = 0, 0
        if self.config.typo_mask and self.training:
            loss, probs = self._decode_feats(lang_emb, masked_indexes, unmasked_lang_vector)
            binary_acc = self._calculate_accuracy(unmasked_lang_vector[masked_indexes], probs[masked_indexes])

        return lang_emb, (loss, binary_acc)

    def _decode_feats(self, lang_emb, masked_indexes, lang_vector):
        hidden = self.activation(self.up_project(lang_emb))

        # TODO: check if this is problematics
        loss = self.loss_fct(hidden[masked_indexes], lang_vector[masked_indexes])
        loss = loss * self.config.typo_loss_weight

        probs = F.sigmoid(hidden)

        return loss, probs

    def _calculate_accuracy(self, y_true, y_prob):
        assert len(y_true.shape) == 1 and y_true.shape == y_prob.shape
        y_prob = y_prob > 0.5
        return (y_true == y_prob.float()).sum().item() / y_true.shape[0]

    def get_language_emb(self, lang_ids, update=False):

        if self.update_embs:
            self.lang_emb, (self.loss, self.accuracy) = self.forward(lang_ids)
            self.update_embs = False

        if update:
            self.update_embs = True

        return self.lang_emb, (self.loss, self.accuracy)

    def get_accuracy(self):
        main_metrics = {
            f".run/typo/acc": self.accuracy
        }
        return {**main_metrics}

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
        lang_str = self.in_language_list[language_id] if language_id < 1000 else self.oov_language_list[language_id-1000]
        if self.training or self.config.eval_with_knn:
            features = self.l2v_cache_knn[lang_str]
        else:
            features = self.l2v_cache_avg[lang_str]

        features = features if not self.do_onehot else one_hot + features
        missing_feats = self.l2v_cache_missing[lang_str]

        return features, missing_feats

    def _cache_language_features(self):

        features_knn = dict()
        features_avg = dict()
        missing_feats = dict()
        for lang in self.in_language_list + self.oov_language_list:
            features_knn[lang] = l2v.get_features(l2v.LETTER_CODES[lang], self.config.language_features)[l2v.LETTER_CODES[lang]]
            features_avg[lang] = l2v.get_features(l2v.LETTER_CODES[lang], self.config.language_features.replace('knn', 'average'))[l2v.LETTER_CODES[lang]]
            missing_feats[lang] = [1 if f == '--' else 0 for f in features_avg[lang]]
        self.l2v_cache_knn = features_knn
        self.l2v_cache_avg = features_avg
        self.l2v_cache_missing = missing_feats

    def _mask_features(self, missing_feats, masking_ratio):
        # TODO: use typological heuristics for masking

        missing = torch.tensor(missing_feats).byte()
        if self.training and self.config.typo_mask:
            mask = torch.rand(len(missing_feats)) > (1 - masking_ratio)
            mask = (mask.float() * (~missing).float()).byte()
        else:
            mask = missing
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
