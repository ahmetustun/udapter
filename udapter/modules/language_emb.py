import torch
import torch.nn as nn


class LanguageEmbeddings(nn.Module):
    def __init__(self, config):
        super(LanguageEmbeddings, self).__init__()
        self.config = config

        self.language_emb = nn.Embedding(num_embeddings=config.num_languages, embedding_dim=config.low_rank_dim)
        self.dropout = nn.Dropout(config.language_emb_dropout)

    def forward(self, lang_ids):

        if lang_ids < 1000:
            lang_emb = self.language_emb(lang_ids.clone().detach())
            lang_emb = self.dropout(lang_emb)
        else:
            lang_emb = torch.mean(self.language_emb.weight, dim=0)

        return lang_emb.view(-1)
