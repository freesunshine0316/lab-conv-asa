
import torch
import torch.nn as nn
import os, sys, json, codecs
from utils import AdditiveAttention
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

class BertAsaMe(BertPreTrainedModel):
    def __init__(self, config, tok2word_strategy, max_relative_position):
        super(BertZP, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tok2word_strategy = tok2word_stratery # first, last, mean, sum
        hidden_size = config.hidden_size
        self.mention_st_classifier = AdditiveAttention(hidden_size, hidden_size, 384)
        self.mention_ed_classifier = AdditiveAttention(hidden_size, hidden_size, 384)


    def forward(self, batch):
        tok_repre, _ = self.bert(batch['input_ids'], None, batch['input_mask'], output_all_encoded_layers=False)
        tok_repre = self.dropout(tok_repre) # [batch, seq, dim]

        # cast from tok-level to word-level
        batch_size, seq_num, hidden_dim = list(tok_repre.size())
        _, wordseq_num, word_len = list(batch['input_tok2word'].size())
        if self.tok2word in ('first', 'last', ):
            assert word_len == 1
        offset = torch.arange(batch_size).view(batch_size, 1, 1).expand(-1, wordseq_num, word_len) * seq_num
        if torch.cuda.is_available():
            offset = offset.cuda()
        positions = (batch['input_tok2word'] + offset).view(batch_size * wordseq_num * word_len)
        word_repre = torch.index_select(tok_repre.contiguous().view(batch_size * seq_num, hidden_dim), 0, positions)
        # [batch, wordseq, wordlen, dim]
        word_repre = word_repre.view(batch_size, wordseq_num, word_len, hidden_dim)
        word_repre = word_repre * input_tok2word_mask.unsqueeze(-1)
        # [batch, wordseq, dim]
        word_repre = word_repre.mean(dim=2) if self.tok2word == 'mean' else word_repre.sum(dim=2)

        # make decision
        senti_repre = (word_repre * batch['senti_mask'].unsqueeze(-1)).sum(dim=1) # [batch, dim]
        _, st_dist = self.mention_st_classifier(senti_repre, word_repre, batch['input_content_mask']) # [batch, wordseq]
        _, ed_dist = self.mention_ed_classifier(senti_repre, word_repre, batch['input_content_mask']) # [batch, wordseq]
        predictions = torch.stack([st_dist.argmax(dim=-1), ed_dist.argmax(dim=-1)], dim=1) # [batch, 2]

        # calculate loss
        if batch['input_ref'] is not None:
            st_ref, ed_ref = batch['input_ref'].split(1, dim=-1)
            st_ref, ed_ref = st_ref.squeeze(dim=-1), ed_ref.squeeze(dim=-1) # [batch, wordseq]
            tmp = (st_ref * st_dist.log() + ed_ref * ed_dist.log()) * batch['input_content_mask'] # [batch, wordseq]
            loss = -1.0 * tmp.sum() / batch['input_content_mask'].sum()
        else:
            loss = torch.tensor(0.0).cuda() if torch.cuda.is_available() else torch.tensor(0.0)

        return {'loss': loss, 'predictions': predictions}


