
import torch
import torch.nn as nn
import os, sys, json, codecs
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from nn_utils import AdditiveAttention, gather_tok2word
from asa_datastream import TAG_MAPPING


class BertAsaSe(BertPreTrainedModel):
    def __init__(self, config):
        super(BertAsaSe, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.label_num = len(TAG_MAPPING)
        self.classifier = nn.Linear(config.hidden_size, self.label_num)


    def forward(self, batch):
        # bert encoding
        tok_repre, _ = self.bert(batch['input_ids'], None, batch['input_mask'], output_all_encoded_layers=False)
        tok_repre = self.dropout(tok_repre) # [batch, seq, dim]

        # cast from tok-level to word-level
        word_repre = gather_tok2word(tok_repre, batch['input_tok2word'], batch['input_tok2word_mask']) # [batch, wordseq, dim]

        # make predictions
        logits = self.classifier(word_repre) # [batch, wordseq, label]
        predictions = logits.argmax(dim=2) # [batch, wordseq]

        if batch['input_tags'] is not None:
            active_positions = batch['input_tok2word_mask'].sum(dim=2).view(-1) > 0 # [batch * wordseq]
            active_logits = logits.view(-1, self.label_num)[active_positions]
            active_refs = batch['input_tags'].view(-1)[active_positions]
            loss = nn.CrossEntropyLoss()(active_logits, active_refs)
        else:
            loss = torch.tensor(0.0).cuda() if torch.cuda.is_available() else torch.tensor(0.0)

        wordseq_mask_bool = batch['input_tok2word_mask'].sum(dim=2) > 0
        wordseq_lengths = wordseq_mask_bool.long().sum(dim=1)
        return {'loss':loss, 'predictions':predictions, 'wordseq_lengths':wordseq_lengths}


class BertAsaMe(BertPreTrainedModel):
    def __init__(self, config):
        super(BertAsaMe, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        hidden_size = config.hidden_size
        self.st_classifier = AdditiveAttention(hidden_size, hidden_size, 384)
        self.ed_classifier = AdditiveAttention(hidden_size, hidden_size, 384)


    def forward(self, batch):
        # bert encoding
        tok_repre, _ = self.bert(batch['input_ids'], None, batch['input_mask'], output_all_encoded_layers=False)
        tok_repre = self.dropout(tok_repre) # [batch, seq, dim]

        # cast from tok-level to word-level
        word_repre = gather_tok2word(tok_repre, batch['input_tok2word'], batch['input_tok2word_mask']) # [batch, wordseq, dim]

        wordseq_mask_bool = batch['input_tok2word_mask'].sum(dim=2) > 0
        wordseq_lengths = wordseq_mask_bool.long().sum(dim=1)
        print(wordseq_lengths)

        # generate final distribution
        senti_repre = (word_repre * batch['input_senti_mask'].unsqueeze(-1)).sum(dim=1) # [batch, dim]
        _, st_dist = self.st_classifier(senti_repre, word_repre, batch['input_content_mask']) # [batch, wordseq]
        _, ed_dist = self.ed_classifier(senti_repre, word_repre, batch['input_content_mask']) # [batch, wordseq]
        final_dist = st_dist.unsqueeze(dim=2) * ed_dist.unsqueeze(dim=1) # [batch, wordseq, wordseq]

        # make predictions
        batch_size, wordseq_num, bert_dim = list(word_repre.size())
        a = torch.arange(wordseq_num).view(1, wordseq_num, 1)
        b = torch.arange(wordseq_num).view(1, 1, wordseq_num)
        if torch.cuda.is_available():
            a = a.cuda()
            b = b.cuda()
        mask = (a <= b).float() # [batch, wordseq, wordseq]
        predictions = (final_dist * mask).view(batch_size, wordseq_num * wordseq_num).argmax(dim=1) # [batch]

        # calculate loss
        if batch['input_ref'] is not None:
            st_ref, ed_ref = batch['input_ref'].split(1, dim=-1)
            st_ref, ed_ref = st_ref.squeeze(dim=-1), ed_ref.squeeze(dim=-1) # [batch, wordseq]
            tmp = (st_ref * st_dist.log() + ed_ref * ed_dist.log()) * batch['input_content_mask'] # [batch, wordseq]
            loss = -1.0 * tmp.sum() / batch['input_content_mask'].sum()
        else:
            loss = torch.tensor(0.0).cuda() if torch.cuda.is_available() else torch.tensor(0.0)

        return {'loss': loss, 'predictions': predictions, 'wordseq_num': wordseq_num}


