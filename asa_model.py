
import torch
import torch.nn as nn
import os, sys, json, codecs
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from nn_utils import AdditiveAttention, gather_tok2word, has_nan
from asa_datastream import TAG_MAPPING
from tencent_transformer import MultiheadAttention, TransformerEncoderLayer


class BertAsaSe(BertPreTrainedModel):
    def __init__(self, config):
        super(BertAsaSe, self).__init__(config)
        self.embed = None
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.encoder = nn.ModuleList([TransformerEncoderLayer(config.hidden_size, 8, dim_feedforward=256) \
                for i in range(4)])
        self.label_num = len(TAG_MAPPING)
        self.classifier = nn.Sequential(torch.nn.Linear(config.hidden_size, 384),
                nn.ReLU(), nn.Linear(384, self.label_num))


    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False


    def setup_embedding(self, vocab_size):
        self.embed = nn.Embedding(vocab_size, 768)


    def forward(self, batch):
        # embedding
        if self.embed is None:
            tok_repre, _ = self.bert(batch['input_ids'], None, batch['input_mask'], output_all_encoded_layers=False)
            tok_repre = self.dropout(tok_repre) # [batch, seq, dim]
        else:
            tok_repre = self.embed(batch['input_ids']) # [batch, seq, dim]

        # cast from tok-level to word-level
        word_repre = gather_tok2word(tok_repre, batch['input_tok2word'], batch['input_tok2word_mask']) # [batch, wordseq, dim]

        # encode
        word_repre = word_repre.transpose(0,1).contiguous()
        word_mask_bool = batch['input_tok2word_mask'].sum(dim=2) > 0
        for encoder_layer in self.encoder:
            word_repre = encoder_layer(word_repre, src_key_padding_mask=~word_mask_bool)
        word_repre = word_repre.transpose(0,1).contiguous()
        assert has_nan(word_repre) == False

        batch_size, wordseq_len, _ = list(word_repre.size())
        total_len = batch_size * wordseq_len

        # make predictions
        logits = self.classifier(word_repre).log_softmax(dim=2) # [batch, wordseq, label]
        predictions = logits.argmax(dim=2) # [batch, wordseq]

        if batch['input_tags'] is not None:
            active_positions = word_mask_bool.view(total_len) # [batch * wordseq]
            active_logits = logits.view(total_len, self.label_num)[active_positions]
            active_refs = batch['input_tags'].view(total_len)[active_positions]
            loss = nn.CrossEntropyLoss()(active_logits, active_refs)
        else:
            loss = torch.tensor(0.0).cuda() if torch.cuda.is_available() else torch.tensor(0.0)

        wordseq_mask_bool = batch['input_tok2word_mask'].sum(dim=2) > 0
        wordseq_lengths = wordseq_mask_bool.long().sum(dim=1)
        return {'loss':loss, 'predictions':predictions, 'wordseq_lengths':wordseq_lengths}


class BertAsaMe(BertPreTrainedModel):
    def __init__(self, config):
        super(BertAsaMe, self).__init__(config)
        self.embed = None
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        hidden_size = config.hidden_size
        self.st_classifier = AdditiveAttention(hidden_size, hidden_size, 384)
        self.ed_classifier = AdditiveAttention(hidden_size, hidden_size, 384)


    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False


    def setup_embedding(self, vocab_size):
        self.embed = nn.Embedding(vocab_size, 768)


    def forward(self, batch):
        # embedding
        if self.embed == None:
            tok_repre, _ = self.bert(batch['input_ids'], None, batch['input_mask'], output_all_encoded_layers=False)
            tok_repre = self.dropout(tok_repre) # [batch, seq, dim]
        else:
            tok_repre = self.embed(batch['input_ids'])

        # cast from tok-level to word-level
        word_repre = gather_tok2word(tok_repre, batch['input_tok2word'], batch['input_tok2word_mask']) # [batch, wordseq, dim]

        wordseq_mask_bool = batch['input_tok2word_mask'].sum(dim=2) > 0
        wordseq_lengths = wordseq_mask_bool.long().sum(dim=1)

        # generate final distribution
        senti_repre = (word_repre * batch['input_senti_mask'].unsqueeze(-1)).sum(dim=1) # [batch, dim]
        _, st_dist = self.st_classifier(senti_repre, word_repre, batch['input_content_mask']) # [batch, wordseq]
        _, ed_dist = self.ed_classifier(senti_repre, word_repre, batch['input_content_mask']) # [batch, wordseq]
        assert has_nan(st_dist.log()) == False and has_nan(ed_dist.log()) == False
        final_dist = st_dist.unsqueeze(dim=2) * ed_dist.unsqueeze(dim=1) # [batch, wordseq, wordseq]

        # make predictions
        batch_size, wordseq_num, bert_dim = list(word_repre.size())
        a = torch.arange(wordseq_num).view(1, wordseq_num, 1)
        b = torch.arange(wordseq_num).view(1, 1, wordseq_num)
        if torch.cuda.is_available():
            a = a.cuda()
            b = b.cuda()
        mask = (a <= b).float() # [batch, wordseq, wordseq]
        span_mask = (torch.abs(a - b) <= 6).float()
        sentid = batch['input_sentid']
        sentid_mask = (sentid.unsqueeze(dim=1) == sentid.unsqueeze(dim=2)).float() # [batch, wordseq, wordseq]
        predictions = (final_dist * mask * span_mask * sentid_mask).view(batch_size,
                wordseq_num * wordseq_num).argmax(dim=1) # [batch]

        # calculate loss
        if batch['input_ref'] is not None:
            st_ref, ed_ref = batch['input_ref'].split(1, dim=-1)
            st_ref, ed_ref = st_ref.squeeze(dim=-1), ed_ref.squeeze(dim=-1) # [batch, wordseq]
            assert has_nan(st_ref) == False and has_nan(ed_ref) == False
            tmp_st = st_ref * (st_dist.log())
            tmp_ed = ed_ref * (ed_dist.log())
            assert has_nan(tmp_st) == False and has_nan(tmp_ed) == False
            tmp = (tmp_st + tmp_ed) * batch['input_content_mask'] # [batch, wordseq]
            loss = -1.0 * tmp.sum() / batch['input_content_mask'].sum()
        else:
            loss = torch.tensor(0.0).cuda() if torch.cuda.is_available() else torch.tensor(0.0)

        return {'loss': loss, 'predictions': predictions, 'wordseq_num': wordseq_num}


