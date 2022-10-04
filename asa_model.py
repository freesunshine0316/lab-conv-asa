import torch
import torch.nn as nn
from transformers import BertModel
from nn_utils import AdditiveAttention, has_nan
from asa_datastream import TAG_MAPPING


class BertAsaSe(nn.Module):

    def __init__(self, path):
        super(BertAsaSe, self).__init__()
        self.embed = None
        self.bert = BertModel.from_pretrained(path)
        self.dropout = nn.Dropout(0.1)
        self.label_num = len(TAG_MAPPING)
        self.classifier = nn.Sequential(torch.nn.Linear(self.bert.config.hidden_size, 384), nn.ReLU(),
                                        nn.Linear(384, self.label_num))

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def setup_embedding(self, vocab_size):
        self.embed = nn.Embedding(vocab_size, 768)

    def forward(self, batch):
        # embedding
        if self.embed is None:
            tok_repre = self.bert(input_ids=batch['input_ids'], attention_mask=batch['input_mask'],
                                  return_dict=True).last_hidden_state
            tok_repre = self.dropout(tok_repre)  # [batch, seq, dim]
        else:
            tok_repre = self.embed(batch['input_ids'])  # [batch, seq, dim]

        batch_size, seq_len, _ = list(tok_repre.size())
        total_len = batch_size * seq_len

        # make predictions
        logits = self.classifier(tok_repre).log_softmax(dim=2)  # [batch, seq, label]
        predictions = logits.argmax(dim=2)  # [batch, seq]

        if batch['input_tags'] is not None:
            active_positions = (batch['input_mask'] == 1.0).view(total_len)  # [batch * seq]
            active_logits = logits.view(total_len, self.label_num)[active_positions]
            active_refs = batch['input_tags'].view(total_len)[active_positions]
            loss = nn.CrossEntropyLoss()(active_logits, active_refs)
        else:
            loss = torch.tensor(0.0).cuda() if torch.cuda.is_available() else torch.tensor(0.0)

        seq_lengths = batch['input_mask'].sum(dim=1).long()
        return {'loss': loss, 'predictions': predictions, 'seq_lengths': seq_lengths}


class BertAsaMe(nn.Module):

    def __init__(self, path):
        super(BertAsaMe, self).__init__()
        self.embed = None
        self.bert = BertModel.from_pretrained(path)
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.bert.config.hidden_size
        self.st_classifier = AdditiveAttention(hidden_size, hidden_size, 384)
        self.ed_classifier = AdditiveAttention(hidden_size, hidden_size, 384)

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def setup_embedding(self, vocab_size):
        self.embed = nn.Embedding(vocab_size, 768)

    def get_bert_embedding(self, batch):
        tokseq_num = batch['input_ids'].shape[1]
        if tokseq_num > 512:
            tok_repre = []
            st = 0
            while st < tokseq_num:
                ed = min(tokseq_num, st + 512)
                cur_ids = batch['input_ids'][:, st:ed]
                cur_mask = batch['input_mask'][:, st:ed]
                cur_tok_repre = self.bert(input_ids=cur_ids, attention_mask=cur_mask,
                                          return_dict=True).last_hidden_state
                tok_repre.append(cur_tok_repre)
                st = ed
            tok_repre = torch.cat(tok_repre, dim=1)
        else:
            tok_repre = self.bert(input_ids=batch['input_ids'], attention_mask=batch['input_mask'],
                                  return_dict=True).last_hidden_state
        return tok_repre

    def forward(self, batch):
        # embedding
        if self.embed == None:
            tok_repre = self.get_bert_embedding(batch)  # [batch, seq, dim]
            tok_repre = self.dropout(tok_repre)  # [batch, seq, dim]
        else:
            tok_repre = self.embed(batch['input_ids'])

        # generate final distribution
        senti_repre = (tok_repre * batch['input_senti_mask'].unsqueeze(-1)).sum(dim=1)  # [batch, dim]
        _, st_dist = self.st_classifier(senti_repre, tok_repre, batch['input_content_mask'])  # [batch, seq]
        _, ed_dist = self.ed_classifier(senti_repre, tok_repre, batch['input_content_mask'])  # [batch, seq]
        assert has_nan(st_dist.log()) == False and has_nan(ed_dist.log()) == False
        final_dist = st_dist.unsqueeze(dim=2) * ed_dist.unsqueeze(dim=1)  # [batch, wordseq, wordseq]

        # make predictions
        batch_size, seq_num, bert_dim = list(tok_repre.size())
        a = torch.arange(seq_num).view(1, seq_num, 1)
        b = torch.arange(seq_num).view(1, 1, seq_num)
        if torch.cuda.is_available():
            a = a.cuda()
            b = b.cuda()
        mask = (a <= b).float()  # [batch, seq, seq]
        span_mask = (torch.abs(a - b) <= 6).float()
        sentid = batch['input_sentids']
        sentid_mask = (sentid.unsqueeze(dim=1) == sentid.unsqueeze(dim=2)).float()  # [batch, seq, seq]
        predictions = (final_dist * mask * span_mask * sentid_mask).view(batch_size,
                                                                         seq_num * seq_num).argmax(dim=1)  # [batch]

        # calculate loss
        if batch['input_ref'] is not None:
            st_ref, ed_ref = batch['input_ref'].split(1, dim=-1)  # [batch, seq, 1]
            st_ref, ed_ref = st_ref.squeeze(dim=-1), ed_ref.squeeze(dim=-1)  # [batch, seq]
            assert has_nan(st_ref) == False and has_nan(ed_ref) == False
            tmp_st = st_ref * (st_dist.log())
            tmp_ed = ed_ref * (ed_dist.log())
            assert has_nan(tmp_st) == False and has_nan(tmp_ed) == False
            tmp = (tmp_st + tmp_ed) * batch['input_content_mask']  # [batch, seq]
            loss = -1.0 * tmp.sum() / batch['input_content_mask'].sum()
        else:
            loss = torch.tensor(0.0).cuda() if torch.cuda.is_available() else torch.tensor(0.0)

        return {'loss': loss, 'predictions': predictions, 'seq_num': seq_num}
