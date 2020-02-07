
import torch
import torch.nn as nn
import os, sys, json, codecs

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

class BertAsaMe(BertPreTrainedModel):
    def __init__(self, config, char2word, pro_num, max_relative_position):
        super(BertZP, self).__init__(config)
        assert type(pro_num) is int and pro_num > 1
        self.pro_num = pro_num
        self.char2word = char2word
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.resolution_classifier = SpanClassifier(config.hidden_size)
        self.detection_classifier = nn.Linear(config.hidden_size, 2)
        self.recovery_classifier = nn.Linear(config.hidden_size, pro_num)


    def forward(self, input_ids, mask, decision_mask, word_mask, input_char2word, input_char2word_mask,
            detection_refs, resolution_refs, recovery_refs, batch_type):
        char_repre, _ = self.bert(input_ids, None, mask, output_all_encoded_layers=False)
        char_repre = self.dropout(char_repre) # [batch, seq, dim]

        # cast from char-level to word-level
        batch_size, seq_num, hidden_dim = list(char_repre.size())
        _, wordseq_num, word_len = list(input_char2word.size())
        if self.char2word in ('first', 'last', ):
            assert word_len == 1
        offset = torch.arange(batch_size).view(batch_size, 1, 1).expand(-1, wordseq_num, word_len) * seq_num
        if torch.cuda.is_available():
            offset = offset.cuda()
        positions = (input_char2word + offset).view(batch_size * wordseq_num * word_len)
        word_repre = torch.index_select(char_repre.contiguous().view(batch_size * seq_num, hidden_dim), 0, positions)
        word_repre = word_repre.view(batch_size, wordseq_num, word_len, hidden_dim)
        word_repre = word_repre * input_char2word_mask.unsqueeze(-1)
        # word_repre: [batch, wordseq, dim]
        word_repre = word_repre.mean(dim=2) if self.char2word == 'mean' else word_repre.sum(dim=2)

        #detection
        detection_logits = self.detection_classifier(word_repre) # [batch, wordseq, 2]
        detection_outputs = detection_logits.argmax(dim=-1) # [batch, wordseq]
        detection_loss = torch.tensor(0.0)
        if torch.cuda.is_available():
            detection_loss = detection_loss.cuda()
        if detection_refs is not None:
            detection_loss = token_classification_loss(detection_logits, 2, detection_refs, decision_mask)

        #resolution
        if batch_type == 'resolution':
            resolution_start_dist, resolution_end_dist = self.resolution_classifier(word_repre, word_mask)
            resolution_start_outputs = resolution_start_dist.argmax(dim=-1) # [batch, wordseq]
            resolution_end_outputs = resolution_end_dist.argmax(dim=-1) # [batch, wordseq]
            resolution_outputs = torch.stack([resolution_start_outputs, resolution_end_outputs], dim=-1) # [batch, wordseq, 2]
            resolution_loss = torch.tensor(0.0)
            if torch.cuda.is_available():
                resolution_loss = resolution_loss.cuda()
            if resolution_refs is not None: # [batch, wordseq, wordseq, 2]
                resolution_start_positions, resolution_end_positions = resolution_refs.split(1, dim=-1)
                resolution_start_positions = resolution_start_positions.squeeze(dim=-1) # [batch, wordseq, wordseq]
                resolution_end_positions = resolution_end_positions.squeeze(dim=-1) # [batch, wordseq, wordseq]
                resolution_loss = span_loss(resolution_start_dist, resolution_end_dist,
                        resolution_start_positions, resolution_end_positions, decision_mask)
            total_loss = detection_loss + resolution_loss
            return {'total_loss': total_loss, 'detection_loss': detection_loss, 'resolution_loss': resolution_loss}, \
                   {'detection_outputs': detection_outputs, 'resolution_outputs': resolution_outputs,
                    'resolution_start_dist': resolution_start_dist, 'resolution_end_dist': resolution_end_dist}

        #recovery
        if batch_type == 'recovery':
            recovery_logits = self.recovery_classifier(word_repre) # [batch, wordseq, pro_num]
            recovery_outputs = recovery_logits.argmax(dim=-1) # [batch, wordseq]
            recovery_loss = torch.tensor(0.0)
            if torch.cuda.is_available():
                recovery_loss = recovery_loss.cuda()
            if recovery_refs is not None:
                recovery_loss = token_classification_loss(recovery_logits, self.pro_num, recovery_refs, decision_mask)
            total_loss = detection_loss + recovery_loss
            return {'total_loss': total_loss, 'detection_loss': detection_loss, 'recovery_loss': recovery_loss}, \
                   {'detection_outputs': detection_outputs, 'recovery_outputs': recovery_outputs}

        assert False, "batch_type need to be either 'recovery' or 'resolution'"


