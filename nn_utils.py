
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys, json, codecs


def is_nan(tensor):
    return torch.isnan(tensor).any().item()


# tok_repre: [batch, seq, dim]
# input_tok2word: [batch, wordseq, wordlen]
def gather_tok2word(tok_repre, input_tok2word, input_tok2word_mask):
    batch_size, seq_num, hidden_dim = list(tok_repre.size())
    _, wordseq_num, word_len = list(input_tok2word.size())
    offset = torch.arange(batch_size).view(batch_size, 1, 1).expand(-1, wordseq_num, word_len) * seq_num
    if torch.cuda.is_available():
        offset = offset.cuda()
    positions = (input_tok2word + offset).view(batch_size * wordseq_num * word_len)
    word_repre = torch.index_select(tok_repre.contiguous().view(batch_size * seq_num, hidden_dim), 0, positions)
    # [batch, wordseq, wordlen, dim]
    word_repre = word_repre.view(batch_size, wordseq_num, word_len, hidden_dim)
    word_repre = word_repre * input_tok2word_mask.unsqueeze(-1)
    # [batch, wordseq, dim]
    word_repre = word_repre.mean(dim=2)
    return word_repre


class AdditiveAttention(nn.Module):
    def __init__(self, query_size, memory_size, attn_size):
        super(AdditiveAttention, self).__init__()
        self.w = nn.Linear(query_size, attn_size)
        self.u = nn.Linear(memory_size, attn_size)
        self.v = nn.Linear(attn_size, 1)

    # query: [batch, query_size]
    # memory: [batch, seq, memory_size]
    # memory_mask: [batch, seq]
    # v^\top * tanh(W * query + U * memory)
    def forward(self, query, memory, memory_mask):
        assert len(query.size()) == 2 and len(memory.size()) == 3
        tmp = F.tanh(self.w(query.unsqueeze(dim=1)) + self.u(memory)) # [batch, seq, attn_size]
        tmp = self.v(tmp).squeeze(dim=2) + memory_mask.log() # [batch, seq]
        print(tmp)
        weights = F.softmax(tmp, dim=-1) # [batch, seq]
        assert is_nan(weights) == False
        aggr = (memory * weights.unsqueeze(dim=2)).sum(dim=1) # [batch, memory_size]
        return aggr, weights
