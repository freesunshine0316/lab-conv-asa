
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys, json, codecs

class AdditiveAttention(nn.Module):
    def __init__(query_size, memory_size, attn_size):
        self.w = nn.Linear(query_size, attn_size)
        self.u = nn.Linear(memory_size, attn_size)
        self.v = nn.Linear(attn_size, 1)

    # query: [batch, query_size]
    # memory: [batch, seq, memory_size]
    # memory_mask: [batch, seq]
    # v^\top * tanh(W * query + U * memory)
    def forward(query, memory, memory_mask):
        assert len(query.size()) == 2 and len(memory.size()) == 3
        tmp = F.tanh(self.w(query.unsqueeze(dim=1)) + self.u(memory)) # [batch, seq, attn_size]
        tmp = self.v(tmp).squeeze(dim=2) + memory_mask.log() # [batch, seq]
        weights = F.softmax(tmp, dim=-1) # [batch, seq]
        aggr = (memory * weights.unsqueeze(dim=2)).sum(dim=1) # [batch, memory_size]
        return aggr, weights
