import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, query_key_dim: int, seq_length: int):
        super (SelfAttentionModule, self).__init__()
        self.d_model = d_model
        self.qk_dim = query_key_dim
        self.seq_length = seq_length

        self.q = nn.Linear(d_model, query_key_dim)
        self.k = nn.Linear(d_model, query_key_dim)
        self.v = nn.Linear(d_model, d_model)
        
        #TODO mlp: [gate_proj, up_proj, down_proj] matrices

    def forward(self, x):
        # is x a set of tokens or a matrix
        # [batch_size, seq_length]
        # or is it a set of token embeddings
        # of [batch_size, seq_length, d_model] ?
        # Lets assume latter for now.: [batch_size, seq_length, d_model]

        Q = torch.matmul(x, self.q)

        # Q = torch.matmul(x, self.q.transpose(-1))
        # K = torch.matmul(x, self.k.transpose(-1))
        # attention = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)

        # multiply by mask
        # then softmax
        # then multiply by embeddings and sum up.
