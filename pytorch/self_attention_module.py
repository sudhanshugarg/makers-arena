import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttentionModule(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 query_key_dim: int, 
                 seq_length: int) -> None:
        super (SelfAttentionModule, self).__init__()
        self.d_model = d_model
        self.qk_dim = query_key_dim
        self.seq_length = seq_length

        #self.token_embedding = token_embedding
        self.q = nn.Linear(d_model, query_key_dim)
        self.k = nn.Linear(d_model, query_key_dim)
        self.v = nn.Linear(d_model, d_model)
        
        #TODO mlp: [gate_proj, up_proj, down_proj] matrices

    def forward(self, x: torch.Tensor):
        # x_token is a sequence of seq_length tokens
        # and we have a batch
        # x_token is [batch_size, seq_length]

        # we need to lookup the corresponding embedding of each token
        
        #x = self.token_embedding(x_token.to(torch.int64)) #[batch_size, seq_length, d_model]
        #print(f"x_token: {x_token.shape}, x: {x.shape}")

        Q = self.q(x) # [batch_size, seq_len, query_key_dim]
        K = self.k(x) # [batch_size, seq_len, query_key_dim]
        attention = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.d_model) # [batch_size, seq_len, seq_len]
        #print(f"Q: {Q.shape}, K: {K.shape}, attention: {attention.shape}")

        #TODO masking
        soft_maxed = F.softmax(attention, dim=2) # [batch_size, seq_len, seq_len]
        V = self.v(x) # [batch_size, seq_len, d_model]

        return torch.matmul(soft_maxed, V) # [batch_size, seq_len, d_model]
