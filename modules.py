import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F


# DATA CLASS CONFIG OBJECT
@dataclass
class GPTConfig:
  block_size: int = 1024 # context window (max seq length)
  vocab_size: int = 50257 # number of tokens
  n_layers: int = 12
  n_head: int = 12
  n_embd: int = 768


# MULTI-LAYER PERCEPTRON
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# CAUSAL SELF-ATTENTION BLOCK
class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0

    # key, query, value projection for all heads, but in a batch
    # multiplying with 3 to get 3 projections: key, value, query
    self.c_attn = nn.Linear(config.n_embd, 3* config.n_embd)

    #output projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1

    # regularization
    self.n_head = config.n_head
    self.n_embd = config.n_embd

    # masking causal masking lookup so that the model does not look into the future context and uses the previous tokens as its context.
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                            .view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    B,T,C = x.shape # batch size, sequence length, embedding dimensionality

    # apply projection to get the vectors
    qkv = self.c_attn(x)

    q, k, v = qkv.split(self.n_embd, dim=2)

    # This operation is done on batches in parallel
    k = k.view(B,T, self.n_head, C//self.n_head).transpose(1,2) # (Batch_size, num_heads, sequence_length, head_size)
    q = q.view(B,T, self.n_head, C//self.n_head).transpose(1,2) # (Batch_size, num_heads, sequence_length, head_size)
    v = v.view(B,T, self.n_head, C//self.n_head).transpose(1,2) # (Batch_size, num_heads, sequence_length, head_size)

    # attention machanism
    '''att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:, :, :T, :T]==0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v'''

    # flash attention 
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)


    y = y.transpose(1, 2).contiguous().view(B,T,C) # reassemble all head outputs

    # output projection
    y = self.c_proj(y)

    return y
  
# BLOCK
class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x+ self.mlp(self.ln_2(x))
    return x

