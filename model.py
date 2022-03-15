from torch import nn
import math
from torch.functional import F


class GPTConfig:
    embed_dim = 128
    n_blocks = 2
    n_heads = 4

    max_seq_len = 5
    vocab_len = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MultiheadSelfAttention(nn.Module):
    """
    Shamelessly ripped from lord Karpathy.
    """
    def __init__(self, config):
        super().__init__()

        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        self.n_heads = config.n_heads

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = MultiheadSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.embed_dim, config.embed_dim)
        self.mlp = nn.Linear(config.embed_dim, config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim, config.embed_dim)

    def forward(self, x):
        y_hat = self.ln1(x + self.attn(x))
        y_hat = self.ln2(y_hat + self.mlp(y_hat))
        return y_hat


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_len, config.embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_blocks)])
        self.mlp = nn.Linear(config.embed_dim * config.max_seq_len, config.vocab_len)

    def forward(self, x, y=None):
        x = x.squeeze(0)

        x = self.embedding(x)
        y_hat = self.blocks(x)
        y_hat = y_hat.view(-1, y_hat.size(1) * y_hat.size(2)).contiguous()
        y_hat = self.mlp(y_hat)

        loss = None
        if y is not None:
            loss = F.cross_entropy(y_hat, y.squeeze(-1), reduction="mean")

        return y_hat, loss
