from torch import nn
import math
from torch.functional import F
import torch


class Config:
    """
    Just decided to chuck everything in together cus' why not.
    """

    # Model configuration.
    embed_dim = 512
    n_blocks = 8
    n_heads = 8
    seq_len = 128

    # Trainer configuration.
    epochs = 16
    batch_size = 128
    learning_rate = 4e-4

    print_loss_every_iter = 10
    test_every_n_epochs = 10
    save_chkpt_every_n_epochs = 5

    # Logging to wandb.
    logging = False

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

        self.register_buffer("mask", torch.tril(torch.ones(config.seq_len, config.seq_len))
                             .view(1, 1, config.seq_len, config.seq_len))

        self.n_heads = config.n_heads

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = MultiheadSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.embed_dim, config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim, config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, vocab_len):
        super().__init__()

        self.positional = nn.Parameter(torch.zeros(1, config.seq_len, config.embed_dim))
        self.embedding = nn.Embedding(vocab_len, config.embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_blocks)])
        self.mlp = nn.Linear(config.embed_dim, vocab_len)

    def forward(self, x, y=None):
        b, t = x.size()

        x = self.embedding(x)
        y_hat = self.blocks(x + self.positional[:, :t, :])
        y_hat = self.mlp(y_hat)

        loss = None
        if y is not None:
            loss = F.cross_entropy(y_hat.view(-1, y_hat.size(-1)), y.view(-1), reduction="mean")

        return y_hat, loss
