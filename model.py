from torch import nn
import math
from torch.functional import F
import torch


class Config:
    """
    Just decided to chuck everything in together cus' why not.
    """

    # Model configuration.
    embed_dim = 128
    n_blocks = 1
    n_heads = 1
    seq_len = 5

    # Trainer configuration.
    epochs = 1000
    batch_size = 32
    learning_rate = 3e-4

    print_loss_every_iter = 10
    test_every_n_epochs = 10

    # Logging to wandb.
    logging = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PositionalEncoding(nn.Module):
    """
        Basically, we use sine and cosine waves at different frequencies
        along the i (0->embed_dim) and p (0->sequence_len) to create a vector
        that is unique with respect to its position in the sequence. Then we
        sum our embedding with this new "unique for position p" vector, as it
        has the same dims as our word embedding, in the hopes that the model
        can learn that there is 'positional' meaning to this pattern.

        TODO (sno6): Just replace this booshi with a learnable param.
    """
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(config.seq_len, config.embed_dim)
        position = torch.arange(0, config.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.embed_dim, 2).float() * (-math.log(10000.0) / config.embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


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

        # Mask here.

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
    def __init__(self, config, vocab_len):
        super().__init__()

        self.positional = PositionalEncoding(config)
        self.embedding = nn.Embedding(vocab_len, config.embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_blocks)])
        self.mlp = nn.Linear(config.embed_dim, vocab_len)

    def forward(self, x, y=None):
        x = self.embedding(x)
        # x = self.positional(x)
        y_hat = self.blocks(x)
        y_hat = self.mlp(y_hat)

        loss = None
        if y is not None:
            loss = F.cross_entropy(y_hat.view(-1, y_hat.size(-1)), y.view(-1), reduction="mean")

        return y_hat, loss
