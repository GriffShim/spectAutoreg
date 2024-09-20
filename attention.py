import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, dim, num_heads, seq_len):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.kqv = nn.Linear(dim, 3 * dim)
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.register_buffer(
            "default_mask", torch.tril(torch.ones(1, 1, seq_len, seq_len))
        )

    def _attention(self, q, k, v, mask):
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)
        return attn @ v

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.kqv(x).chunk(3, dim=-1)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)

        mask = self.default_mask[:, :, :T, :T]

        out = self._attention(q, k, v, mask)
        out = rearrange(out, "b h t d -> b t (h d)")
        return out


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.linear1 = nn.Linear(dim, 4 * dim)
        self.linear2 = nn.Linear(4 * dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, seq_len):
        super().__init__()
        self.dim = dim
        self.ff = FeedForward(dim)
        self.attn = Attention(dim, num_heads, seq_len)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class AdaLnBlock(nn.Module):
    def __init__(self, dim, num_heads, seq_len):
        super().__init__()
        self.dim = dim
        self.ff = FeedForward(dim)
        self.attn = Attention(dim, num_heads, seq_len)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.cond_proj = nn.Linear(dim, 4 * dim)

    def modulate(self, x, shift, scale):
        return (x * (1 + scale)) + shift

    def forward(self, x, c):
        alpha1, beta1, alpha2, beta2 = self.cond_proj(c).chunk(4, dim=-1)
        x = x + self.attn(self.modulate(self.ln1(x), alpha1, beta1))
        x = x + self.ff(self.modulate(self.ln2(x), alpha2, beta2))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, num_heads, depth, seq_len):
        super().__init__()
        self.layers = nn.ModuleList(
            [AttentionBlock(dim, num_heads, seq_len) for _ in range(depth)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AdaLnTransformer(nn.Module):
    def __init__(self, dim, num_heads, depth, seq_len):
        super().__init__()
        self.layers = nn.ModuleList(
            [AdaLnBlock(dim, num_heads, seq_len) for _ in range(depth)]
        )

    def forward(self, x, c):
        for layer in self.layers:
            x = layer(x, c)
        return x
