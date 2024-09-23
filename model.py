import torch.nn as nn
from attention import AdaLnTransformer
from fft_framework import simpDiff

class penis_net(nn.Module):
    def __init__(self, img_size, num_classes, hidden_dim, slice_points, attn_heads=8, attn_depth=6):
        super().__init__()
        h, w = img_size
        width = w // 2 + 1
        self.slice_points = slice_points
        self._diff_obj = simpDiff(img_size, slice_points)
        total_timesteps = self._diff_obj.max_t + 1
        self.transformer = Transformer(hidden_dim, attn_heads, attn_depth, total_timesteps + 1) # +1 for class embed
        self.proj1 = nn.Linear(2 * 3 * h * width, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, 2 * 3 * h * width)
        self.class_embed = nn.Embedding(num_classes, hidden_dim)

    def forward(self, x, c):
        c = self.class_embed(c) # B
        c = c.unsqueeze(1) # B x 1 x C
        x = self.proj1(x)
        x = torch.cat([c, x], axis=1) # B x T + 1 x C
        x = self.transformer(x)
        x = self.proj2(x)
        return x

class adaptive_penis_net(nn.Module):
    def __init__(
        self,
        img_size,
        num_classes,
        hidden_dim,
        slice_points,
        attn_heads=8,
        attn_depth=8,
    ):
        super().__init__()
        h, w = img_size
        width = w // 2 + 1
        self.slice_points = slice_points
        self._diff_obj = simpDiff(img_size, slice_points)
        total_timesteps = self._diff_obj.max_t + 1
        self.transformer = AdaLnTransformer(
            hidden_dim, attn_heads, attn_depth, total_timesteps
        )
        self.proj1 = nn.Linear(2 * 3 * h * width, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, 2 * 3 * h * width)
        self.class_embed = nn.Embedding(num_classes, hidden_dim)

    def forward(self, x, c):
        c = self.class_embed(c)  # B
        c = c.unsqueeze(1)  # B x 1 x C
        x = self.proj1(x)
        x = self.transformer(x, c)
        x = self.proj2(x)
        return x
