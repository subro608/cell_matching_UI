"""Three independent stock UNICOM transformers, jointly trained at three scales.
IV and EX use the same transformer within a level. The soma and neighbourhood
encoders keep separate copies of the stock retrieval neck. The region encoder
uses patch tokens, so its unused retrieval neck is removed. Heads, normalization,
BN policy, level embeddings, tile stitching/mixer and thumb readout match the
shared-model experiment. All three trunks start from official stock weights.
"""
import math
import copy
import os, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .data import FIELD_SIZE, sample_map

EXP = Path(__file__).resolve().parent.parent
BU = EXP / 'marco_onestage_B_unfrozen_2026-10-08'
STOCK_CKPT = Path(os.environ.get('IVEX_UNICOM_STOCK', '/scratch/vsp7230/cell_matching/neurobridge/.cache/unicom/FP16-ViT-B-16.pt'))
STOCK_SHA256 = 'c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef'
VENDOR = Path(__file__).resolve().parent  # ivex: vendored unicom_vit.py
LEVELS = ('soma', 'neighbourhood', 'region')
ARMS = ('unicom_tiles', 'unicom_thumb')
TILE, GRID = 224, 14


class ResidualHead(nn.Module):
    """x + W2 GELU(W1 LN(x)); W2 = 0 at start, so the head starts as the identity."""
    def __init__(self, d=768):
        super().__init__()
        self.norm, self.fc1, self.fc2 = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(d, d)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


def region_head():
    return nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 768))


def rope_2d(side, dim_head, device):
    """Angles for 2-D rotary positions on a side x side token grid: half of each head's pairs rotate with the column, half with the row."""
    quarter = dim_head // 4
    freq = 1.0 / (100.0 ** (torch.arange(quarter, device=device).float() / quarter))
    ys, xs = torch.meshgrid(torch.arange(side, device=device).float(), torch.arange(side, device=device).float(), indexing='ij')
    ang = torch.cat([xs.flatten()[:, None] * freq[None], ys.flatten()[:, None] * freq[None]], dim=-1)   # (N, dim_head / 2)
    return ang.cos(), ang.sin()


def apply_rope(t, cos, sin):
    t1, t2 = t[..., 0::2], t[..., 1::2]
    return torch.stack([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1).flatten(-2)


class RopeBlock(nn.Module):
    """Pre-norm attention + MLP over the stitched token grid; rotary q/k make attention depend on RELATIVE offsets only."""
    def __init__(self, d=768, heads=12, mlp=4):
        super().__init__()
        self.h = heads
        self.n1, self.qkv, self.proj = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n2, self.fc1, self.fc2 = nn.LayerNorm(d), nn.Linear(d, mlp * d), nn.Linear(mlp * d, d)
        for m in (self.proj, self.fc2):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, cos, sin):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(apply_rope(q, cos, sin), apply_rope(k, cos, sin), v)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class SeparateUnicomThreeLevel(nn.Module):
    def __init__(self, arm):
        super().__init__()
        assert arm in ARMS
        from .unicom_vit import load_model_and_transform
        with torch.random.fork_rng(devices=[]):
            self.backbone, _ = load_model_and_transform('ViT-B/16')
        stock = torch.load(STOCK_CKPT, map_location='cpu', weights_only=True)
        self.backbone.load_state_dict(stock, strict=True)
        # Check every tensor, including the retrieval neck and BatchNorm buffers.
        for name, value in self.backbone.state_dict().items():
            assert torch.equal(value, stock[name].to(value.dtype)), name
        self.stock_tensor_count = len(stock)
        del stock
        base = self.backbone
        del self.backbone
        self.backbones = nn.ModuleDict({LEVELS[0]: base, LEVELS[1]: copy.deepcopy(base), LEVELS[2]: copy.deepcopy(base)})
        # The field level consumes patch tokens, never the image retrieval neck.
        # Remove that unused neck; all three transformer trunks remain independent.
        self.backbones['region'].feature = nn.Identity()
        self.register_buffer('pixel_mean', torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('pixel_std', torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1), persistent=False)
        self.arm, self.region_mode = arm, ('tiles' if arm == 'unicom_tiles' else 'thumb')
        self.level_embed = nn.Parameter(torch.zeros(len(LEVELS), 768))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1729)                                             # identical fresh heads in both arms
            self.heads = nn.ModuleDict(dict(soma=ResidualHead(), neighbourhood=ResidualHead(), region=region_head()))
            self.mixer = nn.ModuleList([RopeBlock() for _ in range(2)] if self.region_mode == 'tiles' else [])
        self.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        for backbone in self.backbones.values():
            for m in backbone.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()  # retain shared-run BN policy for a matched comparison
        return self

    @property
    def device(self):
        return self.pixel_mean.device

    def pixels(self, gray):
        x = torch.as_tensor(gray, device=self.device)[:, None].float().div(255).expand(-1, 3, -1, -1)
        return (x - self.pixel_mean) / self.pixel_std                          # stock UNICOM CLIP normalization

    def tokens(self, gray, level):
        t = self.backbones[level]
        x = t.patch_embed(self.pixels(gray)) + t.pos_embed + self.level_embed[LEVELS.index(level)]
        for blk in t.blocks:
            x = checkpoint(blk, x, use_reentrant=False) if (self.training and torch.is_grad_enabled()) else blk(x)
        return t.norm(x.float())                                                # (B, 196, 768)

    def crop_level(self, gray, level, chunk=32):
        out = []
        for s in range(0, len(gray), chunk):
            x = self.tokens(gray[s:s + chunk], level)
            z = self.backbones[level].feature(torch.reshape(x, (len(x), -1)))          # stock neck: 196 x 768 -> 768
            out.append(F.normalize(self.heads[level](z), dim=-1))
        return torch.cat(out)

    def region_maps(self, images):
        """(n, 672, 672) uint8 letterboxed fields -> (n, 768, side, side) token maps covering the whole 672 frame."""
        n = len(images)
        if self.region_mode == 'thumb':
            small = np.stack([np.asarray(Image.fromarray(im).resize((TILE, TILE), Image.Resampling.BILINEAR)) for im in images])
            return self.tokens(small, 'region').transpose(1, 2).reshape(n, 768, GRID, GRID)
        g = FIELD_SIZE // TILE
        tiles = np.stack([im[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] for im in images for r in range(g) for c in range(g)])
        x = self.tokens(tiles, 'region')                                       # (n g g, 196, 768)
        side = g * GRID
        x = x.reshape(n, g, g, GRID, GRID, 768).permute(0, 1, 3, 2, 4, 5).reshape(n, side * side, 768)   # tiles back at their true positions
        cos, sin = rope_2d(side, 768 // 12, x.device)
        for blk in self.mixer:
            x = checkpoint(blk, x, cos, sin, use_reentrant=False) if (self.training and torch.is_grad_enabled()) else blk(x, cos, sin)
        return x.transpose(1, 2).reshape(n, 768, side, side)

    def field_level(self, images, ids, coordinates, chunk=2):
        values, indices, keys = [], [], list(images)
        for s in range(0, len(keys), chunk):
            bk = keys[s:s + chunk]
            maps = self.region_maps(np.stack([images[k] for k in bk]))
            for i, k in enumerate(bk):
                sel = np.flatnonzero(ids == k)
                sampled = sample_map(maps[i:i + 1], torch.as_tensor(coordinates[sel], device=self.device))   # 672-frame coordinates
                values.append(F.normalize(self.heads['region'](sampled), dim=-1))
                indices.append(torch.as_tensor(sel, device=self.device))
        order = torch.argsort(torch.cat(indices))
        return torch.cat(values)[order]

    def describe(self, a, fields, prefix, indices, crop_chunk=32):
        indices = np.asarray(indices, np.int64)
        local = self.crop_level(a[prefix + '_local'][indices], 'soma', crop_chunk)
        context = self.crop_level(a[prefix + '_global'][indices], 'neighbourhood', crop_chunk)
        if prefix == 'q':
            ids = a['q_z'][indices]
            images = {int(z): fields['iv'][int(z)] for z in np.unique(ids)}
            xy = fields['q_xy'][indices]
        else:
            ids = np.zeros(len(indices), np.int64)
            images = {0: fields['ex']}
            xy = fields['ex_xy'][indices]
        region = self.field_level(images, ids, xy)
        return torch.stack([local, context, region], dim=1)

    @staticmethod
    def combine(levels):
        return levels.flatten(1) / math.sqrt(3)

    def forward(self, a, fields, qids, cids):
        return self.describe(a, fields, 'q', qids), self.describe(a, fields, 'ex', cids)
