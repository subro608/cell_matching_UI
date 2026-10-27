"""Joint PMAC: stock UNICOM, spatial correspondence and conditional cell identity.

Five predeclared ablations. All train in one optimizer run; no MARCO or task
checkpoint is used. Shared arms register exactly one stock UNICOM instance.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .base_model import PMAC as OriginalPMAC, location_targets
from .model_unicom import LEVELS, STOCK_CKPT, STOCK_SHA256, VENDOR
from .data import FIELD_SIZE, TRAIN

TAU = .07
LOC_WEIGHT = .25
LOCAL_WEIGHT = .5
LOCAL_RADIUS_PX = 50.
MEMBERSHIP_SIGMA_BINS = .75
ARMS = ('independent_local', 'shared_base', 'shared_local',
        'shared_query_local', 'shared_query_conditional')


def positions(xy):
    """Fourier positions in each image's own padded coordinate frame."""
    freq = xy.new_tensor([1., 2., 4., 8.]) * math.pi
    phase = (2 * (xy + .5) / FIELD_SIZE - 1)[..., None] * freq
    return torch.cat([phase.sin(), phase.cos()], -1).flatten(-2)


class QueryBlock(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))

    def forward(self, q, memory, valid):
        # No query-query attention: a cell prediction is independent of query
        # batching, and the target gallery never serves as supervision here.
        q = q + self.attn(self.norm(q)[None], memory[None], memory[None],
                          key_padding_mask=~valid[None], need_weights=False)[0][0]
        return q + self.ff(q)


class QueryConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(3*768), nn.Linear(3*768, 256))
        self.memory = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256))
        self.position = nn.Linear(16, 256, bias=False)
        self.blocks = nn.ModuleList([QueryBlock(), QueryBlock()])
        self.output = nn.Linear(256, 768)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, q, keys, xy, valid):
        y, x = torch.meshgrid(torch.arange(14, device=q.device), torch.arange(14, device=q.device), indexing='ij')
        centres = (torch.stack([x, y], -1).reshape(-1, 2).float() + .5) * FIELD_SIZE / 14 - .5
        memory = self.memory(keys) + self.position(positions(centres))
        value = self.query(q.flatten(1)) + self.position(positions(xy))
        for block in self.blocks:
            value = block(value, memory, valid)
        return F.normalize(q[:, 2] + self.output(value), dim=-1)


class JointPMAC(OriginalPMAC):
    def __init__(self, arm):
        assert arm in ARMS
        super().__init__()
        self.arm = arm
        self.shared = arm != 'independent_local'
        self.use_local = arm != 'shared_base'
        self.use_query = arm.startswith('shared_query_')
        self.use_conditional = arm.endswith('_conditional')
        if self.shared:
            # One registered encoder, including one stock retrieval neck. Heads
            # specialize the readout; IV/EX and all scales share encoder weights.
            shared = self.backbones['soma']
            self.backbones = nn.ModuleDict({'shared': shared})
        with torch.random.fork_rng(devices=[]):
            # New modules are constructed on CPU. Seed that generator only;
            # torch.manual_seed also changes CUDA dropout RNG and would alter
            # the otherwise matched independent-local control.
            torch.random.default_generator.manual_seed(20271017)
            self.conditioner = QueryConditioner() if self.use_query else None

    def trunk(self, level):
        return self.backbones['shared' if self.shared else level]

    def tokens(self, gray, level):
        t = self.trunk(level)
        x = t.patch_embed(self.pixels(gray)) + t.pos_embed + self.level_embed[LEVELS.index(level)]
        for block in t.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.training and torch.is_grad_enabled() else block(x)
        return t.norm(x.float())

    def crop_level(self, gray, level, chunk=32):
        out = []
        for start in range(0, len(gray), chunk):
            tokens = self.tokens(gray[start:start+chunk], level)
            value = self.trunk(level).feature(tokens.flatten(1))
            out.append(F.normalize(self.heads[level](value), dim=-1))
        return torch.cat(out)

    def global_tokens(self, images, bridge=True):
        gray = torch.as_tensor(images, device=self.device)
        thumb = F.interpolate(gray[:, None].float(), size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
        rgb = (thumb.div(255).expand(-1, 3, -1, -1) - self.pixel_mean) / self.pixel_std
        t = self.trunk('region')
        x = t.patch_embed(rgb)
        if bridge:
            delta = checkpoint(self.compressor, gray, use_reentrant=False) if self.training and torch.is_grad_enabled() else self.compressor(gray)
            x = x + delta
        x = x + t.pos_embed + self.level_embed[2]
        for block in t.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.training and torch.is_grad_enabled() else block(x)
        return t.norm(x.float())

    def forward(self, a, fields, qids, cids):
        q = self.describe(a, fields, 'q', qids)
        e, maps = self.describe(a, fields, 'ex', cids, return_maps=True)
        keys = maps[0].flatten(1).T
        valid = torch.as_tensor(fields['ex_valid'].reshape(-1), device=self.device)
        if self.use_query:
            xy = torch.as_tensor(fields['q_xy'][qids], device=self.device)
            region = self.conditioner(q, keys, xy, valid)
            q = torch.stack([q[:, 0], q[:, 1], region], dim=1)
        return q, e, q[:, 2] @ keys.T / TAU, valid


def hierarchical_logprob(local_logits, heatmap, spatial_valid, ex_xy, candidate_valid=None):
    """P(cell|query) = sum_b P(b|query,pair) P(cell|query,b).

Gaussian memberships define overlapping neighbourhoods; they are computed from
candidate input coordinates, never from target labels. Normalize local evidence
inside EACH neighbourhood. All candidates retain soft support.
    """
    y, x = torch.meshgrid(torch.arange(14, device=heatmap.device), torch.arange(14, device=heatmap.device), indexing='ij')
    centres = torch.stack([x, y], -1).reshape(-1, 2).float()[spatial_valid]
    candidate_bins = (ex_xy + .5) * 14 / FIELD_SIZE - .5
    membership = -((centres[:, None] - candidate_bins[None])**2).sum(-1) / (2 * MEMBERSHIP_SIGMA_BINS**2)
    conditional = local_logits[:, None, :] + membership[None]
    if candidate_valid is not None:
        assert candidate_valid.any(-1).all()
        # Avoid logsumexp over an all-minus-infinity column: its backward is
        # undefined even when the downstream loss ignores that candidate.
        conditional = conditional.masked_fill(~candidate_valid[:, None], -1e4)
    conditional = conditional.log_softmax(-1)
    coarse = heatmap[:, spatial_valid].log_softmax(-1)
    result = torch.logsumexp(coarse[:, :, None] + conditional, dim=1)
    return result if candidate_valid is None else result.masked_fill(~candidate_valid, -torch.inf)


def score_tensors(model, q, e, heatmap, spatial_valid, ex_xy, candidate_valid=None):
    terms = {name: q[:, i] @ e[:, i].T for i, name in enumerate(LEVELS)}
    terms['soma+neighbourhood'] = (terms['soma'] + terms['neighbourhood']) / 2
    if model.use_conditional:
        terms['combined'] = torch.cat([
            hierarchical_logprob(terms['soma+neighbourhood'][s:s+32]/TAU, heatmap[s:s+32],
                                 spatial_valid, ex_xy, None if candidate_valid is None else candidate_valid[s:s+32])
            for s in range(0, len(q), 32)])
    else:
        terms['combined'] = sum(terms[name] for name in LEVELS) / 3
    return terms


def training_losses(model, a, fields, qids, cids, candidate_valid, output):
    assert a['_subject'] in TRAIN, 'Only training animals may supply supervision'
    q, e, heatmap, spatial_valid = output
    valid = torch.as_tensor(candidate_valid, device=q.device)
    ex_xy = torch.as_tensor(fields['ex_xy'][cids], device=q.device)
    scores = score_tensors(model, q, e, heatmap, spatial_valid, ex_xy, valid)
    truth = torch.arange(len(q), device=q.device)
    logits = scores['combined'] if model.use_conditional else scores['combined']/TAU
    cell = F.cross_entropy(logits.masked_fill(~valid, -torch.inf), truth)
    target = location_targets(a['target_xy'][qids], fields['ex_geom'], spatial_valid, q.device)
    location = -(target[:, spatial_valid] * heatmap[:, spatial_valid].log_softmax(-1)).sum(-1).mean()
    target_xy = torch.as_tensor(a['target_xy'][qids], device=q.device)
    native_candidates = torch.as_tensor(a['ex_xy'][cids], device=q.device)
    nearby = (torch.cdist(target_xy, native_candidates) <= LOCAL_RADIUS_PX) & valid
    nearby[truth, truth] = True  # paired detections may be offset from exact landmarks
    ambiguous = nearby.sum(-1) > 1
    local = scores['soma+neighbourhood'].sum() * 0.
    if model.use_local and ambiguous.any():
        local = F.cross_entropy((scores['soma+neighbourhood']/TAU).masked_fill(~nearby, -torch.inf)[ambiguous], truth[ambiguous])
    total = cell + LOC_WEIGHT*location + (LOCAL_WEIGHT*local if model.use_local else 0)
    return total, dict(cell=cell, location=location, local=local,
                      local_ambiguous_fraction=ambiguous.float().mean())
