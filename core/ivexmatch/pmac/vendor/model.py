"""Pure Base with controlled overview/CNN/tile input removal."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .factorial_model import *
from .factorial_model import ResolutionPMAC as MatchedPMAC


class InputRefiner(nn.Module):
    """Retain the original decoder, removing only absent input channels."""
    def __init__(self, original, native_cnn, use_tiles):
        super().__init__()
        self.native_cnn, self.use_tiles = native_cnn, use_tiles
        self.coarse = original.coarse
        keep = list(range(128))
        if native_cnn:
            keep += list(range(128, 192))
        if use_tiles:
            keep += list(range(192, 320))
        keep += [320, 321]
        with torch.random.fork_rng(devices=[]):
            first = nn.Conv2d(len(keep), 128, 1)
        with torch.no_grad():
            first.weight.copy_(original.fuse[0].weight[:, keep])
            first.bias.copy_(original.fuse[0].bias)
        self.fuse = nn.Sequential(first, *list(original.fuse.children())[1:])
        self.output = original.output
        self.kept_channels = keep

    def forward(self, coarse, native, tiles, size):
        assert (native is not None) == self.native_cnn
        assert (tiles is not None) == self.use_tiles
        base = F.interpolate(coarse, size=size, mode="bilinear", align_corners=False)
        parts = [F.interpolate(self.coarse(coarse), size=size, mode="bilinear", align_corners=False)]
        if native is not None:
            parts.append(F.interpolate(native, size=size, mode="bilinear", align_corners=False))
        if tiles is not None:
            parts.append(tiles)
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, size[0], device=coarse.device),
            torch.linspace(-1, 1, size[1], device=coarse.device), indexing="ij")
        parts.append(torch.stack([xs, ys])[None].expand(len(coarse), -1, -1, -1))
        return F.normalize(base + self.output(self.fuse(torch.cat(parts, 1))), dim=1)


class ResolutionPMAC(MatchedPMAC):
    def __init__(self, arm):
        requested = dict(CONFIGS[arm])
        assert requested["variant"] == "base"
        assert not any(requested[k] for k in ("cross_view", "coordinate", "uncertainty"))
        # Initialize the exact original all-stream Base first, avoiding changes
        # in common parameter initialization caused by different module shapes.
        CONFIGS[arm] = dict(requested, tile_px=224, decoder=True)
        try:
            super().__init__(arm)
        finally:
            CONFIGS[arm] = requested
        self.cfg = requested
        self.refiner = InputRefiner(self.refiner, requested["native_cnn"], requested["use_tiles"])
        if not requested["native_cnn"]:
            del self.compressor
        if not requested["use_tiles"]:
            self.tile_projection = None
        assert self.cross_view is None and self.coordinate_head is None

    def encode_field(self, image, geom, scale):
        gray = torch.as_tensor(image[None], device=self.device)
        if gray.ndim == 4:                    # PMAC GFP+DAPI: composite EX field (1, H, W, 3) -> (1, 3, 224, 224)
            assert not self.cfg["native_cnn"] and not self.cfg["use_tiles"]
            thumb = F.interpolate(gray.permute(0, 3, 1, 2).float(), size=(224, 224), mode="bilinear",
                                  align_corners=False, antialias=True)
        else:
            thumb = F.interpolate(gray[:, None].float(), size=(224, 224), mode="bilinear",
                                  align_corners=False, antialias=True)[:, 0]
        native = delta = None
        if self.cfg["native_cnn"]:
            def native_fn(v):
                return self.compressor.blocks(self.compressor.stem(v[:, None].float() / 127.5 - 1))
            native = (checkpoint(native_fn, gray, use_reentrant=False)
                      if self.training and torch.is_grad_enabled() else native_fn(gray))
            pooled = torch.cat([F.adaptive_avg_pool2d(native, 14),
                                F.adaptive_max_pool2d(native, 14)], 1).flatten(2).transpose(1, 2)
            delta = self.compressor.projection(self.compressor.norm(pooled))
        tokens = self.backbones["shared"](thumb, level=self.level_embed[2], delta=delta)
        n = math.isqrt(tokens.shape[1])
        keys = self.heads["region"](tokens).transpose(1, 2).reshape(1, 768, n, n)
        coarse = F.normalize(F.interpolate(keys, size=(14, 14), mode="bilinear", align_corners=False), dim=1)
        grid = math.ceil(image.shape[1] / 8)   # letterboxed square side (same as gray.shape[-1] for grayscale)
        tiles = self.tile_features(gray, geom, scale, grid) if self.cfg["use_tiles"] else None
        return coarse, native, tiles, grid

    def region_map(self, image, geom, scale):
        encoded = self.encode_field(image, geom, scale)
        field = self.decode_field(encoded, encoded[0])
        self._last_map = field
        return field, encoded[0]
