"""PMAC: native-image convolutional compression + stock UNICOM spatial memory.

Soma/context retain independent stock UNICOM image encoders. The global encoder
gets a 196-token residual from native-resolution convolutional features, added
to stock thumbnail patch embeddings. Each level shares weights between IV/EX.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .data import FIELD_SIZE, sample_map, map_points, TRAIN
from .model_unicom import SeparateUnicomThreeLevel, LEVELS, STOCK_CKPT, STOCK_SHA256, VENDOR

GRID=14
TAU=.07
LOC_WEIGHT=.25
LOC_SIGMA=.75  # in coarse-grid bins


class ConvBlock(nn.Module):
    """Small ConvNeXt-style depthwise block, channel LayerNorm and residual MLP."""
    def __init__(self, channels=64):
        super().__init__()
        self.depthwise=nn.Conv2d(channels,channels,7,padding=3,groups=channels)
        self.norm=nn.LayerNorm(channels)
        self.up=nn.Linear(channels,2*channels)
        self.down=nn.Linear(2*channels,channels)
        self.scale=nn.Parameter(torch.full((channels,),1e-3))

    def forward(self,x):
        y=self.depthwise(x).permute(0,2,3,1)
        y=self.down(F.gelu(self.up(self.norm(y))))*self.scale
        return x+y.permute(0,3,1,2)


class NativeCompressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv2d(1,64,4,stride=4)
        self.blocks=nn.Sequential(ConvBlock(),ConvBlock())
        self.norm=nn.LayerNorm(128)
        self.projection=nn.Linear(128,768)
        # At initialization, global processing is exactly the stock thumbnail path.
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self,gray):
        with torch.autocast('cuda',dtype=torch.float16,enabled=gray.device.type=='cuda'):
            x=self.blocks(self.stem(gray[:,None].float().div(255).sub(.5).mul(2)))
            x=torch.cat([F.adaptive_avg_pool2d(x,GRID),F.adaptive_max_pool2d(x,GRID)],dim=1)
            x=x.flatten(2).transpose(1,2)
            return self.projection(self.norm(x)).float()


class PMAC(SeparateUnicomThreeLevel):
    def __init__(self):
        super().__init__('unicom_thumb')
        self.arm='pmac_unicom'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2027)
            self.compressor=NativeCompressor()

    def global_tokens(self,images,bridge=True):
        gray=torch.as_tensor(images,device=self.device)
        assert gray.ndim==3 and gray.shape[-1]==gray.shape[-2]
        # Ordinary thumbnail is only the stable residual base. Native images also
        # reach the CNN before any resize, preserving the proposed compression path.
        thumb=F.interpolate(gray[:,None].float(),size=(224,224),mode='bilinear',align_corners=False,antialias=True)
        rgb=thumb.div(255).expand(-1,3,-1,-1)
        rgb=(rgb-self.pixel_mean)/self.pixel_std
        t=self.backbones['region']
        x=t.patch_embed(rgb)
        if bridge:
            delta=checkpoint(self.compressor,gray,use_reentrant=False) if self.training and torch.is_grad_enabled() else self.compressor(gray)
            x=x+delta
        x=x+t.pos_embed+self.level_embed[2]
        for blk in t.blocks:
            x=checkpoint(blk,x,use_reentrant=False) if self.training and torch.is_grad_enabled() else blk(x)
        return t.norm(x.float())

    def region_maps(self,images):
        tokens=self.global_tokens(images)
        # Normalize spatial keys before interpolation so candidate global scores
        # exactly equal bilinear samples of the coarse heatmap logits.
        keys=F.normalize(self.heads['region'](tokens),dim=-1)
        return keys.transpose(1,2).reshape(len(images),768,GRID,GRID)

    def field_level(self,images,ids,coordinates,chunk=2,return_maps=False,query=True):
        values=[];indices=[];stored={};keys=list(images)
        for start in range(0,len(keys),chunk):
            batch=keys[start:start+chunk]
            maps=self.region_maps(np.stack([images[k] for k in batch]))
            for i,k in enumerate(batch):
                sel=np.flatnonzero(ids==k)
                sampled=sample_map(maps[i:i+1],torch.as_tensor(coordinates[sel],device=self.device))
                values.append(F.normalize(sampled,dim=-1) if query else sampled)
                indices.append(torch.as_tensor(sel,device=self.device));stored[k]=maps[i]
        output=torch.cat(values)[torch.argsort(torch.cat(indices))]
        return (output,stored) if return_maps else output

    def describe(self,a,fields,prefix,indices,crop_chunk=32,return_maps=False):
        indices=np.asarray(indices,np.int64)
        soma=self.crop_level(a[prefix+'_local'][indices],'soma',crop_chunk)
        context=self.crop_level(a[prefix+'_global'][indices],'neighbourhood',crop_chunk)
        if prefix=='q':
            ids=a['q_z'][indices];images={int(z):fields['iv'][int(z)] for z in np.unique(ids)}
            xy=fields['q_xy'][indices]
        else:
            ids=np.zeros(len(indices),np.int64);images={0:fields['ex']};xy=fields['ex_xy'][indices]
        region,maps=self.field_level(images,ids,xy,return_maps=True,query=prefix=='q')
        desc=torch.stack([soma,context,region],dim=1)
        return (desc,maps) if return_maps else desc

    def forward(self,a,fields,qids,cids):
        q=self.describe(a,fields,'q',qids)
        e,maps=self.describe(a,fields,'ex',cids,return_maps=True)
        logits=q[:,2]@maps[0].flatten(1)/TAU
        valid=torch.as_tensor(fields['ex_valid'].reshape(-1),device=self.device)
        return q,e,logits,valid


def location_targets(native_xy,geom,valid,device):
    xy=torch.as_tensor(map_points(native_xy,geom),device=device)
    point=(xy+.5)*GRID/FIELD_SIZE-.5
    yy,xx=torch.meshgrid(torch.arange(GRID,device=device),torch.arange(GRID,device=device),indexing='ij')
    centres=torch.stack([xx,yy],dim=-1).reshape(-1,2).float()
    logits=-((point[:,None]-centres[None])**2).sum(-1)/(2*LOC_SIGMA**2)
    return logits.masked_fill(~valid[None],-torch.inf).softmax(-1)


def combined_cosine(q,e):
    return sum(q[:,i]@e[:,i].T for i in range(3))/3


def losses(q,e,heatmap,spatial_valid,candidate_valid,target_xy,geom):
    scores=combined_cosine(q,e)/TAU
    mask=torch.as_tensor(candidate_valid,device=scores.device)
    cell=F.cross_entropy(scores.masked_fill(~mask,-torch.inf),torch.arange(len(q),device=scores.device))
    target=location_targets(target_xy,geom,spatial_valid,scores.device)
    logprob=heatmap[:,spatial_valid].log_softmax(-1)
    location=-(target[:,spatial_valid]*logprob).sum(-1).mean()
    return cell+LOC_WEIGHT*location,cell,location
