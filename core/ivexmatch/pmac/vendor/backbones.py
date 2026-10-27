"""One shared pretrained encoder; all adapters are trained in the same optimizer."""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .model_unicom import VENDOR, STOCK_CKPT

HERE = Path(__file__).resolve().parent
DINO_REPO = Path('/scratch/sd5963/.cache/torch/hub/facebookresearch_dinov2_main')
CELL_STOCK = Path('/scratch/sd5963/.cache/torch/hub/checkpoints/cell_dino_vitl14_pretrain_hpa_fov_highres-f57e7934.pth')
WEIGHTS = {
 'dinov2': Path('/scratch/sd5963/.cache/huggingface/hub/models--facebook--dinov2-with-registers-base/snapshots/a1d738ccfa7ae170945f210395d99dde8adb1805'),
 'dinov3': Path('/scratch/sd5963/.cache/huggingface/hub/models--facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/5931719e67bbdb9737e363e781fb0c67687896bc'),
 'unicom': STOCK_CKPT, 'celldino': CELL_STOCK,
 'celldino_continued': HERE/'assets/celldino_continued_source_val.pth',
}

class Encoder(nn.Module):
 def __init__(self, name):
  super().__init__(); self.name=name
  if name=='unicom':
   from .unicom_vit import load_model_and_transform
   self.net,_=load_model_and_transform('ViT-B/16')
   if STOCK_CKPT.exists():  # ivex: optional; a PMAC checkpoint overwrites every tensor (load_state_dict strict=True)
    weights=torch.load(STOCK_CKPT,map_location='cpu',weights_only=True)
    self.net.load_state_dict(weights,strict=True)
    assert all(torch.equal(v,weights[k].to(v.dtype)) for k,v in self.net.state_dict().items())
   self.width,self.patch,self.pool_width=768,16,768
   mean,std=(.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711)
  elif name.startswith('celldino'):
   self.net=torch.hub.load(str(DINO_REPO),'cell_dino_hpa_vitl14',source='local',pretrained=True,pretrained_path=str(CELL_STOCK),in_channels=4)
   if name=='celldino_continued':
    state=torch.load(WEIGHTS[name],map_location='cpu',weights_only=False)
    self.net.load_state_dict(state["model"],strict=True)
   if self.net.chunked_blocks:
    blocks=[b for c in self.net.blocks for b in c if not isinstance(b,nn.Identity)]
    assert len(blocks)==self.net.n_blocks
    self.net.blocks=nn.ModuleList(blocks); self.net.chunked_blocks=False
   self.width,self.patch,self.pool_width=1024,14,2048
   assert tuple(self.net.pos_embed.shape)==(1,1370,1024)
   mean,std=(0.,),(1.,)
  else:
   from transformers import Dinov2WithRegistersModel,DINOv3ViTModel
   cls=Dinov2WithRegistersModel if name=='dinov2' else DINOv3ViTModel
   kw={} if name=='dinov2' else dict(pos_embed_rescale=None,pos_embed_shift=None,pos_embed_jitter=None)
   self.net=cls.from_pretrained(WEIGHTS[name],local_files_only=True,attn_implementation='sdpa',**kw)
   self.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
   self.width=self.net.config.hidden_size; self.patch=self.net.config.patch_size
   self.pool_width=2*self.width; self.prefix=1+self.net.config.num_register_tokens
   mean,std=(.485,.456,.406),(.229,.224,.225)
  self.register_buffer('mean',torch.tensor(mean)[None,:,None,None],persistent=False)
  self.register_buffer('std',torch.tensor(std)[None,:,None,None],persistent=False)
  self.requires_grad_(True)

 def train(self, mode=True):
  super().train(mode)
  for m in self.net.modules():
   if isinstance(m,nn.BatchNorm1d): m.eval()
  return self

 def pixels(self, gray):
  if gray.ndim==4:                         # PMAC GFP+DAPI: EX composite (N, 3, H, W) = (DAPI, GFP, DAPI), already 3 channels
   assert gray.shape[1]==3 and not self.name.startswith('celldino')
   return (gray.float()/255-self.mean)/self.std
  x=gray[:,None].float()/255
  if self.name.startswith('celldino'):
   return ((x-x.mean((-2,-1),keepdim=True))/(x.std((-2,-1),unbiased=False,keepdim=True)+1e-7)).expand(-1,4,-1,-1).contiguous()
  return (x.expand(-1,3,-1,-1)-self.mean)/self.std

 def forward(self, gray, level=None, delta=None, pooled=False):
  x=self.pixels(gray)
  if self.name=='unicom':
   x=self.net.patch_embed(x)
   if delta is not None:x=x+delta
   x=x+self.net.pos_embed
   if level is not None:x=x+level
   for b in self.net.blocks:
    x=checkpoint(b,x,use_reentrant=False) if self.training and torch.is_grad_enabled() else b(x)
   patch=self.net.norm(x.float())
   return (patch,self.net.feature(patch.flatten(1))) if pooled else patch
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=x.device.type=='cuda' and self.training):
   if self.name.startswith('celldino'):
    x=self.net.prepare_tokens_with_masks(x)
    for b in self.net.blocks:
     x=checkpoint(b,x,use_reentrant=False) if self.training and torch.is_grad_enabled() else b(x)
    x=self.net.norm(x); patch=x[:,1:]
   else:
    x=self.net(pixel_values=x,return_dict=True).last_hidden_state
    patch=x[:,self.prefix:]
   value=torch.cat([x[:,0],patch.mean(1)],-1)
  patch=patch.float()
  if delta is not None:
   n=int(patch.shape[1]**.5)
   d=F.interpolate(delta.transpose(1,2).reshape(len(x),self.width,14,14),size=(n,n),mode='bilinear',align_corners=False).flatten(2).transpose(1,2)
   patch=patch+d
  if level is not None:patch=patch+level
  return (patch,value.float()) if pooled else patch
