"""Controlled resolution/backbone PMAC family, one shared encoder per model."""
import json, math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from . import data
from .data import HERE, SOURCE, TRAIN, FIELD_SIZE, sample_map
from .backbones import Encoder
from .base_model import NativeCompressor, ConvBlock
from .model_unicom import ResidualHead, LEVELS, region_head
from .legacy_joint import QueryConditioner

CONFIGS=json.loads((SOURCE/'arms.json').read_text()); ARMS=tuple(CONFIGS)
TAU=.07; LOC_WEIGHT=.25; LOCAL_WEIGHT=.5; SIGMA_NATIVE_PX=58.2857142857

def spatial_centres(grid, geom, device, native=True):
 y,x=torch.meshgrid(torch.arange(grid,device=device),torch.arange(grid,device=device),indexing='ij')
 xy=(torch.stack([x,y],-1).reshape(-1,2).float()+.5)*FIELD_SIZE/grid-.5
 if native:
  xy=(xy-xy.new_tensor(geom['offset_xy'])+.5)/(xy.new_tensor(geom['resized_wh'])/xy.new_tensor(geom['native_wh']))-.5
 return xy

def location_targets(native_xy,geom,valid,device):
 grid=math.isqrt(len(valid)); centres=spatial_centres(grid,geom,device)
 xy=torch.as_tensor(native_xy,device=device)
 logits=-((xy[:,None]-centres[None])**2).sum(-1)/(2*SIGMA_NATIVE_PX**2)
 return logits.masked_fill(~valid[None],-torch.inf).softmax(-1)

def tile_layout(side, geom, window, scale):
 """Windows use a train-derived IV magnification, never image-size ratios."""
 wh=np.asarray([window,window],float)/np.asarray(scale)
 xs=np.arange(0,side,max(1.,wh[0]/2)); ys=np.arange(0,side,max(1.,wh[1]/2))
 left,top=geom['native_offset_xy']; w,h=geom['native_wh']
 return [(float(x),float(y),float(wh[0]),float(wh[1])) for y in ys for x in xs
         if x<left+w and x+wh[0]>left and y<top+h and y+wh[1]>top]

class Refiner(nn.Module):
 def __init__(self, tiled):
  super().__init__()
  self.coarse=nn.Conv2d(768,128,1)
  self.fuse=nn.Sequential(nn.Conv2d(128+64+(128 if tiled else 0)+2,128,1),nn.GELU(),ConvBlock(128),ConvBlock(128))
  self.output=nn.Conv2d(128,768,1)
  nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
 def forward(self, coarse, native, tiles, size):
  base=F.interpolate(coarse,size=size,mode='bilinear',align_corners=False)
  parts=[F.interpolate(self.coarse(coarse),size=size,mode='bilinear',align_corners=False),F.interpolate(native,size=size,mode='bilinear',align_corners=False)]
  if tiles is not None:parts.append(tiles)
  ys,xs=torch.meshgrid(torch.linspace(-1,1,size[0],device=coarse.device),torch.linspace(-1,1,size[1],device=coarse.device),indexing='ij')
  parts.append(torch.stack([xs,ys])[None].expand(len(coarse),-1,-1,-1))
  return F.normalize(base+self.output(self.fuse(torch.cat(parts,1))),dim=1)

class ResolutionPMAC(nn.Module):
 def __init__(self, arm):
  super().__init__(); self.arm=arm; self.cfg=CONFIGS[arm]
  self.backbones=nn.ModuleDict({'shared':Encoder(self.cfg['backbone'])})
  enc=self.backbones['shared']; width=enc.width
  self.shared=self.use_local=self.use_query=self.use_conditional=True
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(1729)
   self.heads=nn.ModuleDict({name:(ResidualHead() if enc.name=='unicom' else nn.Sequential(nn.LayerNorm(enc.pool_width),nn.Linear(enc.pool_width,768),nn.GELU(),nn.Linear(768,768))) for name in LEVELS[:2]})
   self.heads['region']=region_head() if width==768 else nn.Sequential(nn.LayerNorm(width),nn.Linear(width,768),nn.GELU(),nn.Linear(768,768))
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(2027)
   self.compressor=NativeCompressor()
   if width!=768:
    self.compressor.projection=nn.Linear(128,width)
    nn.init.zeros_(self.compressor.projection.weight);nn.init.zeros_(self.compressor.projection.bias)
  self.level_embed=nn.Parameter(torch.zeros(3,width))
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(20271017)
   self.conditioner=QueryConditioner()
   self.refiner=Refiner(bool(self.cfg['tile_px'])) if self.cfg['decoder'] else None
   self.tile_projection=nn.Linear(width,128) if self.cfg['tile_px'] else None
  self.geom=None; self.grid=14
  source=HERE.parent/'marco_onestage_B_unfrozen_2026-10-08/manifest.json'
  # Magnification is estimated only from this experiment's training subjects.
  self.iv_scale=data.crop_scale()

 @property
 def device(self):return self.level_embed.device
 def trunk(self,level):return self.backbones['shared'].net
 def train(self,mode=True):
  super().train(mode);self.backbones['shared'].train(mode)
  # Evaluation must be stable to query batching and candidate permutation.
  torch.backends.cuda.matmul.allow_tf32=mode
  torch.backends.cudnn.allow_tf32=mode
  return self
 def crop_level(self,gray,level,chunk=16):
  values=[];enc=self.backbones['shared']
  for s in range(0,len(gray),chunk):
   g=torch.as_tensor(gray[s:s+chunk],device=self.device)
   count=len(g)
   if not self.training and count<chunk:
    # Fixed GEMM batch shape also stabilizes stock UNICOM's FP16 inference.
    g=torch.cat([g,g[:1].expand(chunk-count,*g.shape[1:])])
   _,value=enc(g,level=self.level_embed[LEVELS.index(level)],pooled=True)
   values.append(F.normalize(self.heads[level](value[:count]),dim=-1))
  return torch.cat(values)

 def tile_features(self,gray,geom,scale,grid):
  side=gray.shape[-1]; layout=tile_layout(side,geom,self.cfg['tile_px'],scale)
  # Differentiable overlap-add at global grid centres; no tile CLS pooling.
  total=torch.zeros(128,grid*grid,device=self.device); weights=torch.zeros(1,grid*grid,device=self.device)
  centres=(torch.arange(grid,device=self.device)+.5)*side/grid-.5
  for start in range(0,len(layout),8):
   entries=layout[start:start+8]; crops=[]
   for x,y,w,h in entries:
    yy,xx=torch.meshgrid((torch.arange(224,device=self.device)+.5)*h/224+y-.5,(torch.arange(224,device=self.device)+.5)*w/224+x-.5,indexing='ij')
    g=torch.stack([(2*xx+1)/side-1,(2*yy+1)/side-1],-1)[None]
    crops.append(F.grid_sample(gray[:,None].float(),g,align_corners=False,padding_mode='border')[:,0])
   crops=torch.cat(crops)
   def encode(v):
    tokens=self.backbones['shared'](v,level=self.level_embed[2])
    n=math.isqrt(tokens.shape[1]);return self.tile_projection(tokens).transpose(1,2).reshape(len(v),128,n,n)
   maps=checkpoint(encode,crops,use_reentrant=False) if self.training and torch.is_grad_enabled() else encode(crops)
   for i,(x,y,w,h) in enumerate(entries):
    ix=torch.where((centres>=x-.5)&(centres<x+w-.5))[0]; iy=torch.where((centres>=y-.5)&(centres<y+h-.5))[0]
    if not len(ix) or not len(iy):continue
    uy=(centres[iy]+.5-y)/h;ux=(centres[ix]+.5-x)/w
    yy,xx=torch.meshgrid(uy,ux,indexing='ij');g=torch.stack([2*xx-1,2*yy-1],-1)[None]
    values=F.grid_sample(maps[i:i+1],g,align_corners=False,padding_mode='border')[0].flatten(1)
    weight=((1-(2*xx-1).abs()).clamp_min(.05)*(1-(2*yy-1).abs()).clamp_min(.05)).flatten()[None]
    idx=(iy[:,None]*grid+ix[None]).flatten()
    total=total.index_add(1,idx,values*weight);weights=weights.index_add(1,idx,weight)
  self.last_tile_count=len(layout)
  return (total/weights.clamp_min(1e-6)).reshape(1,128,grid,grid)

 def region_map(self,image,geom,scale):
  gray=torch.as_tensor(image[None],device=self.device)
  enc=self.backbones['shared'];side=gray.shape[-1]
  thumb=F.interpolate(gray[:,None].float(),size=(224,224),mode='bilinear',align_corners=False,antialias=True)[:,0]
  # One native CNN supplies both compressed tokens and retained spatial skips.
  def native_fn(v):return self.compressor.blocks(self.compressor.stem(v[:,None].float()/127.5-1))
  native=checkpoint(native_fn,gray,use_reentrant=False) if self.training and torch.is_grad_enabled() else native_fn(gray)
  pooled=torch.cat([F.adaptive_avg_pool2d(native,14),F.adaptive_max_pool2d(native,14)],1).flatten(2).transpose(1,2)
  delta=self.compressor.projection(self.compressor.norm(pooled))
  tokens=enc(thumb,level=self.level_embed[2],delta=delta)
  n=math.isqrt(tokens.shape[1]);keys=self.heads['region'](tokens).transpose(1,2).reshape(1,768,n,n)
  coarse=F.normalize(F.interpolate(keys,size=(14,14),mode='bilinear',align_corners=False),dim=1)
  if not self.cfg['decoder']:return coarse,coarse
  grid=math.ceil(side/8)
  tiles=self.tile_features(gray,geom,scale,grid) if self.cfg['tile_px'] else None
  return self.refiner(coarse,native,tiles,(grid,grid)),coarse

 def describe(self,a,fields,prefix,indices,return_maps=False):
  ids=np.asarray(indices,np.int64)
  soma=self.crop_level(a[prefix+'_local'][ids],'soma');context=self.crop_level(a[prefix+'_global'][ids],'neighbourhood')
  if prefix=='q':
   zs=a['q_z'][ids];images={int(z):fields['iv'][int(z)] for z in np.unique(zs)};xy=fields['q_xy'][ids];geom=fields['iv_geom'];scale=self.iv_scale
  else:
   zs=np.zeros(len(ids),int);images={0:fields['ex']};xy=fields['ex_xy'][ids];geom=fields['ex_geom'];scale=np.ones(2)
  values=[];order=[];maps={}
  for z,image in images.items():
   fmap,coarse=self.region_map(image,geom,scale);selected=np.flatnonzero(zs==z)
   value=sample_map(fmap,torch.as_tensor(xy[selected],device=self.device))
   values.append(F.normalize(value,dim=-1) if prefix=='q' else value);order.append(torch.as_tensor(selected,device=self.device));maps[z]=fmap[0]
   if prefix=='ex':self.overview=coarse[0]
  region=torch.cat(values)[torch.argsort(torch.cat(order))]
  desc=torch.stack([soma,context,region],1)
  return (desc,maps) if return_maps else desc

 def forward(self,a,fields,qids,cids):
  q=self.describe(a,fields,'q',qids);e,maps=self.describe(a,fields,'ex',cids,True)
  self.geom=fields['ex_geom'];self.grid=maps[0].shape[-1]
  valid=torch.as_tensor(data.content_mask(self.geom,self.grid).reshape(-1),device=self.device)
  coarse_valid=torch.as_tensor(data.content_mask(self.geom,14).reshape(-1),device=self.device)
  region=self.conditioner(q,self.overview.flatten(1).T,torch.as_tensor(fields['q_xy'][qids],device=self.device),coarse_valid)
  q=torch.stack([q[:,0],q[:,1],region],1)
  return q,e,q[:,2]@maps[0].flatten(1)/TAU,valid

def score_tensors(model,q,e,heatmap,spatial_valid,ex_xy,candidate_valid=None):
 terms={name:q[:,i]@e[:,i].T for i,name in enumerate(LEVELS)}
 terms['soma+neighbourhood']=(terms['soma']+terms['neighbourhood'])/2
 centres=spatial_centres(model.grid,model.geom,q.device)[spatial_valid]
 xy=(ex_xy-ex_xy.new_tensor(model.geom['offset_xy'])+.5)/(ex_xy.new_tensor(model.geom['resized_wh'])/ex_xy.new_tensor(model.geom['native_wh']))-.5
 membership=-((centres[:,None]-xy[None])**2).sum(-1)/(2*SIGMA_NATIVE_PX**2)
 results=[]
 for s in range(0,len(q),4):
  local=terms['soma+neighbourhood'][s:s+4]/TAU
  def marginal(local,heat,member,mask):
   conditional=local[:,None,:]+member[None]
   conditional=conditional.masked_fill(~mask[:,None],-1e4).log_softmax(-1)
   return torch.logsumexp(heat.log_softmax(-1)[:,:,None]+conditional,1).masked_fill(~mask,-torch.inf)
  cv=torch.ones_like(local,dtype=torch.bool) if candidate_valid is None else candidate_valid[s:s+4]
  args=(local,heatmap[s:s+4,spatial_valid],membership,cv)
  results.append(checkpoint(marginal,*args,use_reentrant=False) if model.training and torch.is_grad_enabled() else marginal(*args))
 terms['combined']=torch.cat(results)
 return terms

def training_losses(model,a,fields,qids,cids,candidate_valid,output):
 assert data.is_training_record(a)
 q,e,heat,spatial_valid=output;valid=torch.as_tensor(candidate_valid,device=q.device)
 terms=score_tensors(model,q,e,heat,spatial_valid,torch.as_tensor(fields['ex_xy'][cids],device=q.device),valid)
 truth=torch.arange(len(q),device=q.device)
 cell=F.cross_entropy(terms['combined'],truth)
 target=location_targets(a['target_xy'][qids],fields['ex_geom'],spatial_valid,q.device)
 location=-(target[:,spatial_valid]*heat[:,spatial_valid].log_softmax(-1)).sum(-1).mean()
 distance=torch.cdist(torch.as_tensor(a['target_xy'][qids],device=q.device),torch.as_tensor(a['ex_xy'][cids],device=q.device))
 nearby=(distance<=50)&valid;nearby[truth,truth]=True;ambiguous=nearby.sum(-1)>1
 local=terms['soma+neighbourhood'].sum()*0
 if ambiguous.any():local=F.cross_entropy((terms['soma+neighbourhood']/TAU).masked_fill(~nearby,-torch.inf)[ambiguous],truth[ambiguous])
 return cell+LOC_WEIGHT*location+LOCAL_WEIGHT*local,dict(cell=cell,location=location,local=local,local_ambiguous_fraction=ambiguous.float().mean())

def sample_batch(a,qids,rng,nearby=False):
 if not nearby:return data.batch_indices(a,qids,rng)
 assert data.is_training_record(a)
 pidx=a['positive'][qids].argmax(1);keep=np.unique(pidx,return_index=True)[1];qids,pidx=qids[keep],pidx[keep]
 eligible=np.flatnonzero(~a['background'] & a['valid'][qids].all(0) & ~a['positive'][qids].any(0))
 selected=[]
 if len(eligible):
  dist=np.linalg.norm(a['target_xy'][qids,None]-a['ex_xy'][eligible][None],axis=-1)
  ranked=np.argsort(dist,axis=1)
  for rank in range(min(len(eligible),32)):
   for i in rng.permutation(len(qids)):
    j=ranked[i,rank];candidate=int(eligible[j])
    if dist[i,j]<=50 and candidate not in selected:selected.append(candidate)
    if len(selected)==32:break
   if len(selected)==32:break
  remaining=np.setdiff1d(eligible,selected)
  selected.extend(rng.choice(remaining,min(32-len(selected),len(remaining)),replace=False).tolist())
 cids=np.concatenate([pidx,np.asarray(selected,int)]);valid=a['valid'][qids][:,cids]
 return qids,cids,valid
