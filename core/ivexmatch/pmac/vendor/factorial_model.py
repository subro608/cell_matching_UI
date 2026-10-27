"""Joint, shared-encoder PMAC: paired spatial attention and continuous hypotheses.

These are controlled adaptations, not a reproduction of RoMa v2. Forward and
scoring never consume correspondence labels. Sparse train landmarks supervise
only hypotheses for annotated queries; they do not define a dense target warp.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from . import data
from .data import TRAIN, sample_map
from . import resolution_base as base
from .resolution_base import (CONFIGS, LEVELS, TAU, LOC_WEIGHT, LOCAL_WEIGHT,
    SIGMA_NATIVE_PX, spatial_centres, location_targets, sample_batch)
from .legacy_joint import positions

ARMS=tuple(CONFIGS)
COORD_WEIGHT=.5
UNCERTAINTY_WEIGHT=.1
OFFSET_BOUND=128.
COORD_UNIT=32.
SIGMA_MIN=8.
SIGMA_MAX=128.


class PairBlock(nn.Module):
    def __init__(self,d=256):
        super().__init__()
        self.self_norm=nn.LayerNorm(d)
        self.self_attn=nn.MultiheadAttention(d,8,batch_first=True,dropout=0.)
        self.cross_norm=nn.LayerNorm(d)
        self.cross_attn=nn.MultiheadAttention(d,8,batch_first=True,dropout=0.)
        self.ff=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))

    def attend(self,module,q,k,valid):
        return module(q[None],k[None],k[None],key_padding_mask=~valid[None],need_weights=False)[0][0]

    def forward(self,a,b,va,vb):
        na,nb=self.self_norm(a),self.self_norm(b)
        a=a+self.attend(self.self_attn,na,na,va)
        b=b+self.attend(self.self_attn,nb,nb,vb)
        na,nb=self.cross_norm(a),self.cross_norm(b)
        # Simultaneous exchange: EX cannot see an already-updated IV here.
        ca=self.attend(self.cross_attn,na,nb,vb)
        cb=self.attend(self.cross_attn,nb,na,va)
        a,b=a+ca,b+cb
        return a+self.ff(a),b+self.ff(b)


class PairedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,256))
        self.position=nn.Linear(16,256,bias=False)
        self.blocks=nn.ModuleList([PairBlock(),PairBlock()])
        self.output=nn.Linear(256,768)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,iv,ex,iv_valid,ex_valid):
        a,b=iv[0].flatten(1).T,ex[0].flatten(1).T
        y,x=torch.meshgrid(torch.arange(14,device=a.device),torch.arange(14,device=a.device),indexing='ij')
        xy=(torch.stack([x,y],-1).reshape(-1,2).float()+.5)*224/14-.5
        pos=self.position(positions(xy))
        qa,qb=self.input(a)+pos,self.input(b)+pos
        for block in self.blocks:qa,qb=block(qa,qb,iv_valid,ex_valid)
        def restore(original,value):
            return (original+self.output(value)).T.reshape(1,768,14,14)
        return restore(a,qa),restore(b,qb)


class CoordinateHead(nn.Module):
    def __init__(self,uncertainty=False,rank=64,offset=True):
        super().__init__();self.rank=rank;self.uncertainty=uncertainty;self.offset=offset
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20271018)
            self.query_norm=nn.LayerNorm(3*768);self.key_norm=nn.LayerNorm(768)
            if offset:
                self.q_offset=nn.Linear(3*768,2*rank)
                self.key_offset=nn.Linear(768,2*rank)
                nn.init.zeros_(self.q_offset.weight);nn.init.zeros_(self.q_offset.bias)
        if uncertainty:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(20271019)
                self.q_scale=nn.Linear(3*768,2*rank);self.key_scale=nn.Linear(768,2*rank)
                nn.init.zeros_(self.q_scale.weight);nn.init.zeros_(self.q_scale.bias)
        p=(SIGMA_NATIVE_PX-SIGMA_MIN)/(SIGMA_MAX-SIGMA_MIN)
        self.scale_bias=math.log(p/(1-p))

    def forward(self,q,keys,centres):
        query=self.query_norm(q.flatten(1));key=self.key_norm(keys)
        def interact(qproj,kproj):
            return torch.einsum('qhd,nhd->qnh',qproj(query).reshape(-1,2,self.rank),kproj(key).reshape(-1,2,self.rank))/math.sqrt(self.rank)
        mean=centres[None].expand(len(q),-1,-1)
        if self.offset:
            mean=mean+OFFSET_BOUND*interact(self.q_offset,self.key_offset).tanh()
        if self.uncertainty:
            sigma=SIGMA_MIN+(SIGMA_MAX-SIGMA_MIN)*torch.sigmoid(interact(self.q_scale,self.key_scale)+self.scale_bias)
        else:sigma=torch.full_like(mean,SIGMA_NATIVE_PX)
        return mean,sigma


class ResolutionPMAC(base.ResolutionPMAC):
    def __init__(self,arm):
        super().__init__(arm)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20271020)
            self.cross_view=PairedAttention() if self.cfg['cross_view'] else None
        self.coordinate_head=(CoordinateHead(self.cfg['uncertainty'],offset=self.cfg['coordinate'])
                              if self.cfg['coordinate'] or self.cfg['uncertainty'] else None)
        self.pair_groups=[];self.means=None;self.sigmas=None;self._last_map=None

    def region_map(self,image,geom,scale):
        result=super().region_map(image,geom,scale)
        self._last_map=result[0]
        return result

    def encode_field(self,image,geom,scale):
        # Same encoder call order, preprocessing and tile construction as base.
        gray=torch.as_tensor(image[None],device=self.device)
        thumb=F.interpolate(gray[:,None].float(),size=(224,224),mode='bilinear',align_corners=False,antialias=True)[:,0]
        def native_fn(v):return self.compressor.blocks(self.compressor.stem(v[:,None].float()/127.5-1))
        native=checkpoint(native_fn,gray,use_reentrant=False) if self.training and torch.is_grad_enabled() else native_fn(gray)
        pooled=torch.cat([F.adaptive_avg_pool2d(native,14),F.adaptive_max_pool2d(native,14)],1).flatten(2).transpose(1,2)
        delta=self.compressor.projection(self.compressor.norm(pooled))
        tokens=self.backbones['shared'](thumb,level=self.level_embed[2],delta=delta)
        n=math.isqrt(tokens.shape[1]);keys=self.heads['region'](tokens).transpose(1,2).reshape(1,768,n,n)
        coarse=F.normalize(F.interpolate(keys,size=(14,14),mode='bilinear',align_corners=False),dim=1)
        grid=math.ceil(gray.shape[-1]/8)
        tiles=self.tile_features(gray,geom,scale,grid)
        return coarse,native,tiles,grid

    def decode_field(self,encoded,coarse):
        _,native,tiles,grid=encoded
        def decode(c,n,t):return self.refiner(c,n,t,(grid,grid))
        return checkpoint(decode,coarse,native,tiles,use_reentrant=False) if self.training and torch.is_grad_enabled() else decode(coarse,native,tiles)

    def paired_forward(self,a,fields,qids,cids):
        qids=np.asarray(qids);cids=np.asarray(cids)
        qs=self.crop_level(a['q_local'][qids],'soma')
        qn=self.crop_level(a['q_global'][qids],'neighbourhood')
        zs=a['q_z'][qids];iv_encoded={}
        for z in np.unique(zs):
            iv_encoded[int(z)]=self.encode_field(fields['iv'][int(z)],fields['iv_geom'],fields.get('iv_scale',self.iv_scale))
        es=self.crop_level(a['ex_local'][cids],'soma')
        en=self.crop_level(a['ex_global'][cids],'neighbourhood')
        ex_encoded=self.encode_field(fields['ex'],fields['ex_geom'],np.ones(2))
        self.geom=fields['ex_geom'];self.grid=ex_encoded[3]
        valid=torch.as_tensor(data.content_mask(self.geom,self.grid).reshape(-1),device=self.device)
        ev=torch.as_tensor(data.content_mask(self.geom,14).reshape(-1),device=self.device)
        iv=torch.as_tensor(data.content_mask(fields['iv_geom'],14).reshape(-1),device=self.device)
        query_parts=[];heat_parts=[];order=[];first_ex=None
        for z,encoded in iv_encoded.items():
            ids=torch.as_tensor(np.flatnonzero(zs==z),device=self.device)
            ci,ce=self.cross_view(encoded[0],ex_encoded[0],iv,ev)
            imap=self.decode_field(encoded,ci);emap=self.decode_field(ex_encoded,ce)
            xy=torch.as_tensor(fields['q_xy'][qids][ids.cpu().numpy()],device=self.device)
            region=F.normalize(sample_map(imap,xy),dim=-1)
            q=torch.stack([qs[ids],qn[ids],region],1)
            updated=self.conditioner(q,ce[0].flatten(1).T,xy,ev)
            q=torch.stack([q[:,0],q[:,1],updated],1)
            query_parts.append(q);heat_parts.append(q[:,2]@emap[0].flatten(1)/TAU);order.append(ids)
            self.pair_groups.append((ids,emap))
            if first_ex is None:first_ex=emap
        ordering=torch.argsort(torch.cat(order))
        q=torch.cat(query_parts)[ordering];heat=torch.cat(heat_parts)[ordering]
        # EX region embeddings depend on IV plane in A. This first-plane view
        # preserves the descriptor API; score_tensors uses each query's map.
        er=sample_map(first_ex,torch.as_tensor(fields['ex_xy'][cids],device=self.device))
        return q,torch.stack([es,en,er],1),heat,valid

    def forward(self,a,fields,qids,cids):
        self.pair_groups=[];self.means=None;self.sigmas=None;self._last_map=None
        if self.cross_view is None:
            output=super().forward(a,fields,qids,cids)
            self.pair_groups=[(torch.arange(len(qids),device=self.device),self._last_map)]
        else:output=self.paired_forward(a,fields,qids,cids)
        if self.coordinate_head is not None:
            q=output[0];centres=spatial_centres(self.grid,self.geom,self.device)
            means=[];sigmas=[];order=[]
            for ids,emap in self.pair_groups:
                args=(q[ids],emap[0].flatten(1).T,centres)
                mu,sd=checkpoint(self.coordinate_head,*args,use_reentrant=False) if self.training and torch.is_grad_enabled() else self.coordinate_head(*args)
                means.append(mu);sigmas.append(sd);order.append(ids)
            ordering=torch.argsort(torch.cat(order))
            self.means=torch.cat(means)[ordering];self.sigmas=torch.cat(sigmas)[ordering]
        return output

    def dense_candidate_region_scores(self,q,ex_xy):
        values=[];order=[]
        for ids,emap in self.pair_groups:
            keys=sample_map(emap,ex_xy)
            values.append(q[ids,2]@keys.T);order.append(ids)
        return torch.cat(values)[torch.argsort(torch.cat(order))]


def marginal_coordinates(local,heat,mean,sigma,xy,mask):
    membership=-.5*(((xy[None,None]-mean[:,:,None])/sigma[:,:,None])**2).sum(-1)
    conditional=(local[:,None]+membership).masked_fill(~mask[:,None],-1e4).log_softmax(-1)
    return torch.logsumexp(heat.log_softmax(-1)[:,:,None]+conditional,1).masked_fill(~mask,-torch.inf)


def score_tensors(model,q,e,heatmap,spatial_valid,ex_xy,candidate_valid=None):
    if model.coordinate_head is None:
        terms=base.score_tensors(model,q,e,heatmap,spatial_valid,ex_xy,candidate_valid)
    else:
        terms={name:q[:,i]@e[:,i].T for i,name in enumerate(LEVELS)}
        terms['soma+neighbourhood']=(terms['soma']+terms['neighbourhood'])/2
        xy=(ex_xy-ex_xy.new_tensor(model.geom['offset_xy'])+.5)/(ex_xy.new_tensor(model.geom['resized_wh'])/ex_xy.new_tensor(model.geom['native_wh']))-.5
        results=[]
        for s in range(0,len(q),4):
            local=terms['soma+neighbourhood'][s:s+4]/TAU
            cv=torch.ones_like(local,dtype=torch.bool) if candidate_valid is None else candidate_valid[s:s+4]
            args=(local,heatmap[s:s+4,spatial_valid],model.means[s:s+4,spatial_valid],model.sigmas[s:s+4,spatial_valid],xy,cv)
            results.append(checkpoint(marginal_coordinates,*args,use_reentrant=False) if model.training and torch.is_grad_enabled() else marginal_coordinates(*args))
        terms['combined']=torch.cat(results)
    if model.cross_view is not None:
        terms['region']=sample_map(heatmap.reshape(len(q),model.grid,model.grid)[None],ex_xy).T*TAU
    return terms


def coordinate_losses(means,sigmas,centres,target_xy,target_weights,uncertainty):
    # Annotation-centred support is loss-only, never an inference gate.
    distance=torch.linalg.vector_norm(centres[None]-target_xy[:,None],dim=-1)
    weights=target_weights*(distance<=OFFSET_BOUND)
    assert (weights.sum(-1)>0).all(), 'annotation has no supported spatial anchors'
    weights=weights/weights.sum(-1,keepdim=True)
    residual=means-target_xy[:,None]
    robust=F.smooth_l1_loss(residual/COORD_UNIT,torch.zeros_like(residual),reduction='none').sum(-1)
    coord=(robust*weights).sum(-1).mean()
    nll=coord*0
    if uncertainty:
        # Proper Gaussian scale fit: mean error is detached for this auxiliary
        # term; joint cell NLL continues to train both offsets and scales.
        gaussian=(.5*(residual.detach()/sigmas)**2+torch.log(sigmas/SIGMA_NATIVE_PX)).sum(-1)
        nll=(gaussian*weights).sum(-1).mean()
    return coord,nll


def training_losses(model,a,fields,qids,cids,candidate_valid,output):
    assert data.is_training_record(a)
    q,e,heat,spatial_valid=output;valid=torch.as_tensor(candidate_valid,device=q.device)
    terms=score_tensors(model,q,e,heat,spatial_valid,torch.as_tensor(fields['ex_xy'][cids],device=q.device),valid)
    truth=torch.arange(len(q),device=q.device)
    cell=F.cross_entropy(terms['combined'],truth)
    target_xy=torch.as_tensor(a['target_xy'][qids],device=q.device)
    target=location_targets(a['target_xy'][qids],fields['ex_geom'],spatial_valid,q.device)
    location=-(target[:,spatial_valid]*heat[:,spatial_valid].log_softmax(-1)).sum(-1).mean()
    distance=torch.cdist(target_xy,torch.as_tensor(a['ex_xy'][cids],device=q.device))
    nearby=(distance<=50)&valid;nearby[truth,truth]=True;ambiguous=nearby.sum(-1)>1
    local=terms['soma+neighbourhood'].sum()*0
    if ambiguous.any():local=F.cross_entropy((terms['soma+neighbourhood']/TAU).masked_fill(~nearby,-torch.inf)[ambiguous],truth[ambiguous])
    coord=local*0;uncertainty=local*0
    if model.coordinate_head is not None:
        centres=spatial_centres(model.grid,model.geom,q.device)
        coord,uncertainty=coordinate_losses(model.means,model.sigmas,centres,target_xy,target,model.cfg['uncertainty'])
        if not model.cfg['coordinate']:
            coord=local*0
    total=cell+LOC_WEIGHT*location+LOCAL_WEIGHT*local+COORD_WEIGHT*coord+UNCERTAINTY_WEIGHT*uncertainty
    return total,dict(cell=cell,location=location,local=local,coordinate=coord,uncertainty=uncertainty,local_ambiguous_fraction=ambiguous.float().mean())
