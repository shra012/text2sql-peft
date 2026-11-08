"""Simulate removing the (``` vs Let/Here) component from selected layer-27 down_proj columns, at P-1."""
import torch
from t2s.wsp.kernel import load
D=torch.load('wsp/passes/pass_03/read.pt')
E=load('model.embed_tokens.weight'); g=load('model.norm.weight')
TICK,LET,HERE=73594,10061,8420
F=[k for k,i in enumerate(D['ids']) if i.startswith('F_')]
W=load('model.layers.27.mlp.down_proj.weight')
d=g*(E[TICK]-0.5*(E[LET]+E[HERE])); dh=d/d.norm()
A0=torch.stack([D['mlp_act:27'][k][0].float() for k in F]); hn=torch.stack([D['resid:27'][k][0].float() for k in F])
for cols in ([3317],[3317,7449]):
    for rms in (4.5,6.0):
        dW=torch.zeros_like(W)
        for j in cols: dW[:,j]=-(W[:,j]@dh)*dh
        delta=(A0@dW.T)          # pre-norm residual change at P-1
        lo=torch.log_softmax((hn+g*delta/rms)@E.T,-1); lo0=torch.log_softmax(hn@E.T,-1)
        p=lo[:,TICK].exp(); p0=lo0[:,TICK].exp()
        am=(lo.argmax(-1)==TICK).float().mean(); am0=(lo0.argmax(-1)==TICK).float().mean()
        top=torch.bincount(lo.argmax(-1),minlength=E.shape[0]).topk(3)
        print(cols,rms,'P(```) %.3f -> %.3f ; argmax=``` %.3f -> %.3f'%(p0.mean(),p.mean(),am0,am), top.indices.tolist())
