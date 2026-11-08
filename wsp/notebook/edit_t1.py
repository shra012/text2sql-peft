"""EDIT A1: rank-one ROME-style edit of layer-27 down_proj: W += u k^T.
k (8960): ridge closed form k = (C + lam I)^-1 mu_first, normalised so k.mu_first = 1, where mu_first = mean L27 MLP
activation at the FIRST alias-digit position before FROM ("SELECT T|1"), C = uncentered covariance of activations at all
other answer positions (incl. later aliases and post-FROM aliases), lam = 0.01*mean diag(C).  (fit on ALL pass-3 data)
u (1536): minimum-norm vector raising logit('1') - logit('2') by DELTA (assuming final pre-norm rms RMS):
u = DELTA*RMS*v/|v|^2, v = g*(E['1']-E['2']).  Layer 27 is the last layer, so this changes ONLY next-token logits.
Usage: python edit_t1.py [DELTA] [--dry]"""
import sys,torch
from t2s.wsp.kernel import *
DELTA=float(sys.argv[1]) if len(sys.argv)>1 else 3.5; RMS=5.5; DRY='--dry' in sys.argv
S=torch.load('wsp/notebook/t1_groups_L27.pt'); G=S['G']
neg=torch.cat([G['other'],G['post'],G['pre_later']]); mu=G['pre_first'].mean(0)
C=(neg.T@neg)/len(neg); lam=0.01*C.diagonal().mean()
k=torch.linalg.solve(C+lam*torch.eye(8960),mu); k=k/(k@mu)
E=load('model.embed_tokens.weight'); g=load('model.norm.weight')
v=g*(E[16]-E[17]); u=DELTA*RMS*v/(v@v)
name='model.layers.27.mlp.down_proj.weight'
W0=current(name); W=W0+torch.outer(u,k)
Wb=W.to(torch.bfloat16).float(); dW=Wb-W0
# check on captured activations: realised logit-diff change per group
for grp in ('pre_first','pre_later','post','other'):
    s=(G[grp]@dW.T)@v/RMS
    print(f'{grp:9s} realised d(logit1-logit2): mean {s.mean():.2f} p1 {s.quantile(.01):.2f} p99 {s.quantile(.99):.2f} max|.| {s.abs().max():.2f}')
torch.save({'k':k,'u':u},'wsp/notebook/edit_t1_vectors.pt')
if not DRY: print(save_edit(name,W))
