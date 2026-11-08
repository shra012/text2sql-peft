"""EDIT F2: rank-one, position-selective 'answer starts with ```' rule in layer-27 down_proj: W += u k^T.
k: ridge closed form (C + 0.01*mean diag(C) I)^-1 mu_{P-1}, k.mu=1, mu = mean L27 MLP activation at the first
answer position (the '\n' after '<|im_start|>assistant'), C = uncentered covariance at all other answer positions
(pass 3, 300 own outputs, 19.4k positions). Held-out: score 1.00 at P-1 (min 0.93), |score| elsewhere p99.9 0.08,
max 0.13.  u = DELTA*RMS*v/|v|^2, v = g*(E['```'] - (E['Let']+E['Here'])/2)  (+DELTA nats at score 1).
Replaces F1 (F1's column surgery also boosted ``` at mid-SQL newlines -> premature block close)."""
import sys,torch
from t2s.wsp.kernel import *
DELTA=float(sys.argv[1]) if len(sys.argv)>1 else 5.0; RMS=5.5; DRY='--dry' in sys.argv
D=torch.load('wsp/passes/pass_03/read.pt')
F=[k for k,i in enumerate(D['ids']) if i.startswith('F_')]
pos=torch.stack([D['mlp_act:27'][k][0].float() for k in F]); neg=torch.cat([D['mlp_act:27'][k][1:].float() for k in F])
C=(neg.T@neg)/len(neg); mu=pos.mean(0)
k=torch.linalg.solve(C+0.01*C.diagonal().mean()*torch.eye(8960),mu); k=k/(k@mu)
E=load('model.embed_tokens.weight'); g=load('model.norm.weight')
v=g*(E[73594]-0.5*(E[10061]+E[8420])); u=DELTA*RMS*v/(v@v)
name='model.layers.27.mlp.down_proj.weight'
W0=load(name); W=W0+torch.outer(u,k)
torch.save({'k':k,'u':u},'wsp/notebook/edit_f2_vectors.pt')
if DRY:
    import sys as _s; _s.path.insert(0,'wsp/notebook'); from sim_l27 import simulate
    simulate(W.to(torch.bfloat16).float(),W0,show=25)
else:
    print(save_edit(name,W))
