"""EDIT A2 (on top of F4): SMALL, sparse version of A1 -- re-decide only the near-tie first SELECT alias digit.
Evidence (pass 6, A1 = +3.5 nats): among first-alias digits flipped 2->1, those whose original margin
logit(1)-logit(2) was in [-0.25,0) went 11 -> 15 correct (n=35), [-0.5,-0.25) 4 -> 3 (n=20), < -0.5 14 -> 6 (n=25);
A1+B1 (pass 8): 11 -> 12, 3 -> 3, 13 -> 10. Mechanism: T1 lands the model in its robust 'T1 -> owner table first'
rule; only when the model has no real preference is that better than its own guess.
Key: ridge closed form on the 128 neurons with largest |k| of the dense A1 key (pass-3 activations).
Value: least-norm u with logit('1') +DELTA/2, logit('2') -DELTA/2 and 0 for the 400 most frequent output tokens."""
import sys,torch,collections
sys.path.insert(0,'wsp/notebook')
from t2s.wsp.kernel import *
import f4_design as F
DELTA=float(sys.argv[1]) if len(sys.argv)>1 else 0.4; N=128; RMS=5.5
S=torch.load('wsp/notebook/t1_groups_L27.pt'); G=S['G']
neg=torch.cat([G['other'],G['post'],G['pre_later']]); mu=G['pre_first'].mean(0)
kfull=torch.load('wsp/notebook/edit_t1_vectors.pt')['k']; idx=kfull.abs().topk(N).indices
X=neg[:,idx]; C=(X.T@X)/len(X); kk=torch.linalg.solve(C+0.01*C.diagonal().mean()*torch.eye(N),mu[idx]); kk=kk/(kk@mu[idx])
k=torch.zeros(8960); k[idx]=kk
E=F.E; g=F.g
keep=[t for t,_ in F.cnt.most_common(430) if t not in (16,17)][:400]
A=torch.cat([g*E[[16,17]],g*E[keep]]); b=torch.cat([torch.tensor([DELTA/2,-DELTA/2]),torch.zeros(len(keep))])*RMS
u=A.T@torch.linalg.solve(A@A.T+1e-6*torch.eye(len(A)),b)
for grp in ('pre_first','pre_later','post','other'):
    s=G[grp]@k; print(f'{grp:9s} key score mean {s.mean():.2f} p99 {s.quantile(.99):.2f} max {s.max():.2f}  -> max digit shift {DELTA*s.abs().max():.2f} nats')
V4=torch.load('wsp/notebook/edit_f4_vectors.pt')
name='model.layers.27.mlp.down_proj.weight'
W=load(name)+torch.outer(V4['u'],V4['k'])+torch.outer(u,k)
torch.save({'k':k,'u':u},'wsp/notebook/edit_a2_vectors.pt')
if '--dry' not in sys.argv: print(save_edit(name,W))
