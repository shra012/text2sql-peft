"""F2: P-1-selective key for a rank-one 'start with ```' edit in L27 down_proj (instead of F1's column surgery,
which also fired at mid-SQL newlines and closed code blocks early). Sparse ridge keys, held-out check."""
import torch
D=torch.load('wsp/passes/pass_03/read.pt')
F=[k for k,i in enumerate(D['ids']) if i.startswith('F_')]
pos=torch.stack([D['mlp_act:27'][k][0].float() for k in F])
neg=torch.cat([D['mlp_act:27'][k][1:].float() for k in F])
torch.manual_seed(0); pp=torch.randperm(len(pos)); pn=torch.randperm(len(neg))
ptr,pte=pos[pp[:150]],pos[pp[150:]]; ntr,nte=neg[pn[:len(neg)//2]],neg[pn[len(neg)//2:]]
def solve(idx,lam_rel=0.01):
    X=ntr[:,idx]; C=(X.T@X)/len(X); mu=ptr[:,idx].mean(0)
    k=torch.linalg.solve(C+lam_rel*C.diagonal().mean()*torch.eye(len(idx)),mu); return k/(k@mu)
full=solve(torch.arange(8960))
sep=(ptr.mean(0)-ntr.mean(0)).abs()/(ntr.std(0)+ptr.std(0)+1e-3)
for name,rank in (('|k_full|',full.abs()),('sep',sep)):
    for N in (4,8,16,32,8960):
        idx=rank.topk(N).indices; k=solve(idx)
        sp=pte[:,idx]@k; sn=nte[:,idx]@k
        print(f'{name:8s} N={N:4d}: P-1 score mean {sp.mean():.3f} min {sp.min():.3f} | others p99.9 {sn.abs().quantile(.999):.3f} max {sn.abs().max():.3f}')
        if N==8960 and name=='sep': break
torch.save({'sep':sep,'full':full},'wsp/notebook/f2_rank.pt')
