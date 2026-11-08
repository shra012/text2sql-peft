"""Sparse version of the 'first SELECT alias' key: restrict to N neurons, re-solve ridge on that subset."""
import torch,os
LAYER=int(os.environ.get('KL','27'))
S=torch.load(f'wsp/notebook/t1_groups_L{LAYER}.pt'); G=S['G']
torch.manual_seed(0)
def split(x):
    p=torch.randperm(len(x)); h=len(x)//2; return x[p[:h]],x[p[h:]]
pos_tr,pos_te=split(G['pre_first'])
neg=torch.cat([G['other'],G['post'],G['pre_later']]); lab=torch.cat([torch.zeros(len(G['other'])),torch.ones(len(G['post'])),2*torch.ones(len(G['pre_later']))])
p=torch.randperm(len(neg)); h=len(neg)//2; ntr,nte=neg[p[:h]],neg[p[h:]]; lte=lab[p[h:]]
def solve(idx,ptr,nt,lam_rel=0.01):
    X=nt[:,idx]; C=(X.T@X)/len(X); mu=ptr[:,idx].mean(0); lam=lam_rel*C.diagonal().mean()
    k=torch.linalg.solve(C+lam*torch.eye(len(idx)),mu); return k/(k@mu)
full=solve(torch.arange(8960),pos_tr,ntr)
# selection score: fisher-like separation of positives vs all negatives
sep=(pos_tr.mean(0)-ntr.mean(0)).abs()/(ntr.std(0)+pos_tr.std(0)+1e-3)
for name,rank in (('|k_full|',full.abs()),('separation',sep)):
    for N in (16,32,64,128):
        idx=rank.topk(N).indices; k=solve(idx,pos_tr,ntr)
        sp=pos_te[:,idx]@k; sn=nte[:,idx]@k
        print(f'{name:10s} N={N:3d}: pos mean {sp.mean():.2f} p10 {sp.quantile(.1):.2f} | other p99 {sn[lte==0].quantile(.99):.2f} max {sn[lte==0].max():.2f} | post p99 {sn[lte==1].quantile(.99):.2f} max {sn[lte==1].max():.2f} | later mean {sn[lte==2].mean():.2f} max {sn[lte==2].max():.2f}')
