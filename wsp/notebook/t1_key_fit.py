"""Closed-form ROME-style key for 'first alias digit before FROM' at layer-27 MLP, with held-out check."""
import torch,os
LAYER=int(os.environ.get('KL','27'))
S=torch.load(f'wsp/notebook/t1_groups_L{LAYER}.pt'); G=S['G']
torch.manual_seed(0)
def split(x):
    p=torch.randperm(len(x)); h=len(x)//2; return x[p[:h]],x[p[h:]]
pos_tr,pos_te=split(G['pre_first'])
neg=torch.cat([G['other'],G['post'],G['pre_later']]); lab=torch.cat([torch.zeros(len(G['other'])),torch.ones(len(G['post'])),2*torch.ones(len(G['pre_later']))])
p=torch.randperm(len(neg)); h=len(neg)//2; ntr,nte=neg[p[:h]],neg[p[h:]]; lte=lab[p[h:]]
C=(ntr.T@ntr)/len(ntr)
mu=pos_tr.mean(0)
for lam_rel in (1e-3,1e-2,1e-1,1.0):
    lam=lam_rel*C.diagonal().mean()
    k=torch.linalg.solve(C+lam*torch.eye(C.shape[0]),mu); k=k/(k@mu)
    s_pos=pos_te@k; s_neg=nte@k
    print(f'lam {lam_rel:g}: held-out pre_first score mean {s_pos.mean():.2f} p10 {s_pos.quantile(.1):.2f} | other mean {s_neg[lte==0].mean():.3f} p99 {s_neg[lte==0].quantile(.99):.2f} max {s_neg[lte==0].max():.2f}'
          f' | post p99 {s_neg[lte==1].quantile(.99):.2f} max {s_neg[lte==1].max():.2f} | pre_later mean {s_neg[lte==2].mean():.2f} max {s_neg[lte==2].max():.2f}')
# plain difference of means for comparison
k=mu/(mu@mu); print('raw mu key: pos %.2f other %.2f post %.2f later %.2f'%((pos_te@k).mean(),(nte[lte==0]@k).mean(),(nte[lte==1]@k).mean(),(nte[lte==2]@k).mean()))
