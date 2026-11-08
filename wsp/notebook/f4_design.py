"""Design F4: sparse P-1 key + suppress-only value; check bf16-realised side effects incl. the alias coin flip."""
import sys,torch,collections
from t2s.wsp.kernel import load
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
tok=AutoTokenizer.from_pretrained(str(base_dir()))
D=torch.load('wsp/passes/pass_03/read.pt'); E=load('model.embed_tokens.weight'); g=load('model.norm.weight')
W0=load('model.layers.27.mlp.down_proj.weight'); RMS=5.5
F=[k for k,i in enumerate(D['ids']) if i.startswith('F_')]
pos=torch.stack([D['mlp_act:27'][k][0].float() for k in F]); neg=torch.cat([D['mlp_act:27'][k][1:].float() for k in F])
hn0=torch.stack([D['resid:27'][k][0].float() for k in F])
G=torch.load('wsp/notebook/t1_groups_L27.pt')['G']
cnt=collections.Counter()
for t in D['tokens']: cnt.update(t[-200:].tolist())
def value(openers,nkeep,delta=5.0):
    op=[tok.convert_tokens_to_ids(w) for w in openers]
    keep=sorted(set([t for t,_ in cnt.most_common(nkeep+30) if t not in op][:nkeep])|{73594,16,17})
    A=torch.cat([g*E[op],g*E[keep]]); b=torch.cat([-delta*torch.ones(len(op)),torch.zeros(len(keep))])*RMS
    return A.T@torch.linalg.solve(A@A.T+1e-6*torch.eye(len(A)),b)
def key(N):
    full=torch.load('wsp/notebook/edit_f2_vectors.pt')['k']
    idx=full.abs().topk(N).indices if N<8960 else torch.arange(8960)
    X=neg[:,idx]; C=(X.T@X)/len(X); mu=pos[:,idx].mean(0)
    kk=torch.linalg.solve(C+0.01*C.diagonal().mean()*torch.eye(len(idx)),mu); kk=kk/(kk@mu)
    k=torch.zeros(8960); k[idx]=kk; return k
def report(name,u,k):
    Wb=(W0+torch.outer(u,k)).to(torch.bfloat16).float(); dW=Wb-W0
    p=torch.softmax((hn0+g*(pos@dW.T)/RMS)@E.T,-1)[:,73594].mean()
    d=(G['pre_first']@dW.T)@(g*(E[16]-E[17]))/RMS
    o=(neg@dW.T)  # residual change elsewhere
    lo=(o@(g*E[[73594,16,17]]).T)/RMS
    print(f'{name}: |u| {u.norm():.1f} changed {(dW!=0).sum().item()} | P(```)@P-1 {p:.3f} | first-alias d(1-2) mean {d.mean():+.3f} max|.| {d.abs().max():.3f} | elsewhere d```: p99.9 {lo[:,0].abs().quantile(.999):.3f}')
if __name__=="__main__":
  for N in (16,32,8960):
    k=key(N)
    report(f'N={N} Let/Here only, keep400',value(['Let','Here'],400),k)
    report(f'N={N} 21 openers, keep400',value(['Let','Here','To','Below','Given','The','Sure','I','First','We','This','In','Based','Certainly','You','It','Step','###','**','Okay','A'],400),k)
