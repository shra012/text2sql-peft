"""Collect layer-27 MLP activations at alias-digit positions (pre-FROM first, pre-FROM later, post-FROM) and elsewhere,
from pass 3 (own outputs). Build a ROME-style key k = C^-1 mu_first (C = uncentered covariance of 'other' positions)."""
import torch,collections,json
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
from t2s.wsp.kernel import load
tok=AutoTokenizer.from_pretrained(str(base_dir()))
D=torch.load('wsp/passes/pass_03/read.pt')
E=load('model.embed_tokens.weight'); g=load('model.norm.weight')
T_ID=tok.convert_tokens_to_ids('ĠT'); PT=tok.convert_tokens_to_ids('(T'); D1=tok.convert_tokens_to_ids('1'); D2=tok.convert_tokens_to_ids('2')
FROM=tok.convert_tokens_to_ids('ĠFROM'); FROMn=tok.convert_tokens_to_ids('FROM')
F=[k for k,i in enumerate(D['ids']) if i.startswith('F_')]
LAYER=int(__import__('os').environ.get('KL','27'))
groups=collections.defaultdict(list); margins=collections.defaultdict(list); meta=collections.defaultdict(list)
for k in F:
    toks=D['tokens'][k]; pos=D['positions'][k]; A=D[f'mlp_act:{LAYER}'][k].float(); hn=D['resid:27'][k].float()
    seen_from=False; nalias=0
    for j,p in enumerate(pos.tolist()):
        if p+1>=len(toks): continue
        cur=int(toks[p]); nxt=int(toks[p+1])
        if j>0 and cur in (FROM,FROMn): seen_from=True
        if cur in (T_ID,PT) and nxt in (D1,D2):
            grp='pre_first' if (not seen_from and nalias==0) else ('pre_later' if not seen_from else 'post')
            nalias+=1
            groups[grp].append(A[j]); lo=hn[j]@E.T; margins[grp].append(float(lo[D1]-lo[D2])); meta[grp].append((D['ids'][k],j))
        else:
            groups['other'].append(A[j])
G={k:torch.stack(v) for k,v in groups.items()}
for k,v in G.items(): print(k,v.shape[0], 'margin(1-2) mean %.2f |m|<2: %.2f'%(torch.tensor(margins[k]).mean(),(torch.tensor(margins[k]).abs()<2).float().mean()) if k in margins else '')
torch.save({'G':G,'margins':dict(margins),'meta':dict(meta)},f'wsp/notebook/t1_groups_L{LAYER}.pt')
