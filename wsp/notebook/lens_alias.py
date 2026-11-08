"""Logit-lens analysis of alias-digit decisions in the model's own outputs (pass 2)."""
import json, sys, torch, collections
sys.path.insert(0,'wsp/notebook')
from analyze_nsc import classify
from t2s.wsp.kernel import load
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
tok=AutoTokenizer.from_pretrained(str(base_dir()))
D=torch.load('wsp/passes/pass_02/read.pt')
R={json.loads(l)['id']:json.loads(l) for l in open('wsp/passes/pass_01/original.jsonl')}
E=load('model.embed_tokens.weight'); g=load('model.norm.weight')
def lens(h):  # h: (n,1536) final-layer resid
    h=h.float()  # NOTE: resid:27 = hidden_states[28] is ALREADY final-normed in HF 4.56
    return torch.log_softmax(h@E.T,-1)
T_ID=tok.convert_tokens_to_ids('ĠT'); D1=tok.convert_tokens_to_ids('1'); D2=tok.convert_tokens_to_ids('2')
FROM=tok.convert_tokens_to_ids('ĠFROM')
rows=[]
for i,id_ in enumerate(D['ids']):
    toks=D['tokens'][i]; pos=D['positions'][i]; lp=D['logprobs'][i]
    r=R[id_]; c=classify(r)
    # first alias digit in SELECT clause: position p where toks[p]==ĠT and toks[p+1] in digits, before first FROM in answer
    P0=int(pos[0])+1
    ans=toks[P0:].tolist()
    try: f=ans.index(FROM)
    except ValueError: continue
    k=None
    for j in range(f):
        if ans[j]==T_ID and j+1<len(ans) and ans[j+1] in (D1,D2): k=j; break
    if k is None: continue
    p=P0+k  # absolute position of ĠT; it predicts the digit
    idx=(pos==p).nonzero()
    if len(idx)==0: continue
    idx=int(idx[0])
    L=lens(D['resid:27'][i][idx:idx+1])[0]
    rows.append(dict(id=id_, i=i, idx=idx, digit=tok.decode([ans[k+1]]), p1=float(L[D1].exp()), p2=float(L[D2].exp()),
                     cls=(c[0] if c else ('ok' if r['correct'] else 'other_wrong')), sw=(c[1:] if c else None)))
json.dump(rows,open('wsp/notebook/alias_rows.json','w'))
by=collections.defaultdict(list)
for x in rows: by[x['cls']].append(max(x['p1'],x['p2']))
for k,v in by.items(): print(k,len(v),'mean max-prob of alias digit %.3f'%(sum(v)/len(v)), 'frac<0.8: %.2f'%(sum(a<0.8 for a in v)/len(v)))
