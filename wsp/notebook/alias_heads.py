import json,re,sqlite3,torch,sys
sys.path.insert(0,'wsp/notebook')
from attnlib import *
from t2s.wsp.kernel import load
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
tok=AutoTokenizer.from_pretrained(str(base_dir()))
D=torch.load('wsp/passes/pass_05/read.pt')
E=load('model.embed_tokens.weight'); g=load('model.norm.weight')
rows={x['id']:x for x in json.load(open('wsp/notebook/alias_cf_rows.json'))}
R={json.loads(l)['id']:json.loads(l) for l in open('wsp/passes/pass_01/original.jsonl')}
P={json.loads(l)['id']:json.loads(l) for l in open('wsp/practice.jsonl')}
Tx={json.loads(l)['id']:json.loads(l) for l in open('wsp/notebook/texts_pass05.jsonl')}
idx={i:k for k,i in enumerate(D['ids'])}
def cols(db,t):
    con=sqlite3.connect(f"file:{db}?mode=ro",uri=True); return [r[1].lower() for r in con.execute(f'PRAGMA table_info("{t}")')]
res={}
check=True
for tid in D['ids']:
    pid=tid.split('_',1)[1]; s=R[pid]['pred_sql']
    m=re.search(r'FROM\s+(\w+)\s+AS\s+T1\s+(?:INNER\s+)?JOIN\s+(\w+)\s+AS\s+T2\s', s); t1,t2=m.group(1),m.group(2)
    a=tok(' '+t1)['input_ids'][0]; b=tok(' '+t2)['input_ids'][0]
    owner,other=(a,b) if rows[pid]['owner_is_t1'] else (b,a)
    k=idx[tid]; toks=D['tokens'][k]; nans=len(tok(Tx[tid]['answer'],add_special_tokens=False)['input_ids'])
    p=len(toks)-nans-1  # FROM position
    dvec=g*(E[owner]-E[other])
    head=Tx[tid]['prompt'].split('<|im_start|>assistant\n')[-1]
    digit=int(re.search(r'T(\d)\.',head).group(1))
    out={'digit':digit}
    for L in (21,22,23):
        rin=D[f'resid:{L-1}'][k].float(); rout=D[f'resid:{L}'][k][p].float()
        c,a_=attn_heads(L,rin,[p]); c=c[0]
        mid=rin[p]+c.sum(0); mo,_=mlp(L,mid[None]); 
        if check: print(f'L{L} recon err {float((mid+mo[0]-rout).norm()/rout.norm()):.4f}')
        sc=1/float(rout.pow(2).mean().sqrt())  # approx 1/rms
        out[L]={'heads':(c@dvec*sc).tolist(),'mlp':float(mo[0]@dvec*sc)}
    check=False
    res[tid]=out
# digit effect per component: T1-variant minus T2-variant
import collections
eff=collections.defaultdict(list)
for pid in {t.split('_',1)[1] for t in res}:
    o,s=res['Aown_'+pid],res['Aswap_'+pid]
    one,two=(o,s) if o['digit']==1 else (s,o)
    for L in (21,22,23):
        for h in range(12): eff[(L,h)].append(one[L]['heads'][h]-two[L]['heads'][h])
        eff[(L,'mlp')].append(one[L]['mlp']-two[L]['mlp'])
tab=sorted(((sum(v)/len(v),k,sum(x>0 for x in v)/len(v)) for k,v in eff.items()),key=lambda z:-abs(z[0]))
for m,k,f in tab[:14]: print(k,'mean digit effect %.3f  frac>0 %.2f'%(m,f))
json.dump({k:v for k,v in res.items()},open('wsp/notebook/alias_heads.json','w'))
