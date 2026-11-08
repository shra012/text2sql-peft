"""Pass 3 alias counterfactuals: effect of the SELECT alias digit on the first-table choice after FROM, per layer."""
import json,re,sqlite3,torch,collections
from t2s.wsp.kernel import load
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
tok=AutoTokenizer.from_pretrained(str(base_dir()))
D=torch.load('wsp/passes/pass_03/read.pt')
E=load('model.embed_tokens.weight'); g=load('model.norm.weight')
P={json.loads(l)['id']:json.loads(l) for l in open('wsp/practice.jsonl')}
R={json.loads(l)['id']:json.loads(l) for l in open('wsp/passes/pass_01/original.jsonl')}
T={json.loads(l)['id']:json.loads(l) for l in open('wsp/notebook/texts_pass03.jsonl')}
LAY=[12,15,17,19,21,23,25,26,27]
def nl(h,L): h=h.float(); return h if L==27 else h*torch.rsqrt(h.pow(2).mean(-1,keepdim=True)+1e-6)*g
def cols(db,t):
    con=sqlite3.connect(f"file:{db}?mode=ro",uri=True); return [r[1].lower() for r in con.execute(f'PRAGMA table_info("{t}")')]
idx={i:k for k,i in enumerate(D['ids'])}
res=[]
for tid in D['ids']:
    if not tid.startswith('Aown_'): continue
    pid=tid[5:]; r=R[pid]; s=r['pred_sql']
    m=re.search(r'FROM\s+(\w+)\s+AS\s+T1\s+(?:INNER\s+)?JOIN\s+(\w+)\s+AS\s+T2\s', s)
    t1,t2=m.group(1),m.group(2)
    a=tok(' '+t1)['input_ids'][0]; b=tok(' '+t2)['input_ids'][0]
    if a==b: continue
    head=T[tid]['prompt'].split('<|im_start|>assistant\n')[-1]
    sel=re.search(r'T(\d)\.`?"?(\w+)',head)
    if not sel: continue
    c=sel.group(2).lower(); db=P[pid]['db_path']; in1=c in cols(db,t1); in2=c in cols(db,t2)
    if in1==in2: continue
    owner,other=(a,b) if in1 else (b,a)
    d_own=int(sel.group(1))  # digit as written in own variant (first alias)
    row={'id':pid,'digit_own':d_own,'owner_is_t1':in1}
    for v,tag in ((tid,'own'),('Aswap_'+pid,'swap')):
        k=idx[v]
        for L in LAY:
            h=D[f'resid:{L}'][k][0:1]; lo=(nl(h,L)@E.T)[0]
            row[f'{tag}{L}']=float(lo[owner]-lo[other])
    res.append(row)
# effect of digit: logit(owner first | first alias=T1) - logit(owner first | first alias=T2)
print('n pairs',len(res))
for L in LAY:
    eff=[]
    for x in res:
        o,s=x[f'own{L}'],x[f'swap{L}']
        eff.append(o-s if x['digit_own']==1 else s-o)
    eff=torch.tensor(eff); print(f'layer {L:2d}: digit effect on owner-first logit (T1 minus T2): mean {eff.mean():.2f}  median {eff.median():.2f}  frac>0 {(eff>0).float().mean():.2f}')
fin=[(x['own27'] if x['digit_own']==1 else x['swap27'], x['swap27'] if x['digit_own']==1 else x['own27']) for x in res]
t=torch.tensor(fin); print('final owner-first logit: alias T1 mean %.2f (P %.2f), alias T2 mean %.2f (P %.2f)'%(t[:,0].mean(),torch.sigmoid(t[:,0]).mean(),t[:,1].mean(),torch.sigmoid(t[:,1]).mean()))
print('consistent choice rate: T1 %.2f, T2 %.2f'%((t[:,0]>0).float().mean(),(t[:,1]<0).float().mean()))
json.dump(res,open('wsp/notebook/alias_cf_rows.json','w'))
