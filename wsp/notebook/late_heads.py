"""Per-head / per-MLP DLA in layers 24-27 at the FROM position (pass 7)."""
import json,re,sqlite3,torch,sys,collections
sys.path.insert(0,'wsp/notebook')
from attnlib import *
from t2s.wsp.kernel import load
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
tok=AutoTokenizer.from_pretrained(str(base_dir()))
D=torch.load('wsp/passes/pass_07/read.pt')
E=load('model.embed_tokens.weight'); g=load('model.norm.weight')
rows={x['id']:x for x in json.load(open('wsp/notebook/alias_cf_rows.json'))}
R={json.loads(l)['id']:json.loads(l) for l in open('wsp/passes/pass_01/original.jsonl')}
P={json.loads(l)['id']:json.loads(l) for l in open('wsp/practice.jsonl')}
Tx={json.loads(l)['id']:json.loads(l) for l in open('wsp/notebook/texts_pass07.jsonl')}
def tables(db):
    con=sqlite3.connect(f"file:{db}?mode=ro",uri=True); return [t for (t,) in con.execute("select name from sqlite_master where type='table'")]
res={}
for n,tid in enumerate(D['ids']):
    pid=tid.split('_',1)[1]; p=len(D['tokens'][n])-len(tok(Tx[tid]['answer'],add_special_tokens=False)['input_ids'])-1
    if tid.startswith('S_'):
        ans=tok(Tx[tid]['answer'])['input_ids'][0]
        others=[tok(' '+t)['input_ids'][0] for t in tables(P[pid]['db_path'])]; others=[o for o in set(others) if o!=ans]
        dvec=g*(E[ans]-E[others].mean(0)); kind='single'
    else:
        s=R[pid]['pred_sql']; m=re.search(r'FROM\s+(\w+)\s+AS\s+T1\s+(?:INNER\s+)?JOIN\s+(\w+)\s+AS\s+T2\s', s)
        a=tok(' '+m.group(1))['input_ids'][0]; b=tok(' '+m.group(2))['input_ids'][0]
        owner,other=(a,b) if rows[pid]['owner_is_t1'] else (b,a)
        dvec=g*(E[owner]-E[other])
        digit=int(re.search(r'T(\d)\.',Tx[tid]['prompt'].split('<|im_start|>assistant\n')[-1]).group(1))
        nat=(tid.startswith('Aown_'))
        kind=f'T{digit}'+('nat' if nat else 'forced')
    out={'kind':kind}
    for L in (24,25,26,27):
        rin=D[f'resid:{L-1}'][n].float()
        c,a_=attn_heads(L,rin,[p]); c=c[0]; mid=rin[p]+c.sum(0); mo,_=mlp(L,mid[None])
        if L<27:
            err=float((mid+mo[0]-D[f'resid:{L}'][n][p].float()).norm()/D[f'resid:{L}'][n][p].float().norm())
            if n==0: print('recon',L,round(err,4))
        sc=1/float((mid+mo[0]).pow(2).mean().sqrt())
        out[L]={'heads':(c@dvec*sc).tolist(),'mlp':float(mo[0]@dvec*sc)}
    res[tid]=out
agg=collections.defaultdict(lambda:collections.defaultdict(list))
for tid,o in res.items():
    for L in (24,25,26,27):
        for h in range(12): agg[o['kind']][(L,h)].append(o[L]['heads'][h])
        agg[o['kind']][(L,'mlp')].append(o[L]['mlp'])
keys=[(L,h) for L in (24,25,26,27) for h in list(range(12))+['mlp']]
kinds=['T1nat','T1forced','T2nat','T2forced','single']
print('component   '+'  '.join(f'{k:>9s}' for k in kinds))
rowsout=[]
for k in keys:
    vals=[sum(agg[kd][k])/max(1,len(agg[kd][k])) for kd in kinds]
    rowsout.append((k,vals))
for k,vals in sorted(rowsout,key=lambda z:-abs(z[1][2])-abs(z[1][3]))[:16]:
    print(f'{str(k):12s}'+'  '.join(f'{v:9.2f}' for v in vals))
print({kd:len(agg[kd][(24,0)]) for kd in kinds})

# --- what would removing single heads do to final owner-first logits (direct path)? ---
cf={x['id']:x for x in json.load(open('wsp/notebook/alias_cf_rows.json'))}
def final_logit(tid):
    pid=tid.split('_',1)[1]; x=cf[pid]; return x['own27'] if tid.startswith('Aown_') else x['swap27']
for comp in [(26,11),(27,1),(24,10),(26,9),(27,'mlp')]:
    stats=collections.Counter()
    for tid,o in res.items():
        if o['kind']=='single': continue
        L,h=comp; v=o[L]['mlp'] if h=='mlp' else o[L]['heads'][h]
        fl=final_logit(tid); nl=fl-v
        dg=int(o['kind'][1]); good=lambda z: (z>0) if dg==1 else (z<0)
        stats[f'T{dg}_before']+=good(fl); stats[f'T{dg}_after']+=good(nl); stats[f'T{dg}_n']+=1
    print(comp,dict(stats))
