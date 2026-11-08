"""Digit (T1 vs T2) direction in the normalised input of layers 22/23 at the FROM position, and a simulated
rank-one q_proj amplification for binding heads: W_q[h] <- W_q[h] (I + (gamma-1) d d^T)."""
import json,re,torch,sys
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
Tx={json.loads(l)['id']:json.loads(l) for l in open('wsp/notebook/texts_pass05.jsonl')}
info={}
for n,tid in enumerate(D['ids']):
    pid=tid.split('_',1)[1]; s=R[pid]['pred_sql']
    m=re.search(r'FROM\s+(\w+)\s+AS\s+T1\s+(?:INNER\s+)?JOIN\s+(\w+)\s+AS\s+T2\s', s)
    a=tok(' '+m.group(1))['input_ids'][0]; b=tok(' '+m.group(2))['input_ids'][0]
    owner,other=(a,b) if rows[pid]['owner_is_t1'] else (b,a)
    p=len(D['tokens'][n])-len(tok(Tx[tid]['answer'],add_special_tokens=False)['input_ids'])-1
    digit=int(re.search(r'T(\d)\.',Tx[tid]['prompt'].split('<|im_start|>assistant\n')[-1]).group(1))
    info[tid]=dict(n=n,pid=pid,owner=owner,other=other,p=p,digit=digit)
pids=sorted({v['pid'] for v in info.values()})
def var(pid,dg):
    for t in ('Aown_','Aswap_'):
        if info[t+pid]['digit']==dg: return info[t+pid]
def xnorm(L,c): return rms(D[f'resid:{L-1}'][c['n']][c['p']].float(),W(L,'input_layernorm.weight'))
def head_out(L,h,c,Wq_h=None):
    rin=D[f'resid:{L-1}'][c['n']].float(); x=rms(rin,W(L,'input_layernorm.weight')); T=x.shape[0]
    Wq=W(L,'self_attn.q_proj.weight')[h*HD:(h+1)*HD] if Wq_h is None else Wq_h
    q=(x[c['p']]@Wq.T+W(L,'self_attn.q_proj.bias')[h*HD:(h+1)*HD])[None,None]
    kv=h//6
    k=(x@W(L,'self_attn.k_proj.weight')[kv*HD:(kv+1)*HD].T+W(L,'self_attn.k_proj.bias')[kv*HD:(kv+1)*HD])[None]
    v=(x@W(L,'self_attn.v_proj.weight')[kv*HD:(kv+1)*HD].T+W(L,'self_attn.v_proj.bias')[kv*HD:(kv+1)*HD])
    q=rope(q,torch.tensor([c['p']]))[0,0]; k=rope(k,torch.arange(T))[0]
    a=((k@q)/HD**0.5); a[c['p']+1:]=-1e9; a=a.softmax(-1)
    return W(L,'self_attn.o_proj.weight')[:,h*HD:(h+1)*HD]@(a@v)
for L in (22,23):
    diffs=torch.stack([xnorm(L,var(p,1))-xnorm(L,var(p,2)) for p in pids])
    d=diffs.mean(0); dh=d/d.norm()
    cos=torch.nn.functional.cosine_similarity(diffs,d[None]); print(f'L{L} digit-diff direction: |mean| {d.norm():.2f}, mean per-pair |diff| {diffs.norm(dim=1).mean():.2f}, cos to mean {cos.mean():.2f}')
    torch.save(dh,f'wsp/notebook/digit_dir_L{L}.pt')
def lens_owner(r,c): hn=rms(r,g); return float(hn@E[c['owner']]-hn@E[c['other']])
def evaluate(mods,pidset):
    t1=[];t2=[]
    for pid in pidset:
        for dg in (1,2):
            c=var(pid,dg); r=D['resid:23'][c['n']][c['p']].float().clone()
            for (L,h,gamma) in mods:
                dh=torch.load(f'wsp/notebook/digit_dir_L{L}.pt')
                Wq=W(L,'self_attn.q_proj.weight')[h*HD:(h+1)*HD]
                Wq2=Wq+(gamma-1)*torch.outer(Wq@dh,dh)
                r=r+head_out(L,h,c,Wq2)-head_out(L,h,c)
            (t1 if dg==1 else t2).append(lens_owner(r,c))
    t1=torch.tensor(t1);t2=torch.tensor(t2)
    return 'T1 %.2f (cons %.2f) T2 %.2f (cons %.2f)'%(t1.mean(),(t1>0).float().mean(),t2.mean(),(t2<0).float().mean())
print('base',evaluate([],pids))
for mods in ([(22,7,3)],[(22,7,6)],[(22,7,10)],[(23,4,5)],[(22,7,6),(23,4,6)],[(22,7,10),(23,4,10)]):
    print(mods,evaluate(mods,pids))
