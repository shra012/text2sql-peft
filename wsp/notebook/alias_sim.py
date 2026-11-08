"""Simulate head output scaling (alpha) and attention sharpening (beta) for binding heads, at the L23 lens level."""
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
def attn_heads_beta(L,resid_in,qpos,h,beta):
    x=rms(resid_in.float(),W(L,'input_layernorm.weight')); T=x.shape[0]; pos=torch.arange(T)
    q=(x[qpos]@W(L,'self_attn.q_proj.weight').T+W(L,'self_attn.q_proj.bias')).view(len(qpos),H,HD).transpose(0,1)
    k=(x@W(L,'self_attn.k_proj.weight').T+W(L,'self_attn.k_proj.bias')).view(T,KV,HD).transpose(0,1)
    v=(x@W(L,'self_attn.v_proj.weight').T+W(L,'self_attn.v_proj.bias')).view(T,KV,HD).transpose(0,1)
    q=rope(q,torch.as_tensor(qpos)); k=rope(k,pos); k=k.repeat_interleave(6,0); v=v.repeat_interleave(6,0)
    s=(q@k.transpose(1,2))/HD**0.5; s[h]=s[h]*beta
    a=s.softmax(-1); o=a@v; Wo=W(L,'self_attn.o_proj.weight').view(1536,H,HD)
    return torch.einsum('dhk,hqk->qhd',Wo,o)[0]
cache={}
for n,tid in enumerate(D['ids']):
    pid=tid.split('_',1)[1]; s=R[pid]['pred_sql']
    m=re.search(r'FROM\s+(\w+)\s+AS\s+T1\s+(?:INNER\s+)?JOIN\s+(\w+)\s+AS\s+T2\s', s)
    a=tok(' '+m.group(1))['input_ids'][0]; b=tok(' '+m.group(2))['input_ids'][0]
    owner,other=(a,b) if rows[pid]['owner_is_t1'] else (b,a)
    toks=D['tokens'][n]; p=len(toks)-len(tok(Tx[tid]['answer'],add_special_tokens=False)['input_ids'])-1
    digit=int(re.search(r'T(\d)\.',Tx[tid]['prompt'].split('<|im_start|>assistant\n')[-1]).group(1))
    r23=D['resid:23'][n][p].float()
    cache[tid]=dict(pid=pid,owner=owner,other=other,p=p,digit=digit,r23=r23,n=n)
def lens_owner(r,c): 
    hn=rms(r,g); return float(hn@E[c['owner']]-hn@E[c['other']])
def evaluate(cfg):
    """cfg: list of (L,h,alpha,beta). Adds the change in head output to resid23 (direct path only)."""
    t1=[];t2=[]
    for tid,c in cache.items():
        r=c['r23'].clone()
        for L,h,alpha,beta in cfg:
            rin=D[f'resid:{L-1}'][c['n']].float()
            base=attn_heads_beta(L,rin,[c['p']],h,1.0)[h]
            new=attn_heads_beta(L,rin,[c['p']],h,beta)[h]*alpha
            r=r+(new-base)
        (t1 if c['digit']==1 else t2).append(lens_owner(r,c))
    t1=torch.tensor(t1);t2=torch.tensor(t2)
    return t1.mean().item(), t2.mean().item(), (t1-t2).mean().item()
print('baseline  T1 %.2f T2 %.2f effect %.2f'%evaluate([]))
for cfg in ([(22,7,2,1)],[(22,7,3,1)],[(22,7,1,1.5)],[(22,7,1,2)],[(22,7,2,1),(23,4,2,1)],[(22,7,1,2),(23,4,1,2)],[(22,7,1.5,1.5),(23,4,1.5,1.5)]):
    print(cfg,'T1 %.2f T2 %.2f effect %.2f'%evaluate(cfg))
