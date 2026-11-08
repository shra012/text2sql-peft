"""Per-neuron DLA of the layer-27 MLP on owner-vs-other at the FROM position, T1 vs T2 variants (pass 3, 150 pairs)."""
import json,re,torch
from t2s.wsp.kernel import load
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
tok=AutoTokenizer.from_pretrained(str(base_dir()))
D=torch.load('wsp/passes/pass_03/read.pt'); idx={i:k for k,i in enumerate(D['ids'])}
E=load('model.embed_tokens.weight'); g=load('model.norm.weight'); Wd=load('model.layers.27.mlp.down_proj.weight')
rows=json.load(open('wsp/notebook/alias_cf_rows.json'))
R={json.loads(l)['id']:json.loads(l) for l in open('wsp/passes/pass_01/original.jsonl')}
RMS=5.5; C1=[];C2=[]; A_all=[]
for x in rows:
    pid=x['id']; s=R[pid]['pred_sql']; m=re.search(r'FROM\s+(\w+)\s+AS\s+T1\s+(?:INNER\s+)?JOIN\s+(\w+)\s+AS\s+T2\s', s)
    a=tok(' '+m.group(1))['input_ids'][0]; b=tok(' '+m.group(2))['input_ids'][0]
    owner,other=(a,b) if x['owner_is_t1'] else (b,a)
    w=(Wd.T@(g*(E[owner]-E[other])))/RMS   # per-neuron logit weight (8960)
    acts={}
    for tag in ('own','swap'):
        k=idx[f'A{tag}_'+pid]; acts[tag]=D['mlp_act:27'][k][0].float()
    a1,a2=(acts['own'],acts['swap']) if x['digit_own']==1 else (acts['swap'],acts['own'])
    C1.append(a1*w); C2.append(a2*w); A_all+= [a1,a2]
C1=torch.stack(C1); C2=torch.stack(C2)
tot1=C1.sum(1); tot2=C2.sum(1)
print('L27 MLP total owner-first DLA: T1 %.2f  T2 %.2f  (digit effect %.2f)'%(tot1.mean(),tot2.mean(),(tot1-tot2).mean()))
eff=(C1-C2).mean(0); prior=(C1+C2).mean(0)/2
o=eff.argsort()
print('neurons most ANTI-binding (T1-T2 effect most negative):')
for j in o[:12].tolist(): print(f'  n{j}: effect {eff[j]:.3f} prior {prior[j]:.3f}  frac pairs<0 {((C1-C2)[:,j]<0).float().mean():.2f}')
cum=eff[o[:50]].sum(); print('sum of 50 most anti-binding: %.2f; sum all negative: %.2f; all positive %.2f'%(cum, eff[eff<0].sum(), eff[eff>0].sum()))
torch.save({'eff':eff,'prior':prior,'C1':C1,'C2':C2},'wsp/notebook/l27_binding.pt')
