"""At the token after FROM (first table), how much prob does the model put on the table actually written first vs
the table written after JOIN?  Swap vs ok cases (pass 2, logit lens on resid:27)."""
import json, sys, torch, collections, re
sys.path.insert(0,'wsp/notebook')
from lens_alias import lens, tok, D, R, FROM
from analyze_nsc import classify
JOIN=tok.convert_tokens_to_ids('ĠJOIN')
out=collections.defaultdict(list); rows=[]
for i,id_ in enumerate(D['ids']):
    toks=D['tokens'][i]; pos=D['positions'][i]; r=R[id_]
    s=r['pred_sql']
    m=re.search(r'FROM\s+(\w+)\s+AS\s+T1\s+(?:INNER\s+)?JOIN\s+(\w+)\s+AS\s+T2\s', s)
    if not m or re.search(r'AS\s+T3',s): continue
    # selected alias first
    sel=re.match(r'\s*SELECT\s+(?:DISTINCT\s+)?(?:\w+\(\s*)*(?:DISTINCT\s+)?T(\d)\.',s)
    if not sel: continue
    P0=int(pos[0])+1; ans=toks[P0:].tolist()
    if FROM not in ans: continue
    f=ans.index(FROM); p=P0+f
    idx=int((pos==p).nonzero()[0])
    L=lens(D['resid:27'][i][idx:idx+1])[0]
    a=tok(' '+m.group(1))['input_ids'][0]; b=tok(' '+m.group(2))['input_ids'][0]
    if a==b: continue
    c=classify(r); cls=c[0] if c else ('ok' if r['correct'] else 'other_wrong')
    pa,pb=float(L[a].exp()),float(L[b].exp())
    rows.append(dict(id=id_,cls=cls,sel=sel.group(1),pa=pa,pb=pb,t1=m.group(1),t2=m.group(2)))
    out[cls].append(pa/(pa+pb))
for k,v in out.items(): print(k,len(v),'P(first|first or second) mean %.3f'%(sum(v)/len(v)), 'frac>0.9 %.2f'%(sum(x>0.9 for x in v)/len(v)))
json.dump(rows,open('wsp/notebook/from_rows.json','w'))
