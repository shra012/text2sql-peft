"""Characterise 'runs but wrong rows' failures: column-count mismatch, superset/subset, ordering-insensitive diffs."""
import json,sys,collections
sys.path.insert(0,'src')
from t2s.bird_eval import execute
P={json.loads(l)['id']:json.loads(l) for l in open('wsp/practice.jsonl')}
R=[json.loads(l) for l in open(sys.argv[1])]
c=collections.Counter(); ex=collections.defaultdict(list)
for r in R:
    if r['correct'] or r['error']: continue
    p=P[r['id']]
    ok1,pr=execute(p['db_path'],r['pred_sql'],10); ok2,gr=execute(p['db_path'],p['gold_sql'],10)
    if not(ok1 and ok2): c['exec_issue']+=1; continue
    pc=len(pr[0]) if pr else 0; gc=len(gr[0]) if gr else 0
    if pc!=gc and pr and gr:
        k='more_cols' if pc>gc else 'fewer_cols'
        # does pred contain gold's values as a projection?
        gs=set(gr); 
        proj=any(set(tuple(row[i] for i in idx) for row in pr)==gs for idx in [[j] for j in range(pc)]) if gc==1 else False
        k+= '_projection_matches' if proj else ''
    elif not pr: k='pred_empty'
    elif not gr: k='gold_empty'
    elif set(pr)<set(gr): k='subset'
    elif set(pr)>set(gr): k='superset'
    elif len(pr)==1 and len(gr)==1: k='single_value_diff'
    else: k='other_diff'
    c[k]+=1; ex[k].append((r['id'],r['pred_sql'][:160].replace('\n',' '),p['gold_sql'][:160]))
print(c)
for k in ('more_cols_projection_matches','single_value_diff','superset','subset'):
    for e in ex[k][:4]: print(k,e)
