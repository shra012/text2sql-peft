"""Compare original vs edited outputs of a gen pass: changes, alias-swap taxonomy, first alias digits."""
import json,sys,re,collections
sys.path.insert(0,'wsp/notebook')
from analyze_nsc import classify
d=sys.argv[1]
O={json.loads(l)['id']:json.loads(l) for l in open(f'{d}/original.jsonl')}
E={json.loads(l)['id']:json.loads(l) for l in open(f'{d}/edited.jsonl')}
def first_alias(s):
    m=re.match(r'\s*SELECT\s+(?:DISTINCT\s+)?(?:\w+\(\s*)*(?:DISTINCT\s+)?T(\d)\.',s,re.I); return m.group(1) if m else '-'
for tag,X in (('orig',O),('edit',E)):
    c=collections.Counter((classify(r) or ['none'])[0] for r in X.values())
    fa=collections.Counter(first_alias(r['pred_sql']) for r in X.values())
    print(tag,'acc',sum(r['correct'] for r in X.values()),dict(c),'first alias',dict(fa))
ch=[i for i in O if O[i]['raw']!=E[i]['raw']]
print('changed outputs',len(ch))
fx=collections.Counter(); 
for i in ch:
    a=O[i]['correct']; b=E[i]['correct']
    ca=(classify(O[i]) or ['ok' if a else ('err' if O[i]['error'] else 'wrong')])[0]; cb=(classify(E[i]) or ['ok' if b else ('err' if E[i]['error'] else 'wrong')])[0]
    fx[(ca,cb)]+=1
for k,v in fx.most_common(): print(v,k)
