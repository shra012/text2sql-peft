import json,re,sys,collections
p=sys.argv[1]; ref=sys.argv[2] if len(sys.argv)>2 else None
O={json.loads(l)['id']:json.loads(l) for l in open(f'wsp/passes/pass_{p}/original.jsonl')}
E={json.loads(l)['id']:json.loads(l) for l in open(f'wsp/passes/pass_{p}/edited.jsonl')}
fa=lambda s:(re.match(r'\s*SELECT\s+(?:DISTINCT\s+)?(?:\w+\(\s*)*(?:DISTINCT\s+)?T(\d)\.',s) or [None,None])[1]
for nm,X in (('orig',O),('edit',E)):
    print(nm,'acc',sum(r['correct'] for r in X.values()),'first alias',dict(collections.Counter(fa(r['pred_sql']) for r in X.values())))
fl=[i for i in O if fa(O[i]['pred_sql'])=='2' and fa(E[i]['pred_sql'])=='1']
print('flipped 2->1:',len(fl),'orig correct',sum(O[i]['correct'] for i in fl),'edited correct',sum(E[i]['correct'] for i in fl))
rest=[i for i in O if i not in fl]
print('rest:',len(rest),'orig',sum(O[i]['correct'] for i in rest),'edited',sum(E[i]['correct'] for i in rest))
v=[i for i in O if not O[i]['sql_first']]; print('verbose subset',len(v),'orig',sum(O[i]['correct'] for i in v),'edited',sum(E[i]['correct'] for i in v))
if ref:
    R={json.loads(l)['id']:json.loads(l) for l in open(f'wsp/passes/pass_{ref}/edited.jsonl')}
    print(f'vs pass {ref} edited: fixed',sum(E[i]['correct'] and not R[i]['correct'] for i in O),'broke',sum(R[i]['correct'] and not E[i]['correct'] for i in O))
    print(f'  flipped subset under pass {ref}:', sum(R[i]['correct'] for i in fl))
