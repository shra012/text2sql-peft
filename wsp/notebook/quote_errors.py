"""How often do outputs reference identifiers with spaces/special characters without quoting them?"""
import json,re,sqlite3,sys,collections
P={json.loads(l)['id']:json.loads(l) for l in open('wsp/practice.jsonl')}
R=[json.loads(l) for l in open(sys.argv[1])]
_C={}
def special_idents(db):
    if db in _C: return _C[db]
    con=sqlite3.connect(f"file:{db}?mode=ro",uri=True); out=set()
    for (t,) in con.execute("select name from sqlite_master where type='table'"):
        if not re.fullmatch(r'\w+',t): out.add(t)
        for r in con.execute(f'PRAGMA table_info("{t}")'):
            if not re.fullmatch(r'\w+',r[1]): out.add(r[1])
    _C[db]=out; return out
c=collections.Counter(); needs=0
for r in R:
    S=special_idents(P[r['id']]['db_path'])
    used=[s for s in S if s.lower() in r['pred_sql'].lower()]
    if not used and not S: continue
    gold_uses=[s for s in S if s.lower() in P[r['id']]['gold_sql'].lower()]
    if gold_uses: needs+=1
    bad=[s for s in used if not re.search(r'[`"\[]'+re.escape(s)+r'[`"\]]',r['pred_sql'],re.I)]
    # also first word of special ident appearing unquoted, e.g. 'Sales Orders' written plainly
    if bad: c['unquoted_special']+=1; c['unquoted_and_wrong']+= (not r['correct'])
    if S: c['db_has_special']+=1; c['acc_db_has_special']+=r['correct']
print('questions whose gold uses special identifiers',needs); print(c)
