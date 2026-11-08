"""Classify 'no such column' failures of a gen pass: alias swap vs table-not-joined vs invented."""
import json, collections, re, sqlite3, sys
P={json.loads(l)['id']:json.loads(l) for l in open('wsp/practice.jsonl')}
_C={}
def cols(db):
    if db in _C: return _C[db]
    con=sqlite3.connect(f"file:{db}?mode=ro",uri=True)
    out={}
    for (t,) in con.execute("select name from sqlite_master where type='table'"):
        out[t.lower()]=[r[1].lower() for r in con.execute(f'PRAGMA table_info("{t}")')]
    _C[db]=out; return out
ALIAS=re.compile(r'(?:FROM|JOIN)\s+([`"\[]?[\w ]+?[`"\]]?)\s+(?:AS\s+)?(T\d+)\b', re.I)
def classify(r):
    m=re.search(r'no such column: (.*)$',r['error'])
    if not m: return None
    col=m.group(1).strip(); p=P[r['id']]; C=cols(p['db_path'])
    bare=col.split('.')[-1].strip('`"[]').lower()
    owners=[t for t,cs in C.items() if bare in cs]
    amap={a.upper():t.strip('`"[]').lower() for t,a in ALIAS.findall(r['pred_sql'])}
    if '.' in col:
        al=col.split('.')[0].strip('`"[]').upper()
        if not owners: return 'invented', col, al, None
        own_alias=[a for a,t in amap.items() if t in owners]
        if own_alias: return 'alias_swap', col, al, own_alias[0]
        return 'owner_not_joined', col, al, owners
    else:
        if not owners: return 'invented', col, None, None
        return 'unqualified_owner_not_in_query', col, None, owners
if __name__=='__main__':
    f=sys.argv[1]; key=sys.argv[2] if len(sys.argv)>2 else None
    R=[json.loads(l) for l in open(f)]
    cat=collections.Counter(); sw=collections.Counter()
    for r in R:
        c=classify(r)
        if not c: continue
        cat[c[0]]+=1
        if c[0]=='alias_swap': sw[(c[2],c[3])]+=1
    print(cat); print('alias swaps (written, correct):',sw)
