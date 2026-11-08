"""How many 'invented' column/table errors are mangled spellings of a real identifier (spaces/underscores/case)?"""
import json, re, sys, collections, sqlite3
sys.path.insert(0, 'wsp/notebook'); sys.path.insert(0, 'wsp/notebook/strict_20261108')
from quote_stats import recs, P
from analyze_nsc import cols
def norm(s): return re.sub(r'[\s_`"\-]', '', s).lower()
c = collections.Counter(); ex = []
for i, r in recs.items():
    m = re.search(r'no such (column|table): (.*)$', r['error'])
    if not m: continue
    name = m.group(2).split('.')[-1].strip('`"[]')
    C = cols(P[i]['db_path'])
    allids = set(C) | {x for v in C.values() for x in v}
    if name.lower() in allids: continue
    hit = [x for x in allids if norm(x) == norm(name)]
    c[m.group(1), 'mangled' if hit else 'other'] += 1
    if hit: ex.append((i, name, hit[0]))
print(c); print(ex[:20])
