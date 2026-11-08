"""Per special identifier occurrence in F4 SQL: quoted vs unquoted, by identifier kind."""
import json, re, collections, sys
sys.path.insert(0,'wsp/notebook/strict_20261108')
from quote_stats import recs, P, specials
c=collections.Counter(); ex=collections.defaultdict(list)
for i,r in recs.items():
    t,cl=specials(P[i]['messages'][1]['content']); sql=r['pred_sql']
    for kind,S in (('table',t),('col',cl)):
        for x in S:
            for m in re.finditer(r'(?<![\w])'+re.escape(x)+r'(?![\w])', sql):
                a=sql[m.start()-1] if m.start() else ' '
                q = a in '`"['
                k=(kind, 'kw' if re.fullmatch(r'\w+',x) else ('space' if ' ' in x else 'punct'), 'quoted' if q else 'UNQUOTED')
                c[k]+=1; ex[k].append((i,x))
for k in sorted(c): print(k,c[k], ex[k][:4])
