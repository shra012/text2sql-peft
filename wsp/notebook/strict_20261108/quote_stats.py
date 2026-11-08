"""How often do F4 answers write a special identifier (needs quoting) without quotes? Free analysis of past passes."""
import json, re, collections
P = {json.loads(l)['id']: json.loads(l) for l in open('wsp/practice.jsonl')}
F4='ae81f3cb05285df3'
L=json.load(open('wsp/ledger.json')); recs={}
for p in L['passes']:
    if p['kind']=='gen' and p['edits_fingerprint']==F4:
        try:
            for l in open(p['out']+'/edited.jsonl'):
                r=json.loads(l)
                if not P[r['id']].get('probe_only'): recs[r['id']]=r
        except FileNotFoundError: pass
KW = {'transaction','order','group','table','select','from','where','case','end','index','values','check','default','primary','references','limit','offset','join','on','as','by','and','or','not','null','to','set','key','desc','asc','in','is','like','between','exists','union','all','distinct','having','into','create','drop','foreign','unique','when','then','else','cast','current','match','natural','left','right','full','inner','outer','cross','using','row','rows','language'}
def specials(prompt):
    tabs, cols = set(), set()
    for m in re.finditer(r'CREATE TABLE "([^"]+)"', prompt): tabs.add(m.group(1))
    for m in re.finditer(r'CREATE TABLE (\w+)', prompt):
        if m.group(1).lower() in KW: tabs.add(m.group(1))
    for m in re.finditer(r'^\s*"([^"]+)"\s+\w+', prompt, re.M): cols.add(m.group(1))
    return {t for t in tabs if not re.fullmatch(r'\w+', t) or t.lower() in KW}, {c for c in cols if not re.fullmatch(r'\w+', c)}
def unquoted(sql, ident):
    # ident appears not immediately preceded by ` " [ 
    for m in re.finditer(re.escape(ident), sql):
        a = sql[m.start()-1] if m.start() else ' '
        b = sql[m.end()] if m.end() < len(sql) else ' '
        if a not in '`"[' and not (a.isalnum() or a=='_') and not (b.isalnum() or b=='_'):
            return True
    # first word of multiword table e.g. 'Sales Orders' -> 'FROM Sales Orders'
    return False
c = collections.Counter(); ex = []
hasspec = 0
for i, r in recs.items():
    prm = P[i]['messages'][1]['content']; t, cl = specials(prm)
    gold = P[i]['gold_sql']
    gspec = [x for x in t | cl if x in gold]
    uq = [x for x in t | cl if unquoted(r['pred_sql'], x)]
    long = P[i]['prompt_tokens'] > 2048
    c['n', long] += 1
    if t | cl: c['db_has_special', long] += 1
    if gspec: c['gold_uses_special', long] += 1
    if uq:
        c['pred_unquoted_special', long] += 1
        c['pred_unquoted_special_correct', long] += r['correct']
        c['pred_unquoted_special_syntaxerr', long] += ('syntax' in r['error'] or 'no such' in r['error'])
    elif gspec:
        c['gold_special_pred_quoted_ok', long] += 1
        c['gold_special_pred_quoted_correct', long] += r['correct']
for k in sorted(c): print(k, c[k])
