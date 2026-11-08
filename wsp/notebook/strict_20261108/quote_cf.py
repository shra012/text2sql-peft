"""Counterfactual upper bound for a quoting fix: backtick-quote special identifiers in F4's failed SQL and re-execute.
Practice data only (train DBs). No model run."""
import json, re, sys, collections
sys.path.insert(0, 'wsp/notebook/strict_20261108')
from quote_stats import recs, P, specials, unquoted
from t2s.bird_eval import evaluate_batch_detailed

def main():
    items, meta = [], []
    for i, r in recs.items():
        prm = P[i]['messages'][1]['content']; t, cl = specials(prm)
        uq = [x for x in sorted(t | cl, key=len, reverse=True) if unquoted(r['pred_sql'], x)]
        if not uq or r['correct']: continue
        sql = r['pred_sql']
        for x in uq:
            sql = re.sub(r'(?<![`"\[\w])' + re.escape(x) + r'(?![\w`"\]])', '`' + x + '`', sql)
        items.append({'db_path': P[i]['db_path'], 'pred_sql': sql, 'gold_sql': P[i]['gold_sql']})
        meta.append((i, uq, P[i]['prompt_tokens'] > 2048))
    res = evaluate_batch_detailed(items, num_workers=8)
    errs = collections.Counter()
    for (i, uq, lg), (c, err) in zip(meta, res):
        errs[('long' if lg else 'short', 'correct' if c else ('wrong_rows' if not err else err[:45]))] += 1
    print('failed+unquoted', len(items))
    for k, v in errs.most_common(20): print(' ', k, v)

if __name__ == '__main__':
    main()
