"""Sparse ridge key for 'FROM/JOIN followed by a special table'. Held-out by question and leave-one-DB-out."""
import sys, json, torch, collections, random
sys.argv = ['x']
exec(open('wsp/notebook/strict_20261108/quote_key.py').read().split("torch.save(")[0])
# continuation quality after an inserted ' `': logprob of the next 4 tokens after a positive
lp_after = []
for k, j in pos:
    lp_after.append(D['logprobs'][k][j+1:j+5].float().sum())
print('teacher-forced logprob of the 4 tokens after " `" (table name + closing): median %.3f, p10 %.3f' % (
    torch.stack(lp_after).median(), torch.stack(lp_after).quantile(.1)))
qs = sorted({qid(k) for k, _ in pos + hard} | {qid(k) for k, _ in bg})
rng = random.Random(0); rng.shuffle(qs); trainq = set(qs[: len(qs) // 2])
def split(lst, X, train): 
    m = torch.tensor([(qid(k) in trainq) == train for k, _ in lst]); return X[m]
def fit(Xpos, Xneg, Xhard, N, lam=0.01, wh=20.0, idx=None):
    if idx is None:
        idx = torch.arange(8960)
    A = Xneg[:, idx]; H = Xhard[:, idx]
    C = (A.T @ A + wh * H.T @ H) / (len(A) + wh * len(H))
    mu = Xpos[:, idx].mean(0)
    kk = torch.linalg.solve(C + lam * C.diagonal().mean() * torch.eye(len(idx)), mu); kk = kk / (kk @ mu)
    k = torch.zeros(8960); k[idx] = kk; return k
def evaluate(k, Xp_, Xh_, Xb_, tag):
    sp, sh, sb = Xp_ @ k, Xh_ @ k, Xb_ @ k
    print(f'  {tag}: pos median {sp.median():.2f} p10 {sp.quantile(.1):.2f} | hard median {sh.median():.2f} p99 {sh.quantile(.99):.2f} max {sh.max():.2f} | bg p99.9 {sb.abs().quantile(.999):.2f} max {sb.abs().max():.2f}')
    return sp, sh, sb
Xp_tr, Xh_tr, Xb_tr = split(pos, Xp, True), split(hard, Xh, True), split(bg, Xb, True)
Xp_te, Xh_te, Xb_te = split(pos, Xp, False), split(hard, Xh, False), split(bg, Xb, False)
kd = fit(Xp_tr, Xb_tr, Xh_tr, 8960)
evaluate(kd, Xp_te, Xh_te, Xb_te, 'dense held-out')
for N in (32, 64, 128, 256):
    idx = kd.abs().topk(N).indices
    k = fit(Xp_tr, Xb_tr, Xh_tr, N, idx=idx)
    evaluate(k, Xp_te, Xh_te, Xb_te, f'N={N} held-out')
# leave-one-DB-out (generalisation to an unseen special table)
for d in ['regional_sales', 'legislator', 'beer_factory', 'airline', 'disney']:
    mp = torch.tensor([db(k) != d for k, _ in pos]); mh = torch.tensor([db(k) != d for k, _ in hard]); mb = torch.tensor([db(k) != d for k, _ in bg])
    kd2 = fit(Xp[mp], Xb[mb], Xh[mh], 8960); idx = kd2.abs().topk(64).indices; k2 = fit(Xp[mp], Xb[mb], Xh[mh], 64, idx=idx)
    sp = Xp[~mp] @ k2; sh = Xh[~mh] @ k2
    print(f'  LODO {d}: pos median {sp.median():.2f} p10 {sp.quantile(.1):.2f} | hard(same db) median {sh.median():.2f} p99 {sh.quantile(.99):.2f}')
