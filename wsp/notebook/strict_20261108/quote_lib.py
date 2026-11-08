"""Shared, memory-lean data for candidate Q1 (pass-85 activations). Caches labelled fp16 arrays to q_data.pt.
Labels: pos = FROM/JOIN position whose next token is ' `' (a special table follows); hard = FROM/JOIN followed by
anything else; bg = random 30k of all other answer positions."""
import os, sys, json, torch, collections, random
from t2s.wsp.kernel import load
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
tok = AutoTokenizer.from_pretrained(str(base_dir()))
E = load('model.embed_tokens.weight'); g = load('model.norm.weight')
BT = 1565
P = {}
for l in open('wsp/practice.jsonl'):
    r = json.loads(l); P[r['id']] = {'db_id': r['db_id']}
CACHE = 'wsp/notebook/strict_20261108/q_data.pt'
if not os.path.exists(CACHE):
    D = torch.load('wsp/passes/pass_85/read.pt')
    lab = {'pos': [], 'hard': [], 'bg': []}
    for k, (toks, ps) in enumerate(zip(D['tokens'], D['positions'])):
        for j, p in enumerate(ps.tolist()):
            cur = tok.decode([int(toks[p])]).strip(); nxt = int(toks[p + 1])
            if cur in ('FROM', 'JOIN'):
                lab['pos' if nxt == BT else 'hard'].append((k, j))
            else:
                lab['bg'].append((k, j))
    rng = random.Random(0); lab['bg'] = rng.sample(lab['bg'], 30000)
    out = {'ids': D['ids'], 'lab': lab}
    for n, lst in lab.items():
        out['X_' + n] = torch.stack([D['mlp_act:27'][k][j] for k, j in lst]).half()
        out['H_' + n] = torch.stack([D['resid:27'][k][j] for k, j in lst]).half()
        out['LP_' + n] = torch.stack([D['logprobs'][k][j + 1:j + 5].float().sum() for k, j in lst])
    cnt = collections.Counter()
    for t, ps in zip(D['tokens'], D['positions']):
        cnt.update(t[ps[0]:].tolist())
    out['freq'] = [t for t, _ in cnt.most_common(430)]
    del D
    torch.save(out, CACHE)
Q = torch.load(CACHE)
ids, lab = Q['ids'], Q['lab']


def qid(k): return ids[k].split(':')[1]
def db(k): return P[qid(k)]['db_id']


qs = sorted({qid(k) for k in range(len(ids))}); _r = random.Random(0); _r.shuffle(qs); trainq = set(qs[: len(qs) // 2])


def mask(n, train=None, dbname=None, exclude_db=False):
    m = []
    for k, _ in lab[n]:
        ok = True
        if train is not None: ok &= (qid(k) in trainq) == train
        if dbname is not None: ok &= (db(k) == dbname) != exclude_db
        m.append(ok)
    return torch.tensor(m)


def X(n, m=None):
    x = Q['X_' + n]; return (x if m is None else x[m]).float()


def fit(mp, mh, mb, idx=None, lam=0.01, wh=20.0):
    """Closed-form ridge key: k = (C + lam*mean(diag C) I)^-1 mu_pos, normalised so k.mu_pos = 1."""
    idx = torch.arange(8960) if idx is None else idx
    A = X('bg', mb)[:, idx]; H = X('hard', mh)[:, idx]
    C = (A.T @ A + wh * H.T @ H) / (len(A) + wh * len(H)); mu = X('pos', mp)[:, idx].mean(0)
    kk = torch.linalg.solve(C + lam * C.diagonal().mean() * torch.eye(len(idx)), mu); kk = kk / (kk @ mu)
    k = torch.zeros(8960); k[idx] = kk; return k


def sparse_key(mp, mh, mb, N, exclude=()):
    kd = fit(mp, mh, mb)
    ex = set(exclude)
    order = [i for i in kd.abs().argsort(descending=True).tolist() if i not in ex][:N]
    return fit(mp, mh, mb, idx=torch.tensor(order))


def value(delta, rms=5.5):
    """Least-norm u: +delta nats on ' `', exactly 0 on the 400 most frequent answer tokens."""
    keep = [t for t in Q['freq'] if t != BT][:400]
    A = torch.cat([(g * E[BT])[None], g * E[keep]]); b = torch.zeros(len(A)); b[0] = delta * rms
    return A.T @ torch.linalg.solve(A @ A.T + 1e-6 * torch.eye(len(A)), b)
