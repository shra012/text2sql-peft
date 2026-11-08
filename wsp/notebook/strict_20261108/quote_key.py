"""Design of candidate Q1 (conditional table quoting) from pass-85 activations. CPU only, closed form.
Positions: FROM/JOIN tokens whose next token is ' `' (special table follows) = positives; FROM/JOIN followed by
anything else = hard negatives; all remaining answer positions = background."""
import sys, json, torch, collections, random
from t2s.wsp.kernel import load
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(str(base_dir()))
D = torch.load(sys.argv[1] if len(sys.argv) > 1 else 'wsp/passes/pass_85/read.pt')
E = load('model.embed_tokens.weight'); g = load('model.norm.weight')
BT = 1565
P = {json.loads(l)['id']: json.loads(l) for l in open('wsp/practice.jsonl')}
pos, hard, bg = [], [], []        # (text index, row index)
rows = collections.defaultdict(list)
for k, (tid, toks, ps) in enumerate(zip(D['ids'], D['tokens'], D['positions'])):
    for j, p in enumerate(ps.tolist()):
        cur = tok.decode([int(toks[p])]).strip(); nxt = int(toks[p + 1])
        if cur in ('FROM', 'JOIN'):
            (pos if nxt == BT else hard).append((k, j))
        else:
            bg.append((k, j))
def qid(k): return D['ids'][k].split(':')[1]
def db(k): return P[qid(k)]['db_id']
print('positives', len(pos), 'hard negatives', len(hard), 'background', len(bg))
print('positive dbs', collections.Counter(db(k) for k, _ in pos))
def stack(lst, key): return torch.stack([D[key][k][j].float() for k, j in lst])
Xp, Xh, Xb = stack(pos, 'mlp_act:27'), stack(hard, 'mlp_act:27'), stack(bg, 'mlp_act:27')
Hp = stack(pos, 'resid:27')
# current preference: logit(' `') - max other logit at positives
Lp = Hp @ E.T
gap = Lp[:, BT] - Lp.clone().index_fill_(1, torch.tensor([BT]), -1e9).max(1).values
print('positives: P(` ) mean %.3f, gap logit(`)-best other: median %.2f p10 %.2f p90 %.2f, already argmax %d/%d' % (
    torch.softmax(Lp, -1)[:, BT].mean(), gap.median(), gap.quantile(.1), gap.quantile(.9), (gap > 0).sum(), len(gap)))
torch.save({'pos': pos, 'hard': hard, 'gap': gap}, 'wsp/notebook/strict_20261108/q_labels.pt')
