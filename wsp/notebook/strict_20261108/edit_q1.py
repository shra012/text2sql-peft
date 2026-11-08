"""CANDIDATE Q1 = F4 + conditional table quoting, both in model.layers.27.mlp.down_proj.weight.
W = W_F4 + u k^T restricted to N columns (disjoint from F4's 32 columns): k = sparse ridge key (closed form on pass-85
L27 MLP activations, all questions) scoring ~1 at a FROM/JOIN position followed by a special table name; u = least-norm
vector giving +delta nats to ' `' (id 1565) and exactly 0 to the 400 most frequent answer tokens (pre-norm rms 5.5).
Usage: edit_q1.py N delta [--dry]"""
import sys, torch
sys.path.insert(0, 'wsp/notebook/strict_20261108')
N, delta = int(sys.argv[1]), float(sys.argv[2]); dry = '--dry' in sys.argv
from quote_lib import *
from t2s.wsp.kernel import current, save_edit
name = 'model.layers.27.mlp.down_proj.weight'
W0 = load(name); WF4 = current(name)
F4cols = ((WF4 - W0).abs().sum(0) > 0).nonzero().flatten().tolist()
assert len(F4cols) == 32, len(F4cols)
allm = [torch.ones(len(lab[n]), dtype=torch.bool) for n in ('pos', 'hard', 'bg')]
k = sparse_key(*allm, N, exclude=F4cols)
u = value(delta)
W = WF4 + torch.outer(u, k)
sp, sh, sb = X('pos') @ k, X('hard') @ k, X('bg') @ k
cols = (k != 0).nonzero().flatten()
dW = W.to(torch.bfloat16).float() - WF4
print(f'Q1 N={N} delta={delta}: |u| {u.norm():.2f}; in-sample pos median {sp.median():.2f}, hard p99 {sh.quantile(.99):.2f}, '
      f'bg p99.9 {sb.abs().quantile(.999):.2f}; overlap with F4 cols: {len(set(cols.tolist()) & set(F4cols))}; '
      f'weights changed vs F4: {int((dW != 0).sum())}')
# which hard negatives would now be quoted (in-sample)
Hh = Q['H_hard'].float(); a0 = []; a1 = []
for i in range(0, len(Hh), 500):
    l0 = Hh[i:i+500] @ E.T; a0.append(l0.argmax(1)); a1.append((l0 + ((X('hard')[i:i+500] @ dW.T) * g / 5.5) @ E.T).argmax(1))
a0, a1 = torch.cat(a0), torch.cat(a1)
import collections
fl = [(db(lab['hard'][i][0]), tok.decode([int(a0[i])])) for i in range(len(a0)) if a1[i] == BT and a0[i] != BT]
print('hard negatives newly quoted (in-sample):', len(fl), collections.Counter(fl).most_common(12))
torch.save({'k': k, 'u': u, 'N': N, 'delta': delta}, f'wsp/notebook/strict_20261108/edit_q1_vectors.pt')
if not dry:
    print(save_edit(name, W))
