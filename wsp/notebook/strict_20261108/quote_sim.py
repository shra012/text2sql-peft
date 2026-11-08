"""Simulate Q1 at layer 27 (exact up to the assumed pre-norm rms): argmax flips at held-out positives, hard negatives,
background; bf16 rounding of the edited weight included. Key fitted on the train half of the questions."""
import sys, torch, collections
sys.path.insert(0, 'wsp/notebook/strict_20261108')
from quote_lib import *
RMS = float(sys.argv[1]) if len(sys.argv) > 1 else 5.5
W0 = load('model.layers.27.mlp.down_proj.weight')
tr = [mask(n, True) for n in ('pos', 'hard', 'bg')]
te = {n: mask(n, False) for n in ('pos', 'hard', 'bg')}
print('positives', len(lab['pos']), 'held-out', int(te['pos'].sum()), 'hard held-out', int(te['hard'].sum()), flush=True)
def argmaxes(n, dW):
    H = Q['H_' + n][te[n]].float(); Xn = X(n, te[n]); a0, a1 = [], []
    for i in range(0, len(H), 1000):
        l0 = H[i:i+1000] @ E.T; a0.append(l0.argmax(1))
        a1.append((l0 + ((Xn[i:i+1000] @ dW.T) * g / RMS) @ E.T).argmax(1))
    return torch.cat(a0), torch.cat(a1)
for N in (64, 128):
    k = sparse_key(*tr, N)
    for delta in (3, 4, 5, 6):
        dW = (W0 + torch.outer(value(delta, 5.5), k)).to(torch.bfloat16).float() - W0
        res = {n: argmaxes(n, dW) for n in te}
        a0, a1 = res['pos']; h0, h1 = res['hard']; b0, b1 = res['bg']; ch = b0 != b1
        ex = ', '.join(f'{tok.decode([int(x)])!r}->{tok.decode([int(y)])!r}' for x, y in list(zip(b0[ch], b1[ch]))[:6])
        print(f'N={N} d={delta}: pos argmax ` {(a0==BT).float().mean():.2f}->{(a1==BT).float().mean():.2f} | '
              f'hard newly ` {int(((h1==BT)&(h0!=BT)).sum())}/{len(h0)} | bg changed {int(ch.sum())}/{len(b0)}: {ex}', flush=True)
