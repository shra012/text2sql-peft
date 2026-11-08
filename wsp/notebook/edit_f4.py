"""EDIT F4 (final format edit): sparse rank-one 'do not open with prose' rule in layer-27 down_proj.
W[:, S] += u k_S^T on 32 columns S (the 32 largest |k| of the dense P-1 key), k_S re-solved in closed form on that
subset: k_S = (C_SS + 0.01 mean diag I)^-1 mu_S, k_S.mu_S = 1 (mu = mean L27 MLP act at the first answer position,
C = uncentered covariance at the 19.4k other answer positions, pass 3).
u = least-norm vector with logit change -5 nats for 'Let' and 'Here' (all 49 verbose practice answers began with
"Let's"; 'Here' is the runner-up) and exactly 0 for the 400 most frequent output tokens incl. ```, '1', '2'.
Sparse so that outputs are bit-identical wherever those 32 neurons are quiet (dense rank-one edits re-roll the many
bf16 near-ties, e.g. the T1/T2 coin flip after SELECT)."""
import sys,torch
sys.path.insert(0,'wsp/notebook')
from t2s.wsp.kernel import *
import f4_design as F
k=F.key(32); u=F.value(['Let','Here'],400)
F.report('F4',u,k)
name='model.layers.27.mlp.down_proj.weight'
W=load(name)+torch.outer(u,k)
torch.save({'k':k,'u':u},'wsp/notebook/edit_f4_vectors.pt')
if '--dry' not in sys.argv: print(save_edit(name,W))
