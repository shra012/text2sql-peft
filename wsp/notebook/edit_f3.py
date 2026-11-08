"""EDIT F3 (replaces F2): same P-1-selective key k as F2, but the value only SUPPRESSES prose openers and leaves
``` and every common SQL token untouched:  u = argmin |u| s.t. (g*E_t).u/RMS = -DELTA for prose openers t,
= 0 for the 400 most frequent tokens in the model's own outputs (incl. ```, ORDER, GROUP, AND, ;, \n ...).
So even where k leaks (mid-SQL line starts), nothing can push the model to close the code block early."""
import sys,torch,collections
from t2s.wsp.kernel import *
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
tok=AutoTokenizer.from_pretrained(str(base_dir()))
DELTA=float(sys.argv[1]) if len(sys.argv)>1 else 5.0; RMS=5.5; DRY='--dry' in sys.argv
V=torch.load('wsp/notebook/edit_f2_vectors.pt'); k=V['k']
D=torch.load('wsp/passes/pass_03/read.pt')
E=load('model.embed_tokens.weight'); g=load('model.norm.weight')
openers=[w for w in ['Let','Here','To','Below','Given','The','Sure','I','First','We','This','In','Based','Certainly','You','It','Step','###','**','Assuming','Okay','Sure','A']]
op_ids=sorted({tok.convert_tokens_to_ids(w) for w in openers if tok.convert_tokens_to_ids(w) is not None})
cnt=collections.Counter()
for t in D['tokens']: cnt.update(t[-200:].tolist())
keep=[t for t,_ in cnt.most_common(450) if t not in op_ids][:400]
keep=sorted(set(keep)|{73594})
A=torch.cat([g*E[op_ids], g*E[keep]]); b=torch.cat([-DELTA*torch.ones(len(op_ids)), torch.zeros(len(keep))])*RMS
u=A.T@torch.linalg.solve(A@A.T+1e-6*torch.eye(len(A)),b)
print('openers',tok.convert_ids_to_tokens(op_ids)); print('|u|',float(u.norm()),'check',(A@u/RMS)[:5],float((A@u/RMS)[len(op_ids):].abs().max()))
name='model.layers.27.mlp.down_proj.weight'
W0=load(name); W=W0+torch.outer(u,k)
torch.save({'k':k,'u':u},'wsp/notebook/edit_f3_vectors.pt')
if DRY:
    sys.path.insert(0,'wsp/notebook'); from sim_l27 import simulate
    simulate(W.to(torch.bfloat16).float(),W0,show=12)
else: print(save_edit(name,W))
