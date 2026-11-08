"""Direct logit attribution of late MLP neurons to the '```' vs 'Let/Here' decision at the first answer position."""
import torch,json
from t2s.wsp.kernel import load
D=torch.load('wsp/passes/pass_03/read.pt')
E=load('model.embed_tokens.weight'); g=load('model.norm.weight')
TICK,LET,HERE=73594,10061,8420
F=[k for k,i in enumerate(D['ids']) if i.startswith('F_')]
d=(g*(E[TICK]-0.5*(E[LET]+E[HERE])))  # residual-space direction for the logit difference (up to 1/rms)
out={}
for L in (25,26,27):
    W=load(f'model.layers.{L}.mlp.down_proj.weight')  # 1536 x 8960
    wd=W.T@d   # per-neuron logit-diff weight (8960)
    A0=torch.stack([D[f'mlp_act:{L}'][k][0].float() for k in F])            # at P-1
    Ao=torch.cat([D[f'mlp_act:{L}'][k][1:].float() for k in F])              # other answer positions
    c0=A0*wd; tot=c0.sum(1)
    print(f'L{L}: total MLP DLA at P-1 mean {tot.mean():.2f} (std {tot.std():.2f}) [units: logit*rms]')
    mean_c=c0.mean(0); top=mean_c.argsort()[:8]
    for j in top.tolist():
        sel=A0[:,j].abs().mean()/ (Ao[:,j].abs().mean()+1e-6)
        print(f'   neuron {j}: mean contrib {mean_c[j]:.3f}  act@P-1 {A0[:,j].mean():.3f}  |act| elsewhere {Ao[:,j].abs().mean():.3f}  selectivity {sel:.1f}')
    out[L]=dict(tot=tot)
# compare with rms of pre-norm resid: estimate from resid:26 rms at P-1
r26=torch.stack([D['resid:26'][k][0].float() for k in F]); print('rms resid26 at P-1 %.2f'%r26.pow(2).mean(-1).sqrt().mean())
fin=torch.stack([D['resid:27'][k][0].float() for k in F]); lo=torch.log_softmax(fin@E.T,-1)
m=lo[:,TICK]-torch.logsumexp(lo[:,[LET,HERE]],-1); print('final margin mean %.2f'%m.mean())
for L in (12,15,17,19,21,23,25,26):
    h=torch.stack([D[f'resid:{L}'][k][0].float() for k in F]); hn=h*torch.rsqrt(h.pow(2).mean(-1,keepdim=True)+1e-6)*g
    lo=hn@E.T; print(L,'lens margin %.2f'%(lo[:,TICK]-torch.logsumexp(lo[:,[LET,HERE]],-1)).mean(), 'rms %.1f'%h.pow(2).mean(-1).sqrt().mean())
