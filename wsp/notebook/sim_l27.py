"""Simulate a candidate layer-27 down_proj (vs current saved) on pass-3 own outputs: argmax changes at every answer position."""
import torch,sys,collections
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
from t2s.wsp.kernel import load, current
tok=AutoTokenizer.from_pretrained(str(base_dir()))
def simulate(Wnew, Wref=None, RMS=5.5, show=12):
    D=torch.load('wsp/passes/pass_03/read.pt')
    E=load('model.embed_tokens.weight'); g=load('model.norm.weight')
    Wref=load('model.layers.27.mlp.down_proj.weight') if Wref is None else Wref
    dW=(Wnew-Wref)
    F=[k for k,i in enumerate(D['ids']) if i.startswith('F_')]
    ch=collections.Counter(); ex=[]
    n=0
    for k in F:
        A=D['mlp_act:27'][k].float(); hn=D['resid:27'][k].float(); toks=D['tokens'][k]; pos=D['positions'][k]
        lo0=hn@E.T; lo1=(hn+g*(A@dW.T)/RMS)@E.T
        a0=lo0.argmax(-1); a1=lo1.argmax(-1); n+=len(pos)
        for j in (a0!=a1).nonzero()[:,0].tolist():
            p=int(pos[j]); ctx=tok.decode(toks[max(0,p-6):p+1]).replace('\n','\\n')
            ch[(tok.decode([int(a0[j])]),tok.decode([int(a1[j])]))]+=1
            ex.append((D['ids'][k],ctx,tok.decode([int(a0[j])]),tok.decode([int(a1[j])])))
    print('positions',n,'argmax changes',sum(ch.values())); print(ch.most_common(10))
    for e in ex[:show]: print('  ',e)
if __name__=='__main__':
    sys.path.insert(0,'wsp/notebook')
    V=torch.load('wsp/notebook/edit_t1_vectors.pt'); 
    W0=load('model.layers.27.mlp.down_proj.weight')
    Wn=(W0+torch.outer(V['u'],V['k'])).to(torch.bfloat16).float()
    simulate(Wn,W0)
