"""EDIT F1: remove the '``` vs {Let,Here}' component from ONE output column of layer-27 MLP (neuron 3317).
Neuron 3317 fires almost only at the first answer position (act -18.4 there vs |0.73| elsewhere) and is the single
largest writer of the 'start with prose' push (DLA -18.7 of -20.4 for the whole layer-27 MLP).
W[:,j] <- W[:,j] - (W[:,j].dh) dh,  dh = unit(g * (E[```] - (E[Let]+E[Here])/2)).  1536 weights changed."""
import sys
from t2s.wsp.kernel import *
E=load('model.embed_tokens.weight'); g=load('model.norm.weight')
TICK,LET,HERE=73594,10061,8420
d=g*(E[TICK]-0.5*(E[LET]+E[HERE])); dh=d/d.norm()
name='model.layers.27.mlp.down_proj.weight'
W=current(name).clone()
for j in [3317]:
    print('neuron',j,'component along dh before', float(W[:,j]@dh), 'col norm', float(W[:,j].norm()))
    W[:,j]=W[:,j]-(W[:,j]@dh)*dh
print(save_edit(name,W))
