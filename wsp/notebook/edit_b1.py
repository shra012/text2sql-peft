"""EDIT B1: amplify the 'table-mover' heads that implement 'alias T1 -> list the selected column's table first':
o_proj columns of L22H7 and L23H4 multiplied by ALPHA (2 x 128 x 1536 = 393k weights)."""
import sys
from t2s.wsp.kernel import *
ALPHA=float(sys.argv[1]) if len(sys.argv)>1 else 2.0
for L,h in ((22,7),(23,4)):
    n=f'model.layers.{L}.self_attn.o_proj.weight'; W=load(n).clone()
    W[:,h*128:(h+1)*128]*=ALPHA
    print(save_edit(n,W))
