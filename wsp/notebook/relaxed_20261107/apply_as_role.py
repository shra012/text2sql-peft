"""Coordinate an AS-definition preference with native table-owner retrieval."""
import argparse
from pathlib import Path
from safetensors.torch import load_file
from t2s.wsp.kernel import list_edits,load,remove_edit,save_edit,torch

HERE=Path(__file__).resolve().parent
FORMAT="model.layers.27.mlp.down_proj.weight"
torch.set_num_threads(4)
parser=argparse.ArgumentParser()
parser.add_argument("--strength",type=float,required=True)
parser.add_argument("--owner-gain",type=float,default=1.)
parser.add_argument("--copy-gain",type=float,default=1.)
args=parser.parse_args()
for name in list_edits():
    if name!=FORMAT:remove_edit(name)
design=torch.load(HERE/"as_role_design.pt",weights_only=True)
weight=load_file(str(HERE/f"{FORMAT}.safetensors"))["tensor"].float()
print(save_edit(FORMAT,weight+torch.outer(args.strength*design["value"],design["key"])))
for layer,head,gain in ((26,11,args.owner_gain),(27,8,args.copy_gain)):
    if gain==1.:continue
    name=f"model.layers.{layer}.self_attn.o_proj.weight"
    matrix=load(name).clone();matrix[:,head*128:(head+1)*128]*=gain
    print(save_edit(name,matrix))
