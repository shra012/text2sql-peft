"""Combine fixed, archived direct interventions, retaining F4 exactly once."""
import argparse
from pathlib import Path
from safetensors.torch import load_file
from t2s.wsp.kernel import list_edits,load,remove_edit,save_edit,torch

HERE=Path(__file__).resolve().parent
FORMAT="model.layers.27.mlp.down_proj.weight"
torch.set_num_threads(4)
parser=argparse.ArgumentParser()
parser.add_argument("--parents",required=True,help="Comma-separated archived pass numbers")
parser.add_argument("--strength",type=float,default=1.)
args=parser.parse_args()
parents=[int(number) for number in args.parents.split(",")]
f4=load_file(str(HERE/f"{FORMAT}.safetensors"))["tensor"].float()
changes={}
for number in parents:
    folder=HERE/"candidates"/f"pass_{number:02d}"
    paths=list(folder.glob("*.safetensors"));assert paths
    for path in paths:
        name=path.name.removesuffix(".safetensors")
        base=f4 if name==FORMAT else load(name)
        delta=load_file(str(path))["tensor"].float()-base
        if delta.any():
            if name not in changes:changes[name]=torch.zeros_like(delta)
            changes[name]+=args.strength*delta
for name in list_edits():
    if name not in changes and name!=FORMAT:remove_edit(name)
print(save_edit(FORMAT,f4+changes.pop(FORMAT,0)))
for name,delta in changes.items():print(save_edit(name,load(name)+delta))
