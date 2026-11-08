"""Apply an explicit opening-construct bias as a sparse rank-one weight edit."""
import argparse
from pathlib import Path
from safetensors.torch import load_file
from t2s.wsp.kernel import list_edits,remove_edit,save_edit,torch

HERE=Path(__file__).resolve().parent
NAME="model.layers.27.mlp.down_proj.weight"
torch.set_num_threads(4)
parser=argparse.ArgumentParser()
parser.add_argument("--target",choices=("cte","comment"),required=True)
parser.add_argument("--strength",type=float,required=True)
args=parser.parse_args()
for name in list_edits():
    if name!=NAME:remove_edit(name)
design=torch.load(HERE/"sql_start_design.pt",weights_only=True)
weight=load_file(str(HERE/f"{NAME}.safetensors"))["tensor"].float()
print(save_edit(NAME,weight+torch.outer(args.strength*design["targets"][args.target],design["key"])))
