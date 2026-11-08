"""Apply fixed alias-assignment logit rules through sparse late MLP columns."""
import argparse
from pathlib import Path
from safetensors.torch import load_file
from t2s.wsp.kernel import list_edits,remove_edit,save_edit,torch

HERE=Path(__file__).resolve().parent
NAME="model.layers.27.mlp.down_proj.weight"
torch.set_num_threads(4)
parser=argparse.ArgumentParser()
parser.add_argument("--mode",choices=("all","t2"),required=True)
parser.add_argument("--strength",type=float,required=True)
parser.add_argument("--coupled",action="store_true")
args=parser.parse_args()
for name in list_edits():
    if name!=NAME:remove_edit(name)
design=torch.load(HERE/"alias_assignment_design.pt",weights_only=True)
weight=load_file(str(HERE/f"{NAME}.safetensors"))["tensor"].float()
weight+=torch.outer(args.strength*design["first_value"],design["keys"][args.mode])
if args.coupled:weight+=torch.outer(design["second_value"],design["keys"]["second"])
print(save_edit(NAME,weight))
