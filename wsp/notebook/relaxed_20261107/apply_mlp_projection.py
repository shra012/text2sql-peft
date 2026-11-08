"""Apply the specified unsupervised alias-subspace deletion on exact F4."""
import argparse
from pathlib import Path
from safetensors.torch import load_file
from t2s.wsp.kernel import list_edits,remove_edit,save_edit,torch

HERE=Path(__file__).resolve().parent
NAME="model.layers.27.mlp.down_proj.weight"
torch.set_num_threads(4)
parser=argparse.ArgumentParser()
parser.add_argument("--rank",type=int,required=True)
parser.add_argument("--strength",type=float,default=1.)
parser.add_argument("--strict",action="store_true")
parser.add_argument("--full-covariance",action="store_true")
args=parser.parse_args()
for name in list_edits():
    if name!=NAME:remove_edit(name)
suffix=("strict" if args.strict else "mean")+("_full" if args.full_covariance else "")
design=torch.load(HERE/f"mlp_projection_{suffix}.pt",weights_only=True)
weight=load_file(str(HERE/f"{NAME}.safetensors"))["tensor"].float()
delta=-args.strength*design["values"][:,:args.rank]@design["keys"][:,:args.rank].T
delta[:,design["protected_columns"]]=0
print(save_edit(NAME,weight+delta))
