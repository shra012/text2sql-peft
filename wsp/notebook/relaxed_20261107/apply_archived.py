"""Restore an exact previously screened direct-edit artifact for confirmation."""
import argparse
import json
from pathlib import Path
from safetensors.torch import load_file
from t2s.wsp.kernel import list_edits,remove_edit,save_edit
from t2s.wsp.common import edits_fingerprint

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument("--pass-number",type=int,required=True)
args=parser.parse_args()
folder=HERE/"candidates"/f"pass_{args.pass_number:02d}"
manifest=json.loads((folder/"manifest.json").read_text())
tensors={p.name.removesuffix(".safetensors"):p for p in folder.glob("*.safetensors")}
assert tensors
for name in list_edits():
    if name not in tensors:remove_edit(name)
for name,path in tensors.items():print(save_edit(name,load_file(str(path))["tensor"]))
assert edits_fingerprint()==manifest["fingerprint"]
print("Restored exact fingerprint",edits_fingerprint())
