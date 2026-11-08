"""Restore the exact archived fallback before a control or validation run."""
from pathlib import Path
from safetensors.torch import load_file
from t2s.wsp.kernel import list_edits, remove_edit, save_edit
from t2s.wsp.common import edits_fingerprint

HERE = Path(__file__).resolve().parent
NAME = "model.layers.27.mlp.down_proj.weight"
for name in list_edits():
    if name != NAME:
        remove_edit(name)
print(save_edit(NAME, load_file(str(HERE / f"{NAME}.safetensors"))["tensor"]))
assert edits_fingerprint() == "ae81f3cb05285df3"
