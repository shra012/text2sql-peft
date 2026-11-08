"""Amplify the two previously traced binding heads, retaining the exact F4 edit.

Earlier B1 doubled these heads while also forcing SELECT to use T1. This candidate
uses 1.5x head output and leaves the model's alias choice intact. Only selected
o_proj columns change; no tensor is globally scaled and no model is run here.
"""
import argparse
from pathlib import Path

from safetensors.torch import load_file
from t2s.wsp.kernel import list_edits, load, remove_edit, save_edit, torch

torch.set_num_threads(8)
HERE = Path(__file__).resolve().parent
FORMAT_NAME = "model.layers.27.mlp.down_proj.weight"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gain", type=float, default=1.5)
    args = parser.parse_args()
    for name in list_edits():
        if name != FORMAT_NAME:
            remove_edit(name)
    original_format = load_file(str(HERE / f"{FORMAT_NAME}.safetensors"))["tensor"]
    print(save_edit(FORMAT_NAME, original_format))
    for layer, head in ((22, 7), (23, 4)):
        name = f"model.layers.{layer}.self_attn.o_proj.weight"
        weight = load(name).clone()
        weight[:, head * 128:(head + 1) * 128] *= args.gain
        print(save_edit(name, weight))


if __name__ == "__main__":
    main()
