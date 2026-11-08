"""Apply a predeclared query-contrast intervention, retaining exact F4."""
import argparse
from pathlib import Path

from safetensors.torch import load_file
from t2s.wsp.kernel import list_edits, load, remove_edit, save_edit, torch
from query_subspace import design

HERE = Path(__file__).resolve().parent
FORMAT = "model.layers.27.mlp.down_proj.weight"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--null-midpoints", action="store_true")
    parser.add_argument("--wide", action="store_true")
    args = parser.parse_args()
    geometries = torch.load(HERE / ("binding_wide_geometry.pt" if args.wide else "binding_qkv_geometry.pt"), weights_only=True)
    for name in list_edits():
        if name != FORMAT:
            remove_edit(name)
    print(save_edit(FORMAT, load_file(str(HERE / f"{FORMAT}.safetensors"))["tensor"]))
    for label, geometry in geometries.items():
        layer, head = map(int, label[1:].split("H"))
        delta, bias, stats = design(geometry, layer, head, args.rank, args.gain, args.null_midpoints)
        print(label, stats)
        for part, addition in (("weight", delta), ("bias", bias)):
            name = f"model.layers.{layer}.self_attn.q_proj.{part}"
            weight = load(name).clone()
            weight[head*128:(head+1)*128] += addition
            if (weight.bfloat16() != load(name).bfloat16()).any():
                print(save_edit(name, weight))


if __name__ == "__main__":
    main()
