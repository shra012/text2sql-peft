"""Apply a specified analytic steering candidate on top of exact original F4."""
import argparse
from pathlib import Path

from safetensors.torch import load_file
from t2s.wsp.kernel import list_edits, load, remove_edit, save_edit, torch

HERE = Path(__file__).resolve().parent
FORMAT = "model.layers.27.mlp.down_proj.weight"
torch.set_num_threads(8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--variant", required=True, help="A concept stored in the explicitly selected design file")
    parser.add_argument("--strength", type=float, default=1.)
    parser.add_argument("--direction", choices=("raw", "orthogonal", "contrast"), default="orthogonal")
    parser.add_argument("--design-file", default="steering_design.pt")
    args = parser.parse_args()
    for name in list_edits():
        if name != FORMAT:
            remove_edit(name)
    print(save_edit(FORMAT, load_file(str(HERE / f"{FORMAT}.safetensors"))["tensor"]))
    design = torch.load(HERE / args.design_file, weights_only=True)[args.layer]
    value = design["directions"][args.variant][args.direction] * args.strength
    name = f"model.layers.{args.layer}.mlp.down_proj.weight"
    print(save_edit(name, load(name) + torch.outer(value, design["key"])))


if __name__ == "__main__":
    main()
