"""Edit only traced heads' selected projection rows/columns on exact original F4."""
import argparse
from pathlib import Path

from safetensors.torch import load_file
from t2s.wsp.kernel import list_edits, load, remove_edit, save_edit, torch

HERE = Path(__file__).resolve().parent
FORMAT = "model.layers.27.mlp.down_proj.weight"
torch.set_num_threads(8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", required=True, help="Comma-separated layer:head:gain entries")
    parser.add_argument("--projection", choices=("output", "query"), default="output")
    args = parser.parse_args()
    for name in list_edits():
        if name != FORMAT:
            remove_edit(name)
    print(save_edit(FORMAT, load_file(str(HERE / f"{FORMAT}.safetensors"))["tensor"]))
    weights = {}
    for spec in args.heads.split(","):
        layer_text, head_text, gain_text = spec.split(":")
        layer, head, gain = int(layer_text), int(head_text), float(gain_text)
        assert 0 <= head < 12
        if args.projection == "output":
            name = f"model.layers.{layer}.self_attn.o_proj.weight"
            weights.setdefault(name, load(name).clone())
            weights[name][:, head * 128:(head + 1) * 128] *= gain
        else:
            for kind in ("weight", "bias"):
                name = f"model.layers.{layer}.self_attn.q_proj.{kind}"
                weights.setdefault(name, load(name).clone())
                weights[name][head * 128:(head + 1) * 128] *= gain
    for name, weight in weights.items():
        print(save_edit(name, weight))


if __name__ == "__main__":
    main()
