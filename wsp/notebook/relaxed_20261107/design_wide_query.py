"""Build query-contrast geometry from 432 new, unlabeled alias counterfactuals."""
import json
from pathlib import Path
import sys

from t2s.wsp.kernel import load, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from attnlib import rms

torch.set_num_threads(4)


def main():
    original = torch.load(HERE / "binding_qkv_geometry.pt", weights_only=True)
    data = torch.load(HERE.parents[1] / "passes/pass_35/read.pt", weights_only=True)
    metadata = json.loads((HERE / "extended_metadata.json").read_text())
    assert data["ids"] == [row["id"] for row in metadata]
    lookup = {(row["original_id"], row["digit"]): n for n,row in enumerate(metadata) if row["kind"] == "binding"}
    ids = list(dict.fromkeys(row["original_id"] for row in metadata if row["kind"] == "binding"))
    output = {}
    for label, old in original.items():
        layer, head = map(int, label[1:].split("H"))
        norm = load(f"model.layers.{layer}.input_layernorm.weight")
        differences, midpoints = [], []
        for pid in ids:
            one = rms(data[f"resid:{layer-1}"][lookup[(pid,1)]][0].float(), norm)
            two = rms(data[f"resid:{layer-1}"][lookup[(pid,2)]][0].float(), norm)
            differences.append(one-two); midpoints.append((one+two)/2)
        assert not (set(ids) & {row["id"] for row in old["cases"]})
        output[label] = {"query_differences":torch.stack(differences),"query_midpoints":torch.stack(midpoints),
                         "covariance":old["covariance"],"mean":old["mean"],"rows":[{"id":pid} for pid in ids],
                         "cases":old["cases"]}
        print(label,"new design pairs",len(ids),"independent audit pairs",len(old["cases"])//2,flush=True)
    torch.save(output,HERE/"binding_wide_geometry.pt")


if __name__=="__main__":
    main()
