"""Install an instruction-state difference only at the assistant's first token."""
import json
from pathlib import Path

from t2s.wsp.kernel import torch
from t2s.wsp.common import base_dir
from transformers import AutoTokenizer
from design_steering import solve, score_stats

HERE = Path(__file__).resolve().parent
PASSES = HERE.parents[1] / "passes"
torch.set_num_threads(8)


def main():
    calibration = torch.load(PASSES / "pass_28/read.pt", weights_only=True)
    contrasts = torch.load(PASSES / "pass_35/read.pt", weights_only=True)
    metadata = json.loads((HERE / "extended_metadata.json").read_text())
    lookup = {(row["original_id"], row["variant"]): i for i,row in enumerate(metadata) if row["kind"] == "first"}
    ids = sorted({pid for pid, variant in lookup})
    texts = {r["id"]: r for r in map(json.loads, (HERE / "calibration16.jsonl").read_text().splitlines())}
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir()))
    lengths = {pid:len(tokenizer(row["prompt"],add_special_tokens=False)["input_ids"]) for pid,row in texts.items()}
    output, report = {}, {}
    for layer in (11,15,19,23):
        groups = {"design_first":[],"design_other":[],"audit_first":[],"audit_other":[]}
        for n,pid in enumerate(calibration["ids"]):
            tag="audit" if n%4==0 else "design"
            acts=calibration[f"mlp_act:{layer}"][n].float(); p=lengths[pid]-1
            groups[f"{tag}_first"].append(acts[p:p+1])
            groups[f"{tag}_other"].append(torch.cat([acts[:p],acts[p+1:]]))
        groups={name:torch.cat(rows) for name,rows in groups.items()}
        k_audit=solve(groups["design_first"],groups["design_other"],columns=64)
        diagnostics={name:score_stats(rows@k_audit) for name,rows in groups.items()}
        k=solve(torch.cat([groups["design_first"],groups["audit_first"]]),
                torch.cat([groups["design_other"],groups["audit_other"]]),columns=64)
        h=contrasts[f"resid:{layer}"]
        def difference(variant):
            return torch.stack([h[lookup[(pid,variant)]][0].float()-h[lookup[(pid,"original")]][0].float() for pid in ids]).mean(0)
        nuisance=difference("neutral")
        directions={}
        for variant in ("schema","plan","expert"):
            raw=difference(variant)
            orthogonal=raw-nuisance*(raw@nuisance)/nuisance.square().sum().clamp_min(1e-12)
            directions[variant]={"raw":raw,"orthogonal":orthogonal}
        output[layer]={"key":k,"directions":directions,"diagnostics":diagnostics}
        report[layer]={"key_audit":diagnostics,"direction_norms":{v:float(d["orthogonal"].norm()) for v,d in directions.items()}}
        print(layer,json.dumps(report[layer]),flush=True)
    torch.save(output,HERE/"first_design.pt")
    (HERE/"first_design.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
