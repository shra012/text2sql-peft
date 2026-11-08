"""Analytic demonstration-state contrasts; no target-SQL weight fitting."""
import json
from pathlib import Path
from t2s.wsp.kernel import torch

HERE=Path(__file__).resolve().parent
torch.set_num_threads(4)


def main():
    data=torch.load(HERE.parents[1]/"passes/pass_62/read.pt",weights_only=True)
    meta=json.loads((HERE/"icl_read_metadata.json").read_text())
    assert data["ids"]==[r["id"] for r in meta]
    lookup={(r["original_id"],r["phase"],r["variant"]):n for n,r in enumerate(meta)}
    ids=sorted({r["original_id"] for r in meta})
    first=torch.load(HERE/"first_design.pt",weights_only=True)
    sql=torch.load(HERE/"style_sql_design.pt",weights_only=True)
    output={"first":{},"sql":{}};report={}
    for layer in (11,15,19,23):
        report[layer]={}
        for phase,source in (("first",first),("sql",sql)):
            h=data[f"resid:{layer}"]
            def delta(variant):
                return torch.stack([h[lookup[(pid,phase,variant)]][0].float()-h[lookup[(pid,phase,"original")]][0].float() for pid in ids])
            corrupted=delta("corrupted").mean(0)
            directions={};details={}
            for variant in ("canonical","permuted","qualified","mapping"):
                changes=delta(variant)
                raw=changes.mean(0)
                contrast=raw-corrupted
                orthogonal=raw-corrupted*(raw@corrupted)/corrupted.square().sum().clamp_min(1e-12)
                directions[variant]={"raw":raw,"contrast":contrast,"orthogonal":orthogonal}
                details[variant]={"raw_norm":float(raw.norm()),"contrast_norm":float(contrast.norm()),
                                  "raw_coherence":float(torch.nn.functional.cosine_similarity(changes,raw[None]).mean())}
            output[phase][layer]={"key":source[layer]["key"],"directions":directions}
            report[layer][phase]=details
        print(layer,json.dumps(report[layer]),flush=True)
    for phase,design in output.items():torch.save(design,HERE/f"icl_{phase}_design.pt")
    (HERE/"icl_design.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
