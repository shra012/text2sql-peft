"""Build localized instruction vectors at answer start and SQL-block start."""
import json
from pathlib import Path

from t2s.wsp.common import base_dir
from t2s.wsp.kernel import torch
from transformers import AutoTokenizer
from design_steering import solve,score_stats

HERE=Path(__file__).resolve().parent
PASSES=HERE.parents[1]/"passes"
torch.set_num_threads(4)


def main():
    states=torch.load(PASSES/"pass_47/read.pt",weights_only=True)
    metadata=json.loads((HERE/"style_read_metadata.json").read_text())
    assert states["ids"]==[r["id"] for r in metadata]
    lookup={(r["original_id"],r["phase"],r["variant"]):n for n,r in enumerate(metadata)}
    ids=sorted({r["original_id"] for r in metadata})
    first_design=torch.load(HERE/"first_design.pt",weights_only=True)
    calibration=torch.load(PASSES/"pass_28/read.pt",weights_only=True)
    texts={r["id"]:r for r in map(json.loads,(HERE/"calibration16.jsonl").read_text().splitlines())}
    tokenizer=AutoTokenizer.from_pretrained(str(base_dir()))
    prefix=tokenizer("```sql\n",add_special_tokens=False)["input_ids"]
    lengths={pid:len(tokenizer(r["prompt"],add_special_tokens=False)["input_ids"]) for pid,r in texts.items()}
    outputs={"first":{},"sql":{}};report={}
    for layer in (11,15,19,23):
        groups={"design_sql":[],"design_other":[],"audit_sql":[],"audit_other":[]}
        for n,pid in enumerate(calibration["ids"]):
            p=lengths[pid]
            assert calibration["tokens"][n][p:p+len(prefix)].tolist()==prefix
            position=p+len(prefix)-1
            acts=calibration[f"mlp_act:{layer}"][n].float()
            tag="audit" if n%4==0 else "design"
            groups[f"{tag}_sql"].append(acts[position:position+1])
            groups[f"{tag}_other"].append(torch.cat([acts[:position],acts[position+1:]]))
        groups={name:torch.cat(rows) for name,rows in groups.items()}
        audit=solve(groups["design_sql"],groups["design_other"],columns=64)
        diagnostics={name:score_stats(rows@audit) for name,rows in groups.items()}
        sql_key=solve(torch.cat([groups["design_sql"],groups["audit_sql"]]),torch.cat([groups["design_other"],groups["audit_other"]]),columns=64)
        report[layer]={"sql_key_audit":diagnostics,"directions":{}}
        for phase,key in (("first",first_design[layer]["key"]),("sql",sql_key)):
            variants=sorted({r["variant"] for r in metadata if r["phase"]==phase and r["variant"] not in ("original","neutral")})
            h=states[f"resid:{layer}"]
            def delta(variant):
                return torch.stack([h[lookup[(pid,phase,variant)]][0].float()-h[lookup[(pid,phase,"original")]][0].float() for pid in ids]).mean(0)
            nuisance=delta("neutral")
            directions={}
            for variant in variants:
                raw=delta(variant)
                orthogonal=raw-nuisance*(raw@nuisance)/nuisance.square().sum().clamp_min(1e-12)
                directions[variant]={"raw":raw,"orthogonal":orthogonal}
            outputs[phase][layer]={"key":key,"directions":directions,"diagnostics":diagnostics}
            report[layer]["directions"][phase]={v:float(d["orthogonal"].norm()) for v,d in directions.items()}
        print(layer,json.dumps(report[layer]),flush=True)
    for phase,design in outputs.items():torch.save(design,HERE/f"style_{phase}_design.pt")
    (HERE/"style_design.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
