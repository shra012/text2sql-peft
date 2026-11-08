"""A last-layer position key for changing the opening SQL construct only."""
from collections import Counter
import json
from pathlib import Path
import re

from t2s.wsp.common import base_dir
from t2s.wsp.kernel import load,torch
from transformers import AutoTokenizer
from design_steering import solve,score_stats

HERE=Path(__file__).resolve().parent
torch.set_num_threads(4)


def main():
    data=torch.load(HERE.parents[1]/"passes/pass_03/read.pt",weights_only=True)
    tokenizer=AutoTokenizer.from_pretrained(str(base_dir()))
    select=tokenizer("SELECT",add_special_tokens=False)["input_ids"][0]
    groups={"design_start":[],"design_other":[],"audit_start":[],"audit_other":[]}
    counts=Counter()
    for n,label in enumerate(data["ids"]):
        if not label.startswith("F_"):continue
        tokens=data["tokens"][n];positions=data["positions"][n].tolist()
        tag="audit" if n%4==0 else "design"
        for j,p in enumerate(positions):
            if p+1>=len(tokens):continue
            is_start=int(tokens[p+1])==select and bool(re.search(r"```sql\s*$",tokenizer.decode(tokens[max(0,p-8):p+1]),re.I))
            groups[f"{tag}_{'start' if is_start else 'other'}"].append(data["mlp_act:27"][n][j].float())
        counts.update(tokens[positions[0]:].tolist())
    groups={name:torch.stack(rows) for name,rows in groups.items()}
    audit=solve(groups["design_start"],groups["design_other"],columns=64)
    diagnostics={name:score_stats(rows@audit) for name,rows in groups.items()}
    k=solve(torch.cat([groups["design_start"],groups["audit_start"]]),torch.cat([groups["design_other"],groups["audit_other"]]),columns=64)
    embedding,norm=load("model.embed_tokens.weight"),load("model.norm.weight")
    targets={}; details={}
    for name,word in (("cte","WITH"),("comment","--")):
        token_ids=tokenizer(word,add_special_tokens=False)["input_ids"];assert len(token_ids)==1
        token=token_ids[0]
        keep=sorted(set([t for t,_ in counts.most_common() if t!=token][:400])|{select,73594,16,17,10061,8420})
        constraints=norm*embedding[[token,*keep]]
        rhs=torch.zeros(len(constraints));rhs[0]=5.5
        u=constraints.T@torch.linalg.solve(constraints@constraints.T+1e-6*torch.eye(len(rhs)),rhs)
        targets[name]=u;details[name]={"token":token,"unit_value_norm":float(u.norm())}
    torch.save({"key":k,"targets":targets,"diagnostics":diagnostics},HERE/"sql_start_design.pt")
    report={"position_key":diagnostics,"targets":details}
    (HERE/"sql_start_design.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
