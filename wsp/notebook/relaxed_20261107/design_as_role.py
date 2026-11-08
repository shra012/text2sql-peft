"""A simpler AS-definition detector, independent of first/second table order.

Raise token 2 only, preserving tokens 1 and 3. This retains the model's native
choice between unused aliases after T2 is already defined. No SQL answers fit
the update; role labels come from the literal AS T prefix in recorded text.
"""
from collections import Counter
import json
from pathlib import Path
from t2s.wsp.kernel import load,torch
from t2s.wsp.common import base_dir
from transformers import AutoTokenizer
from design_alias_assignment import key,stats

HERE=Path(__file__).resolve().parent
torch.set_num_threads(4)


def main():
    data=torch.load(HERE.parents[1]/"passes/pass_03/read.pt",weights_only=True)
    tokenizer=AutoTokenizer.from_pretrained(str(base_dir()))
    tid=tokenizer.convert_tokens_to_ids("ĠT");asid=tokenizer.convert_tokens_to_ids("ĠAS")
    positive,negative=[],[];counts=Counter()
    for n,label in enumerate(data["ids"]):
        if not label.startswith("F_"):continue
        tokens=data["tokens"][n];a=data["mlp_act:27"][n]
        for j,p in enumerate(data["positions"][n].tolist()):
            is_as=int(tokens[p])==tid and p>0 and int(tokens[p-1])==asid and p+1<len(tokens) and 16<=int(tokens[p+1])<=24
            (positive if is_as else negative).append(a[j].float())
        counts.update(tokens[data["positions"][n][0]:].tolist())
    positive,negative=torch.stack(positive),torch.stack(negative)
    native=torch.load(HERE.parents[1]/"passes/pass_52/read.pt",weights_only=True)
    meta=json.loads((HERE/"alias_assignment_metadata.json").read_text())
    extra={c:torch.stack([native["mlp_act:27"][n][0].float() for n,r in enumerate(meta) if r["condition"]==c]) for c in ("first","second","second_forced2","select")}
    pos=torch.cat([positive,extra["first"],extra["second"],extra["second_forced2"]])
    groups=torch.load(HERE.parent/"t1_groups_L27.pt",weights_only=True)["G"]
    protected=torch.cat([groups["pre_first"],groups["pre_later"],extra["select"]])
    excluded=(torch.load(HERE.parent/"edit_f4_vectors.pt",weights_only=True)["k"]!=0).nonzero().flatten()
    k=key(pos,negative,[protected],excluded,columns=768)
    embedding,norm=load("model.embed_tokens.weight"),load("model.norm.weight")
    keep=sorted(set([i for i,_ in counts.most_common() if i!=17][:400])|{16,18,19,20,73594,10061,8420})
    constraints=norm*embedding[[17,*keep]]
    rhs=torch.zeros(len(constraints));rhs[0]=5.5
    u=constraints.T@torch.linalg.solve(constraints@constraints.T+1e-6*torch.eye(len(rhs)),rhs)
    report={"native_as":stats(positive@k),"native_non_as":stats(negative@k),"protected_select":stats(protected@k),
            "new_roles":{c:stats(x@k) for c,x in extra.items()},"value_norm_per_nat":float(u.norm())}
    torch.save({"key":k,"value":u,"report":report},HERE/"as_role_design.pt")
    (HERE/"as_role_design.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
