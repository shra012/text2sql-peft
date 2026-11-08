"""Two positional alias-assignment rules from native states, without SQL fitting.

Rule 1 relaxes the model's fixed preference for calling the first table T1.
Rule 2 optionally chooses T1 for the second table after the first was named T2.
Keys use position/alias-prefix labels only. Values are hand-specified token
constraints; no correct-SQL regression, optimizer, or gradient is involved.
"""
from collections import Counter
import json
from pathlib import Path
from t2s.wsp.common import base_dir
from t2s.wsp.kernel import load,torch
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
torch.set_num_threads(4)


def key(positive,negative,protected,excluded,columns=512):
    score=(positive.mean(0)-negative.mean(0)).abs()/(positive.std(0)+negative.std(0)+.001)
    score[excluded]=-torch.inf
    indices=score.topk(columns).indices
    x=negative[:,indices].double();covariance=x.T@x/len(x)
    covariance.diagonal().add_(.01*covariance.diagonal().mean())
    constraints=[positive[:,indices].mean(0).double()[:,None]]
    for group in protected:
        _,singular,vh=torch.linalg.svd(group[:,indices].double(),full_matrices=False)
        basis=vh[singular>singular.max()*1e-4].T
        constraints.append(basis)
    matrix=torch.cat(constraints,dim=1)
    inverse=torch.linalg.solve(covariance,matrix)
    target=torch.zeros(matrix.shape[1],dtype=torch.float64);target[0]=1
    small=inverse@torch.linalg.solve(matrix.T@inverse+1e-8*torch.eye(len(target),dtype=torch.float64),target)
    small/=small@positive[:,indices].mean(0).double()
    full=torch.zeros(8960);full[indices]=small.float()
    return full


def stats(values):
    return {"n":len(values),"mean":float(values.mean()),"q01_q50_q99":values.quantile(torch.tensor([.01,.5,.99])).tolist()}


def main():
    data=torch.load(HERE.parents[1]/"passes/pass_52/read.pt",weights_only=True)
    meta=json.loads((HERE/"alias_assignment_metadata.json").read_text())
    assert data["ids"]==[r["id"] for r in meta]
    lookup={(r["original_id"],r["condition"]):n for n,r in enumerate(meta)}
    ids=list(dict.fromkeys(r["original_id"] for r in meta))
    details={r["original_id"]:r for r in meta}
    activations={condition:torch.stack([data["mlp_act:27"][lookup[(pid,condition)]][0].float() for pid in ids])
                 for condition in ("first","second","second_forced2","select")}
    old=torch.load(HERE/"alias_definition_inputs.pt",weights_only=True)
    old_meta={r["id"]:r for r in json.loads((HERE.parent/"alias_cf_rows.json").read_text())}
    old_two=torch.tensor([old_meta[r["id"]]["digit_own"]==2 for r in old["rows"]])
    is_two=torch.tensor([details[pid]["selected_digit"]==2 for pid in ids])
    excluded=(torch.load(HERE.parent/"edit_f4_vectors.pt",weights_only=True)["k"]!=0).nonzero().flatten()
    earlier_select=torch.load(HERE.parent/"t1_groups_L27.pt",weights_only=True)["G"]["pre_first"]
    select_protection=torch.cat([earlier_select,activations["select"]])
    other=torch.cat([old["negative"],activations["second"],activations["second_forced2"],activations["select"]])
    first=torch.cat([old["positive"],activations["first"]])
    first_two=torch.cat([old["positive"][old_two],activations["first"][is_two]])
    first_one=torch.cat([old["positive"][~old_two],activations["first"][~is_two]])
    first_protected=[select_protection,activations["second"],activations["second_forced2"]]
    keys={"all":key(first,other,first_protected,excluded),
          "t2":key(first_two,torch.cat([other,first_one]),[*first_protected,first_one],excluded),
          "second":key(activations["second_forced2"],torch.cat([other,first]),
                       [select_protection,activations["second"],activations["first"]],excluded)}
    report={name:{condition:stats(rows@k) for condition,rows in activations.items()} for name,k in keys.items()}
    report["earlier_select_protection"]={name:stats(earlier_select@k) for name,k in keys.items()}
    tokenizer=AutoTokenizer.from_pretrained(str(base_dir()))
    embedding,norm=load("model.embed_tokens.weight"),load("model.norm.weight")
    digit_ids=[tokenizer(str(i),add_special_tokens=False)["input_ids"][0] for i in range(1,10)]
    for condition in ("first","second","second_forced2"):
        h=torch.stack([data["resid:27"][lookup[(pid,condition)]][0].float() for pid in ids])
        logits=h@embedding[digit_ids].T
        report[condition+"_native_digit_counts"]={str(i+1):int((logits.argmax(-1)==i).sum()) for i in range(9)}
    counts=Counter()
    for row in map(json.loads,(HERE.parents[1]/"passes/pass_26/edited.jsonl").read_text().splitlines()):
        counts.update(tokenizer(row["raw"],add_special_tokens=False)["input_ids"])
    def value(changes):
        target_ids=list(changes)
        keep=sorted(set([t for t,_ in counts.most_common() if t not in target_ids][:400])|{73594,10061,8420})
        matrix=norm*embedding[[*target_ids,*keep]]
        rhs=torch.cat([torch.tensor(list(changes.values()))*5.5,torch.zeros(len(keep))])
        return matrix.T@torch.linalg.solve(matrix@matrix.T+1e-6*torch.eye(len(rhs)),rhs)
    first_value=value({16:-.5,17:.5})
    second_value=value({16:6.,17:-4.,18:-4.})
    native_first=torch.stack([data["resid:27"][lookup[(pid,"first")]][0].float() for pid in ids])@(embedding[16]-embedding[17])
    known=torch.tensor([details[pid]["needed_first"] is not None for pid in ids])
    wanted=torch.tensor([details[pid]["needed_first"] or 1 for pid in ids])
    report["first_native_margin_by_needed"]={str(d):stats(native_first[known&(wanted==d)]) for d in (1,2)}
    report["first_bias_estimates"]={}
    for mode in ("all","t2"):
        for strength in (4.,6.):
            after=native_first-strength*(activations["first"]@keys[mode])
            before_ok=((native_first>0)==(wanted==1))&known
            after_ok=((after>0)==(wanted==1))&known
            report["first_bias_estimates"][f"{mode}_{strength}"]={"n":int(known.sum()),"before":int(before_ok.sum()),"after":int(after_ok.sum())}
    torch.save({"keys":keys,"first_value":first_value,"second_value":second_value,"report":report},HERE/"alias_assignment_design.pt")
    (HERE/"alias_assignment_design.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
