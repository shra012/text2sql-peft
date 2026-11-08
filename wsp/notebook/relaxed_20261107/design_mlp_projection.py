"""Remove a late MLP's alias-contrast subspace while preserving native means.

The subspace is defined by recorded T1/T2 activation differences, not by target
SQL or output regression. Protected vectors include the original format edit's
columns, first-answer states, and pair midpoints. This is a one-shot geometric
intervention on the existing matrix, with no training objective or optimizer.
"""
import argparse
import json
from pathlib import Path
import re

from t2s.wsp.common import base_dir
from t2s.wsp.kernel import load, torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
torch.set_num_threads(4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--full-covariance", action="store_true")
    args = parser.parse_args()
    data = torch.load(HERE.parents[1] / "passes/pass_03/read.pt", weights_only=True)
    rows = json.loads((HERE.parent / "alias_cf_rows.json").read_text())
    heldout = {r["id"].split("_",1)[1] for r in map(json.loads,(HERE.parent / "texts_pass07.jsonl").read_text().splitlines()) if r["id"].startswith("Aown_")}
    indices = {label:n for n,label in enumerate(data["ids"])}
    first_indices = [n for n,label in enumerate(data["ids"]) if label.startswith("F_")]
    other = torch.cat([data["mlp_act:27"][n][1:].float() for n in first_indices])
    first = torch.stack([data["mlp_act:27"][n][0].float() for n in first_indices])
    scale = other.square().mean(0).sqrt().clamp_min(.01)
    one,two=[],[]
    for row in rows:
        a=data["mlp_act:27"][indices["Aown_"+row["id"]]][0].float()
        b=data["mlp_act:27"][indices["Aswap_"+row["id"]]][0].float()
        x,y=(a,b) if row["digit_own"]==1 else (b,a)
        one.append(x);two.append(y)
    one,two=torch.stack(one),torch.stack(two)
    design_mask=torch.tensor([row["id"] not in heldout for row in rows])
    midpoint=(one[design_mask]+two[design_mask])/2
    protected=[first,midpoint,other.mean(0)[None]]
    tokenizer=AutoTokenizer.from_pretrained(str(base_dir()))
    if args.strict:
        alias_ids={tokenizer.convert_tokens_to_ids("ĠT"),tokenizer.convert_tokens_to_ids("(T")}
        aliases=[]
        for n in first_indices:
            tokens=data["tokens"][n]
            for j,p in enumerate(data["positions"][n].tolist()):
                if int(tokens[p]) in alias_ids and p+1<len(tokens) and int(tokens[p+1]) in (16,17):
                    aliases.append(data["mlp_act:27"][n][j].float())
        protected.append(torch.stack(aliases))
    f4=torch.load(HERE.parent / "edit_f4_vectors.pt",weights_only=True)
    protected_columns=(f4["k"]!=0).nonzero().flatten()
    units=torch.zeros(8960,len(protected_columns));units[protected_columns,torch.arange(len(protected_columns))]=1
    if args.full_covariance:
        chol_path=HERE/"mlp_background_cholesky.pt"
        if chol_path.exists():
            chol=torch.load(chol_path,weights_only=True)
        else:
            print("Computing full native-answer covariance",flush=True)
            covariance=other.T@other/len(other)
            covariance.diagonal().add_(.01*covariance.diagonal().mean())
            chol=torch.linalg.cholesky(covariance)
            torch.save(chol,chol_path)
            del covariance
        white_protected=torch.linalg.solve_triangular(chol,torch.cat([*protected,units.T]).T,upper=False)
        white_protected/=white_protected.norm(dim=0,keepdim=True).clamp_min(1e-6)
        difference=torch.linalg.solve_triangular(chol,(one[design_mask]-two[design_mask]).T,upper=False).T
    else:
        white_protected=torch.cat([(torch.cat(protected)/scale).T,units],dim=1)
        difference=(one[design_mask]-two[design_mask])/scale
    u,s,_=torch.linalg.svd(white_protected,full_matrices=False)
    q=u[:,s>s.max()*1e-4]
    difference-= (difference@q)@q.T
    basis=torch.linalg.svd(difference,full_matrices=False).Vh[:64].T
    if args.full_covariance:
        keys=torch.linalg.solve_triangular(chol.T,basis,upper=True)
        keys[protected_columns]=0
        value_axes=chol@basis
    else:
        basis[protected_columns]=0
        keys=basis/scale[:,None]
        value_axes=scale[:,None]*basis
    values=load("model.layers.27.mlp.down_proj.weight")@value_axes
    embedding,norm=load("model.embed_tokens.weight"),load("model.norm.weight")
    predictions={r["id"]:r for r in map(json.loads,(HERE.parents[1]/"passes/pass_01/original.jsonl").read_text().splitlines())}
    contrasts=[];before_one=[];before_two=[]
    for row in rows:
        match=re.search(r"FROM\s+(\w+)\s+AS\s+T1\s+(?:INNER\s+)?JOIN\s+(\w+)\s+AS\s+T2\s",predictions[row["id"]]["pred_sql"])
        a,b=[tokenizer(" "+t,add_special_tokens=False)["input_ids"][0] for t in match.groups()]
        owner,other_token=(a,b) if row["owner_is_t1"] else (b,a)
        contrasts.append(norm*(embedding[owner]-embedding[other_token])/5.5)
        before_one.append(row["own27"] if row["digit_own"]==1 else row["swap27"])
        before_two.append(row["swap27"] if row["digit_own"]==1 else row["own27"])
    contrasts=torch.stack(contrasts);before_one=torch.tensor(before_one);before_two=torch.tensor(before_two)
    report=[]
    for rank in (8,32,64):
        k,v=keys[:,:rank],values[:,:rank]
        shift_one=-((one@k)@v.T*contrasts).sum(1)
        shift_two=-((two@k)@v.T*contrasts).sum(1)
        background=-((other@k)@v.T)
        row={"rank":rank,"strict":args.strict,"background_delta_rms_q50_q99":background.square().mean(-1).sqrt().quantile(torch.tensor([.5,.99])).tolist()}
        for mask,label in ((design_mask,"design"),(~design_mask,"heldout")):
            a,b=before_one[mask],before_two[mask];c,d=a+shift_one[mask],b+shift_two[mask]
            row[label]={"n_pairs":int(mask.sum()),"binding_before":float((a-b).mean()),"binding_after":float((c-d).mean()),
                        "T1_before":int((a>0).sum()),"T1_after":int((c>0).sum()),"T2_before":int((b<0).sum()),"T2_after":int((d<0).sum())}
        report.append(row);print(json.dumps(row),flush=True)
    suffix=("strict" if args.strict else "mean")+("_full" if args.full_covariance else "")
    torch.save({"keys":keys,"values":values,"protected_columns":protected_columns,"report":report},HERE/f"mlp_projection_{suffix}.pt")
    (HERE/f"mlp_projection_{suffix}.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
