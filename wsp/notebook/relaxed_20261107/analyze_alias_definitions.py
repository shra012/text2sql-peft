"""Inspect alias assignment after FROM, rather than the earlier SELECT alias.

The schema and already-generated column determine which alias would bind the
first table correctly. These labels diagnose native states; no weights are fit.
"""
import json
from pathlib import Path

from t2s.wsp.kernel import load,torch
from t2s.wsp.common import base_dir
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
torch.set_num_threads(4)


def main():
    data=torch.load(HERE.parents[1]/"passes/pass_03/read.pt",weights_only=True)
    metadata={r["id"]:r for r in json.loads((HERE.parent/"alias_cf_rows.json").read_text())}
    tokenizer=AutoTokenizer.from_pretrained(str(base_dir()))
    table_alias=tokenizer.convert_tokens_to_ids("ĠT");as_id=tokenizer.convert_tokens_to_ids("ĠAS")
    from_ids={tokenizer.convert_tokens_to_ids("ĠFROM"),tokenizer.convert_tokens_to_ids("FROM")}
    embedding,norm=load("model.embed_tokens.weight"),load("model.norm.weight")
    contrast=norm*(embedding[16]-embedding[17])
    weight=load("model.layers.27.mlp.down_proj.weight")
    feature=weight.T@contrast/5.5
    rows=[];positive=[];negative=[]
    for n,label in enumerate(data["ids"]):
        if not label.startswith("F_"):continue
        pid=label[2:];tokens=data["tokens"][n];chosen=None;seen_from=False
        if pid in metadata:
            for j,p in enumerate(data["positions"][n].tolist()):
                if int(tokens[p]) in from_ids:seen_from=True
                if seen_from and int(tokens[p])==table_alias and p>0 and int(tokens[p-1])==as_id and p+1<len(tokens) and int(tokens[p+1]) in (16,17):
                    chosen=j;break
        acts=data["mlp_act:27"][n].float()
        if chosen is None:
            negative.append(acts);continue
        m=metadata[pid]
        needed=m["digit_own"] if m["owner_is_t1"] else 3-m["digit_own"]
        row={"id":pid,"needed":needed,"margins":{}}
        for layer in (12,15,17,19,21,23,25,26,27):
            h=data[f"resid:{layer}"][n][chosen].float()
            if layer==27:margin=float(h@(embedding[16]-embedding[17]))
            else:margin=float((h*torch.rsqrt(h.square().mean()+1e-6)*norm)@(embedding[16]-embedding[17]))
            row["margins"][layer]=margin
        row["mlp_dla"]=float(acts[chosen]@feature)
        rows.append(row);positive.append(acts[chosen])
        negative.append(torch.cat([acts[:chosen],acts[chosen+1:]]))
    positive=torch.stack(positive);negative=torch.cat(negative)
    print("Cases",len(rows),"needed 1",sum(r["needed"]==1 for r in rows),"needed 2",sum(r["needed"]==2 for r in rows),flush=True)
    for layer in (12,15,17,19,21,23,25,26,27):
        values={digit:[r["margins"][layer] for r in rows if r["needed"]==digit] for digit in (1,2)}
        print("layer",layer,{digit:{"mean_margin_1_minus_2":sum(v)/len(v),"consistent":sum(x>0 if digit==1 else x<0 for x in v),"n":len(v)} for digit,v in values.items()},flush=True)
    contribution=(positive*feature).mean(0)
    selectivity=positive.abs().mean(0)/(negative.abs().mean(0)+1e-6)
    top=contribution.argsort(descending=True)[:16]
    print("Late neurons pushing first AS alias toward 1:",[{"neuron":int(i),"dla":float(contribution[i]),"selectivity":float(selectivity[i])} for i in top],flush=True)
    torch.save({"positive":positive,"negative":negative,"feature":feature,"rows":rows},HERE/"alias_definition_inputs.pt")
    (HERE/"alias_definition_rows.json").write_text(json.dumps(rows,indent=2)+"\n")


if __name__=="__main__":main()
