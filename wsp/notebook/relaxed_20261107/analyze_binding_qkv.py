"""Factor the cached alias-binding effect into Q, K, and V counterfactuals.

This recombines recorded layer inputs within one attention head. It neither
runs the full network nor uses gradients or target SQL fitting.
"""
import itertools
import json
from pathlib import Path
import re
import sys

from t2s.wsp.common import base_dir
from t2s.wsp.kernel import load, torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from attnlib import rms, rope

torch.set_num_threads(4)
HEADS = ((22, 7), (23, 4))


def main():
    data = torch.load(HERE.parents[1] / "passes/pass_05/read.pt", weights_only=True)
    texts = {r["id"]: r for r in map(json.loads, (HERE.parent / "texts_pass05.jsonl").read_text().splitlines())}
    rows = {r["id"]: r for r in json.loads((HERE.parent / "alias_cf_rows.json").read_text())}
    original = {r["id"]: r for r in map(json.loads, (HERE.parents[1] / "passes/pass_01/original.jsonl").read_text().splitlines())}
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir()))
    embedding, norm = load("model.embed_tokens.weight"), load("model.norm.weight")
    indices = {label: n for n, label in enumerate(data["ids"])}
    pids = sorted({label.split("_", 1)[1] for label in data["ids"]})
    output, cache = {}, {}
    for layer, head in HEADS:
        name = f"L{layer}H{head}"
        weights = {part: load(f"model.layers.{layer}.{part}") for part in (
            "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.q_proj.bias",
            "self_attn.k_proj.weight", "self_attn.k_proj.bias", "self_attn.v_proj.weight", "self_attn.v_proj.bias",
            "self_attn.o_proj.weight")}
        kv = head // 6
        q_weight = weights["self_attn.q_proj.weight"][head*128:(head+1)*128]
        q_bias = weights["self_attn.q_proj.bias"][head*128:(head+1)*128]
        k_weight = weights["self_attn.k_proj.weight"][kv*128:(kv+1)*128]
        k_bias = weights["self_attn.k_proj.bias"][kv*128:(kv+1)*128]
        v_weight = weights["self_attn.v_proj.weight"][kv*128:(kv+1)*128]
        v_bias = weights["self_attn.v_proj.bias"][kv*128:(kv+1)*128]
        o_weight = weights["self_attn.o_proj.weight"][:,head*128:(head+1)*128]
        results, differences, query_differences, query_midpoints, cases = [], [], [], [], []
        covariance, mean, count = torch.zeros(1536,1536), torch.zeros(1536), 0
        for pid in pids:
            variants = {}
            for tag in ("Aown", "Aswap"):
                label = f"{tag}_{pid}"; n = indices[label]
                p = len(data["tokens"][n]) - len(tokenizer(texts[label]["answer"], add_special_tokens=False)["input_ids"]) - 1
                prompt_len = len(tokenizer(texts[label]["prompt"].split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n", add_special_tokens=False)["input_ids"])
                x = rms(data[f"resid:{layer-1}"][n][:p+1].float(), weights["input_layernorm.weight"])
                q = rope((q_weight@x[p]+q_bias)[None,None],torch.tensor([p]))[0,0]
                k = rope((x@k_weight.T+k_bias)[None],torch.arange(p+1))[0]
                v = x@v_weight.T+v_bias
                digit = rows[pid]["digit_own"] if tag == "Aown" else 3-rows[pid]["digit_own"]
                variants[digit] = {"x":x,"q":q,"k":k,"v":v,"p":p,"prompt_len":prompt_len}
            a,b = variants[1],variants[2]
            assert a["p"]==b["p"]
            query_differences.append(a["x"][a["p"]]-b["x"][b["p"]])
            query_midpoints.append((a["x"][a["p"]]+b["x"][b["p"]])/2)
            match=re.search(r"FROM\s+(\w+)\s+AS\s+T1\s+(?:INNER\s+)?JOIN\s+(\w+)\s+AS\s+T2\s",original[pid]["pred_sql"])
            t1,t2=[tokenizer(" "+table)["input_ids"][0] for table in match.groups()]
            owner,other=(t1,t2) if rows[pid]["owner_is_t1"] else (t2,t1)
            contrast=norm*(embedding[owner]-embedding[other])/5.5
            for digit, state in variants.items():
                c=rows[pid]
                baseline_margin=c["own27"] if digit==c["digit_own"] else c["swap27"]
                cases.append({"id":pid,"digit":digit,"query_input":state["x"][state["p"]],
                              "query":state["q"],"key":state["k"],"value":state["v"],
                              "position":state["p"],"readout":o_weight.T@contrast,
                              "baseline_margin":baseline_margin})
            values={}; attentions={}
            for qv,kvv,vv in itertools.product((1,2),repeat=3):
                attention=((variants[kvv]["k"]@variants[qv]["q"])/128**.5).softmax(-1)
                out=o_weight@(attention@variants[vv]["v"])
                values[(qv,kvv,vv)]=float(out@contrast)
                if qv==kvv==vv: attentions[qv]=attention
            row={"id":pid,"total":values[(1,1,1)]-values[(2,2,2)]}
            for part,index in (("Q",0),("K",1),("V",2)):
                shifts=[]
                for others in itertools.product((1,2),repeat=2):
                    one=list(others);one.insert(index,1)
                    two=list(others);two.insert(index,2)
                    shifts.append(values[tuple(one)]-values[tuple(two)])
                row[part]=sum(shifts)/4
            results.append(row)
            # Cache a contrastive subspace over the answer locations actually
            # read by this head, rather than only the query at FROM.
            average_attention=(attentions[1]+attentions[2])/2
            start=a["prompt_len"]
            answer_attention=average_attention[start:]
            top=answer_attention.topk(min(6,len(answer_attention)))
            for mass,local in zip(top.values,top.indices):
                position=start+int(local)
                differences.append((a["x"][position]-b["x"][position])*mass.sqrt())
            background=a["x"]
            covariance += background.T@background
            mean += background.sum(0); count +=len(background)
        output[name]={part:sum(row[part] for row in results)/len(results) for part in ("total","Q","K","V")}
        output[name]["n_pairs"]=len(results)
        mean/=count
        covariance=covariance/count-torch.outer(mean,mean)
        cache[name]={"differences":torch.stack(differences),"query_differences":torch.stack(query_differences),
                     "query_midpoints":torch.stack(query_midpoints),
                     "covariance":covariance,"mean":mean,"rows":results,"cases":cases}
        print(name,json.dumps(output[name]),flush=True)
    torch.save(cache,HERE/"binding_qkv_geometry.pt")
    (HERE/"binding_qkv.json").write_text(json.dumps(output,indent=2)+"\n")


if __name__=="__main__":
    main()
