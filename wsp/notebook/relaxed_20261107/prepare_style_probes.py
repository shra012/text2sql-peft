"""Probe SQL planning representations, keeping all final edit prompts unchanged."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
from prepare_probes import INSTRUCTIONS

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
STYLES={
    "cte":"Use a WITH clause to define the joined and filtered records before the final SELECT. In the CTE, include all tables needed by the selected columns or conditions. Then select the requested results from that CTE. Return one SQL code block.",
    "qualified":"Fully qualify every column with its real table name. Do not use T1 or T2 aliases. Include every join required by those columns and conditions. Return only one SQLite query in a SQL code block.",
    "comment_plan":"Before the SELECT query, write a short SQL comment mapping each required table to its alias and stating the required joins. Use exactly those aliases and joins in the query. Return the comment and query together in one SQL code block.",
    "semantic_alias":"Use meaningful table aliases derived from the table names instead of T1 or T2. Choose all required tables and their aliases before choosing the SELECT expressions. Keep each column attached to its owning table. Return only SQL.",
}


def main():
    path=ROOT/"wsp/practice.jsonl"
    rows={r["id"]:r for r in map(json.loads,path.read_text().splitlines())}
    original_meta=json.loads((HERE/"instruction_probe_metadata.json").read_text())
    ids=list(dict.fromkeys(r["original_id"] for r in original_meta))
    new,metadata=[],[]
    for variant,instruction in STYLES.items():
        for pid in ids:
            row=dict(rows[pid]);row["id"]=f"zstyle_{variant}_{pid}"
            row["messages"]=[dict(m) for m in row["messages"]]
            row["messages"][0]["content"]+=" "+instruction
            row["probe_only"]=True
            new.append(row);metadata.append({"id":row["id"],"original_id":pid,"variant":variant})
    assert not any(r["id"] in rows for r in new)
    with path.open("a") as f:
        for row in new:f.write(json.dumps(row)+"\n")
    (HERE/"style_probe512.txt").write_text("\n".join(r["id"] for r in new)+"\n")
    (HERE/"style_probe_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    first_ids=list(dict.fromkeys(r["original_id"] for r in json.loads((HERE/"extended_metadata.json").read_text()) if r["kind"]=="first"))
    tokenizer=AutoTokenizer.from_pretrained(str(base_dir()))
    texts,meta=[],[]
    variants={"original":"","neutral":INSTRUCTIONS["neutral"],**STYLES}
    for pid in first_ids:
        for variant,instruction in variants.items():
            messages=[dict(m) for m in rows[pid]["messages"]]
            if instruction:messages[0]["content"]+=" "+instruction
            label=f"style_first_{pid}_{variant}"
            texts.append({"id":label,"prompt":tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),"answer":""})
            meta.append({"id":label,"original_id":pid,"variant":variant})
    (HERE/"style_first192.jsonl").write_text("".join(json.dumps(r)+"\n" for r in texts))
    (HERE/"style_first_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print("Prepared",len(new),"diagnostic generations and",len(texts),"first-state reads.")


if __name__=="__main__":main()
