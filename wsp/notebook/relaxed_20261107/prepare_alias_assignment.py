"""Probe first alias definition and whether later definitions avoid reused aliases."""
import json
from pathlib import Path
import re
import sqlite3
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
practice={r["id"]:r for r in map(json.loads,(ROOT/"wsp/practice.jsonl").read_text().splitlines())}
predictions={r["id"]:r for r in map(json.loads,(ROOT/"wsp/passes/pass_26/edited.jsonl").read_text().splitlines())}
tokenizer=AutoTokenizer.from_pretrained(str(base_dir()))
catalog={};counts={"p":0,"x":0};texts=[];metadata=[]
for pid,row in predictions.items():
    if counts[pid[0]]>=32:continue
    raw=row["raw"]
    definitions=list(re.finditer(r"\b(FROM|JOIN)\s+([`\"]?\w+[`\"]?)\s+AS\s+T([12])\b",raw,re.I))
    if len(definitions)<2:continue
    first,second=definitions[:2]
    if first.group(1).upper()!="FROM" or second.group(1).upper()!="JOIN" or first.group(3)!="1" or second.group(3)!="2":continue
    selected=re.search(r"\bT([12])\.(?:`([^`]+)`|\"([^\"]+)\"|(\w+))",raw[:first.start()])
    if not selected:continue
    column=next(v for v in selected.groups()[1:] if v is not None).casefold()
    db=practice[pid]["db_path"]
    if db not in catalog:
        conn=sqlite3.connect(f"file:{db}?mode=ro",uri=True);tables={}
        try:
            for table, in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                quoted=table.replace('"','""')
                tables[table.casefold()]={r[1].casefold() for r in conn.execute(f'PRAGMA table_info("{quoted}")')}
        finally:conn.close()
        catalog[db]=tables
    one_table=first.group(2).strip('`"').casefold();two_table=second.group(2).strip('`"').casefold()
    owners=[t for t,columns in catalog[db].items() if column in columns]
    needed=None
    if len(owners)==1 and owners[0] in (one_table,two_table):
        needed=int(selected.group(1)) if owners[0]==one_table else 3-int(selected.group(1))
    original_prompt=tokenizer.apply_chat_template(practice[pid]["messages"],tokenize=False,add_generation_prompt=True)
    prefixes={"first":raw[:first.start(3)],"second":raw[:second.start(3)],
              "second_forced2":raw[:first.start(3)]+"2"+raw[first.end(3):second.start(3)],
              "select":raw[:selected.start(1)]}
    for condition,prefix in prefixes.items():
        label=f"assignment_{pid}_{condition}"
        texts.append({"id":label,"prompt":original_prompt+prefix,"answer":""})
        metadata.append({"id":label,"original_id":pid,"condition":condition,"needed_first":needed,"selected_digit":int(selected.group(1))})
    counts[pid[0]]+=1
(HERE/"alias_assignment_reads.jsonl").write_text("".join(json.dumps(r)+"\n" for r in texts))
(HERE/"alias_assignment_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
print("Prepared",len(texts),"alias-assignment reads on",counts)
