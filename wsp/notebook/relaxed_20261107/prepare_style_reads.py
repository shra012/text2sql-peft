"""Capture instruction states at answer start and at the first SQL word."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
from prepare_probes import INSTRUCTIONS
from prepare_style_probes import STYLES

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
rows={r["id"]:r for r in map(json.loads,(ROOT/"wsp/practice.jsonl").read_text().splitlines())}
ids=list(dict.fromkeys(r["original_id"] for r in json.loads((HERE/"style_first_metadata.json").read_text())))
tokenizer=AutoTokenizer.from_pretrained(str(base_dir()))
texts,metadata=[],[]
for pid in ids:
    for phase in ("first","sql"):
        variants={"original":"","neutral":INSTRUCTIONS["neutral"],**STYLES}
        if phase=="sql":variants.update({v:INSTRUCTIONS[v] for v in ("expert","plan")})
        for variant,instruction in variants.items():
            messages=[dict(m) for m in rows[pid]["messages"]]
            if instruction:messages[0]["content"]+=" "+instruction
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            if phase=="sql":prompt+="```sql\n"
            label=f"{phase}_{pid}_{variant}"
            texts.append({"id":label,"prompt":prompt,"answer":""})
            metadata.append({"id":label,"original_id":pid,"variant":variant,"phase":phase})
(HERE/"style_reads448.jsonl").write_text("".join(json.dumps(r)+"\n" for r in texts))
(HERE/"style_read_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
print("Prepared",len(texts),"controlled style/position reads.")
