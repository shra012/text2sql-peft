"""Construct controlled read texts; no gold completions are used as targets."""
import json
from pathlib import Path
import re

from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
from prepare_probes import INSTRUCTIONS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def write(name, rows):
    (HERE / name).write_text("".join(json.dumps(row) + "\n" for row in rows))


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir()))
    practice = {r["id"]: r for r in map(json.loads, (ROOT / "wsp/practice.jsonl").read_text().splitlines())}
    baseline = {r["id"]: r for r in map(json.loads, (ROOT / "wsp/passes/pass_26/edited.jsonl").read_text().splitlines())}
    screen = (HERE / "screen512.txt").read_text().splitlines()
    short = [i for i in screen if i.startswith("p")]
    long = [i for i in screen if i.startswith("x")]
    calibration = []
    for pid in short[:8] + long[:8]:
        calibration.append({"id": pid,
                            "prompt": tokenizer.apply_chat_template(practice[pid]["messages"], tokenize=False, add_generation_prompt=True),
                            "answer": baseline[pid]["raw"]})
    write("calibration16.jsonl", calibration)
    texts, metadata = [], []
    for pid in short[:16] + long[:16]:
        raw = baseline[pid]["raw"]
        anchors = []
        for pattern in (r"\bSELECT\b", r"\bFROM\b", r"\bWHERE\b"):
            found = re.search(pattern, raw, re.I)
            anchors.append(found.end() if found else min(len(raw.rstrip("`\n ")) - 1, 60))
        for variant in ("original", *INSTRUCTIONS):
            messages = [dict(m) for m in practice[pid]["messages"]]
            if variant != "original":
                messages[0]["content"] += " " + INSTRUCTIONS[variant]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for anchor, end in enumerate(anchors):
                label = f"{pid}_{variant}_{anchor}"
                texts.append({"id": label, "prompt": prompt + raw[:max(1, end)], "answer": ""})
                metadata.append({"id": label, "original_id": pid, "variant": variant, "anchor": anchor})
    write("contrast480.jsonl", texts)
    (HERE / "contrast_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Prepared", len(calibration), "full-context calibration reads and", len(texts), "controlled prefix reads.")


if __name__ == "__main__":
    main()
