"""Correct synthetic SQL demonstrations and controlled, corrupted read prompts.

Demonstrations are inference probes, never supervised weight-training targets.
All final direct-edit scores still use the original benchmark prompts.
"""
import json
from pathlib import Path
import sqlite3
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SCHEMAS=["Dept(id,name);Emp(id,dept_id,salary)","Product(id,name);Item(sale_id,product_id,qty);Sale(id,status)"]
QUESTIONS=["Department names with employees earning more than 100.","Product names in paid sales with item quantity at least 3."]
QUERIES={
    "canonical":["SELECT DISTINCT T1.name FROM Dept AS T1 JOIN Emp AS T2 ON T1.id=T2.dept_id WHERE T2.salary>100;",
                 "SELECT DISTINCT T1.name FROM Product AS T1 JOIN Item AS T2 ON T1.id=T2.product_id JOIN Sale AS T3 ON T2.sale_id=T3.id WHERE T3.status='paid' AND T2.qty>=3;"],
    "permuted":["SELECT DISTINCT T2.name FROM Emp AS T1 JOIN Dept AS T2 ON T1.dept_id=T2.id WHERE T1.salary>100;",
                "SELECT DISTINCT T3.name FROM Sale AS T1 JOIN Item AS T2 ON T1.id=T2.sale_id JOIN Product AS T3 ON T2.product_id=T3.id WHERE T1.status='paid' AND T2.qty>=3;"],
    "qualified":["SELECT DISTINCT Dept.name FROM Dept JOIN Emp ON Dept.id=Emp.dept_id WHERE Emp.salary>100;",
                 "SELECT DISTINCT Product.name FROM Product JOIN Item ON Product.id=Item.product_id JOIN Sale ON Item.sale_id=Sale.id WHERE Sale.status='paid' AND Item.qty>=3;"],
}
QUERIES["mapping"]=["-- T1=Emp,T2=Dept;join dept_id=id\n"+QUERIES["permuted"][0],
                    "-- T1=Sale,T2=Item,T3=Product;join sale_id=id and product_id=id\n"+QUERIES["permuted"][1]]


def examples(variant):
    queries=QUERIES["canonical"][::-1] if variant=="corrupted" else QUERIES[variant]
    return "\nExamples for unrelated schemas; answer the actual user's question using its schema:\n"+"\n".join(
        f"Schema:{schema}\nQuestion:{question}\nSQL:{sql}" for schema,question,sql in zip(SCHEMAS,QUESTIONS,queries))


def main():
    connection=sqlite3.connect(":memory:")
    connection.executescript("CREATE TABLE Dept(id,name);CREATE TABLE Emp(id,dept_id,salary);INSERT INTO Dept VALUES(1,'Tech'),(2,'HR');INSERT INTO Emp VALUES(1,1,150),(2,2,80);CREATE TABLE Product(id,name);CREATE TABLE Item(sale_id,product_id,qty);CREATE TABLE Sale(id,status);INSERT INTO Product VALUES(1,'Widget'),(2,'Gadget');INSERT INTO Sale VALUES(1,'paid'),(2,'pending');INSERT INTO Item VALUES(1,1,3),(1,2,1),(2,2,5);")
    for variant,queries in QUERIES.items():
        assert set(connection.execute(queries[0]).fetchall())=={('Tech',)},variant
        assert set(connection.execute(queries[1]).fetchall())=={('Widget',)},variant
    connection.close()
    tokenizer=AutoTokenizer.from_pretrained(str(base_dir()))
    path=ROOT/"wsp/practice.jsonl"
    rows={r["id"]:r for r in map(json.loads,path.read_text().splitlines())}
    ids=list(dict.fromkeys(r["original_id"] for r in json.loads((HERE/"instruction_probe_metadata.json").read_text())))
    new,metadata=[],[]
    for variant in QUERIES:
        for pid in ids:
            row=dict(rows[pid]);row["id"]=f"zicl_{variant}_{pid}"
            row["messages"]=[dict(m) for m in row["messages"]]
            row["messages"][0]["content"]+=examples(variant)
            tokens=len(tokenizer.apply_chat_template(row["messages"],tokenize=True,add_generation_prompt=True))
            assert tokens<=3584,(variant,pid,tokens)
            row["probe_only"]=True
            new.append(row);metadata.append({"id":row["id"],"original_id":pid,"variant":variant})
    assert not any(r["id"] in rows for r in new)
    with path.open("a") as f:
        for row in new:f.write(json.dumps(row)+"\n")
    (HERE/"icl_probe512.txt").write_text("\n".join(r["id"] for r in new)+"\n")
    (HERE/"icl_probe_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    read_ids=list(dict.fromkeys(r["original_id"] for r in json.loads((HERE/"style_read_metadata.json").read_text())))
    texts,meta=[],[]
    for pid in read_ids:
        for phase in ("first","sql"):
            for variant in ("original","corrupted",*QUERIES):
                messages=[dict(m) for m in rows[pid]["messages"]]
                if variant!="original":messages[0]["content"]+=examples(variant)
                prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                if phase=="sql":prompt+="```sql\n"
                label=f"icl_{phase}_{pid}_{variant}"
                texts.append({"id":label,"prompt":prompt,"answer":""})
                meta.append({"id":label,"original_id":pid,"phase":phase,"variant":variant})
    (HERE/"icl_reads384.jsonl").write_text("".join(json.dumps(r)+"\n" for r in texts))
    (HERE/"icl_read_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print("Validated synthetic SQL, prepared",len(new),"diagnostic generations and",len(texts),"controlled reads.")


if __name__=="__main__":main()
