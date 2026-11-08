import json,torch,sys,collections
sys.path.insert(0,'wsp/notebook')
from attnlib import *
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
tok=AutoTokenizer.from_pretrained(str(base_dir()))
D=torch.load('wsp/passes/pass_05/read.pt')
Tx={json.loads(l)['id']:json.loads(l) for l in open('wsp/notebook/texts_pass05.jsonl')}
for L,h in [(22,7),(23,4),(23,3)]:
    agg=collections.Counter(); where=collections.Counter()
    for n,tid in enumerate(D['ids'][:64]):
        toks=D['tokens'][n]; nans=len(tok(Tx[tid]['answer'],add_special_tokens=False)['input_ids']); p=len(toks)-nans-1
        _,a=attn_heads(L,D[f'resid:{L-1}'][n].float(),[p]); a=a[0,h]
        # find start of assistant answer
        ast=len(tok(Tx[tid]['prompt'].split('<|im_start|>assistant\n')[0]+'<|im_start|>assistant\n',add_special_tokens=False)['input_ids'])
        where['answer-part']+=float(a[ast:].sum()); where['prompt']+=float(a[:ast].sum())
        top=a.topk(4)
        if n<4:
            print(f'L{L}H{h}',tid, [(tok.decode(toks[max(0,int(t)-2):int(t)+1]).replace("\n","\\n"), round(float(v),2), 'ans' if t>=ast else 'prm') for v,t in zip(top.values,top.indices)])
    print(f'L{L}H{h} mass: answer part {where["answer-part"]/64:.2f}, prompt {where["prompt"]/64:.2f}')
