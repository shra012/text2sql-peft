"""Is the eventual FROM order ('plan') decodable at the first SELECT alias-digit position better than the model's own
digit choice? Label y=1 if the table owning the first selected column is listed first (so the right digit is '1')."""
import json,re,sqlite3,torch,collections
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
from t2s.wsp.kernel import load
tok=AutoTokenizer.from_pretrained(str(base_dir()))
D=torch.load('wsp/passes/pass_03/read.pt'); E=load('model.embed_tokens.weight')
P={json.loads(l)['id']:json.loads(l) for l in open('wsp/practice.jsonl')}
R={json.loads(l)['id']:json.loads(l) for l in open('wsp/passes/pass_01/original.jsonl')}
T_ID=tok.convert_tokens_to_ids('ĠT'); PT=tok.convert_tokens_to_ids('(T'); D1,D2=16,17
def cols(db,t):
    con=sqlite3.connect(f"file:{db}?mode=ro",uri=True); return [r[1].lower() for r in con.execute(f'PRAGMA table_info("{t}")')]
X=collections.defaultdict(list); Y=[]; M=[]; dig=[]; ids=[]
for k,tid in enumerate(D['ids']):
    if not tid.startswith('F_'): continue
    pid=tid[2:]; s=R[pid]['pred_sql']
    m=re.search(r'FROM\s+(\w+)\s+AS\s+T1\s+(?:INNER\s+)?JOIN\s+(\w+)\s+AS\s+T2\s', s)
    if not m or re.search(r'AS\s+T3',s): continue
    sel=re.match(r'\s*SELECT\s+(?:DISTINCT\s+)?(?:\w+\(\s*)*(?:DISTINCT\s+)?T(\d)\.`?"?(\w+)',s)
    if not sel: continue
    c=sel.group(2).lower(); db=P[pid]['db_path']; in1=c in cols(db,m.group(1)); in2=c in cols(db,m.group(2))
    if in1==in2: continue
    toks=D['tokens'][k]; pos=D['positions'][k]
    j=[i for i,p in enumerate(pos.tolist()) if p+1<len(toks) and int(toks[p]) in (T_ID,PT) and int(toks[p+1]) in (D1,D2)]
    if not j: continue
    j=j[0]
    for L in (19,21,23,25,26,27): X[L].append(D[f'resid:{L}'][k][j].float())
    lo=D['resid:27'][k][j].float()@E.T; M.append(float(lo[D1]-lo[D2])); dig.append(int(sel.group(1))); Y.append(1 if in1 else 0); ids.append(pid)
Y=torch.tensor(Y); M=torch.tensor(M); dig=torch.tensor(dig)
print('n',len(Y),'frac owner-first',Y.float().mean().item())
print('model digit agrees with plan (digit=1 <=> owner first): %.3f'%((dig==1)==(Y==1)).float().mean())
print('accuracy of written digit relative to its table order, i.e. no swap of first alias: same as above')
# leave-one-out diff-of-means (LDA w/ ridge) probe
for L,xs in X.items():
    Xs=torch.stack(xs); n=len(Y); correct=0
    for i in range(n):
        tr=torch.ones(n,dtype=bool); tr[i]=False
        Xt,Yt=Xs[tr],Y[tr]; mu1=Xt[Yt==1].mean(0); mu0=Xt[Yt==0].mean(0)
        C=torch.cov(Xt.T)+0.1*torch.eye(Xs.shape[1])*torch.cov(Xt.T).diagonal().mean()
        w=torch.linalg.solve(C,mu1-mu0); b=w@(mu1+mu0)/2
        correct+=int(((Xs[i]@w-b)>0)==(Y[i]==1))
    print(f'L{L} LOO LDA probe accuracy for plan: {correct/n:.3f}')
torch.save({'X':{L:torch.stack(v) for L,v in X.items()},'Y':Y,'M':M,'dig':dig,'ids':ids},'wsp/notebook/plan_probe_data.pt')
