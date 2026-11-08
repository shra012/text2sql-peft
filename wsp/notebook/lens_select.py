"""What does the model consider instead of the T-alias right after SELECT (and at first-token / verbosity)?"""
import sys,torch,collections
sys.path.insert(0,'wsp/notebook')
from lens_alias import lens, tok, D, R, T_ID
SEL=tok.convert_tokens_to_ids('SELECT'); PT=tok.convert_tokens_to_ids('(T')
agg=collections.Counter(); pT=[]; first=collections.Counter(); p_tick=[]
for i in range(len(D['ids'])):
    toks=D['tokens'][i]; pos=D['positions'][i]; P0=int(pos[0])+1; ans=toks[P0:].tolist()
    L0=lens(D['resid:27'][i][0:1])[0].exp(); p_tick.append(float(L0[73594]))
    for t,v in zip(L0.topk(5).indices,L0.topk(5).values): first[tok.decode([int(t)])]+=float(v)
    for j in range(len(ans)-1):
        if ans[j]==SEL and ans[j+1]==T_ID:
            L=lens(D['resid:27'][i][P0+j-P0+0+ (j):][:0]) if False else None
            idx=int((pos==P0+j).nonzero()[0]); L=lens(D['resid:27'][i][idx:idx+1])[0].exp()
            pT.append(float(L[T_ID]))
            for v,t in zip(L.topk(8).values,L.topk(8).indices):
                if t!=T_ID: agg[tok.decode([int(t)])]+=float(v)
            break
print('P(" T" | SELECT) mean %.3f n=%d'%(sum(pT)/len(pT),len(pT)), 'quantiles', torch.tensor(pT).quantile(torch.tensor([.1,.5,.9])))
print([(k,round(v/len(pT),3)) for k,v in agg.most_common(20)])
print('first token mass', [(k,round(v/len(D["ids"]),3)) for k,v in first.most_common(12)])
pt=torch.tensor(p_tick); print('P(```) at first position: mean %.3f, frac<0.5 %.3f, frac<0.3 %.3f'%(pt.mean(),(pt<.5).float().mean(),(pt<.3).float().mean()))
