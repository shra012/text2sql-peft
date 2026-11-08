"""CPU re-implementation of one Qwen2 decoder layer (from captured residuals) for head-level attribution."""
import torch
from t2s.wsp.kernel import current
H,KV,HD,EPS,THETA=12,2,128,1e-6,1e6
_W={}
def W(L,n):
    k=(L,n)
    if k not in _W: _W[k]=current(f'model.layers.{L}.{n}')
    return _W[k]
def rms(x,w): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+EPS)*w
def rope(x,pos):  # x: (h,T,128)
    inv=1.0/(THETA**(torch.arange(0,HD,2).float()/HD)); f=pos[:,None].float()*inv[None]
    emb=torch.cat([f,f],-1); cos,sin=emb.cos(),emb.sin()
    x1,x2=x[...,:HD//2],x[...,HD//2:]; rot=torch.cat([-x2,x1],-1)
    return x*cos+rot*sin
def attn_heads(L,resid_in,qpos):
    """resid_in: (T,1536) input to layer L. Returns per-head residual contributions at query positions qpos:
    (len(qpos),H,1536), and attention probs (len(qpos),H,T)."""
    x=rms(resid_in.float(),W(L,'input_layernorm.weight')); T=x.shape[0]; pos=torch.arange(T)
    q=(x[qpos]@W(L,'self_attn.q_proj.weight').T+W(L,'self_attn.q_proj.bias')).view(len(qpos),H,HD).transpose(0,1)
    k=(x@W(L,'self_attn.k_proj.weight').T+W(L,'self_attn.k_proj.bias')).view(T,KV,HD).transpose(0,1)
    v=(x@W(L,'self_attn.v_proj.weight').T+W(L,'self_attn.v_proj.bias')).view(T,KV,HD).transpose(0,1)
    q=rope(q,torch.as_tensor(qpos)); k=rope(k,pos)
    k=k.repeat_interleave(H//KV,0); v=v.repeat_interleave(H//KV,0)
    s=(q@k.transpose(1,2))/HD**0.5
    mask=pos[None,:]>torch.as_tensor(qpos)[:,None]; s=s.masked_fill(mask[None],-1e9)
    a=s.softmax(-1)                       # (H,Q,T)
    o=a@v                                 # (H,Q,128)
    Wo=W(L,'self_attn.o_proj.weight').view(1536,H,HD)
    contrib=torch.einsum('dhk,hqk->qhd',Wo,o)
    return contrib,a.transpose(0,1)
def mlp(L,resid_mid):
    x=rms(resid_mid.float(),W(L,'post_attention_layernorm.weight'))
    a=torch.nn.functional.silu(x@W(L,'mlp.gate_proj.weight').T)*(x@W(L,'mlp.up_proj.weight').T)
    return a@W(L,'mlp.down_proj.weight').T, a
