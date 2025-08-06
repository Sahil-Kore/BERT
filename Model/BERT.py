import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math
@dataclass
class BERT_Config:
    block_size:int=512
    n_layer:int=12
    n_head:int=12
    n_embd:int=768 
    n_outputs:int=2
    cls_token_id:int=261
    pad_token_id:int=262
    

class NonCausalAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head==0
        
        self.attention=nn.Linear(config.n_embd,3 * config.n_embd)
        self.projection=nn.Linear(config.n_embd,config.n_embd)
        self.n_head=config.n_head
        self.n_embd=config.n_embd
    
    def forward(self,x):
        B,T,C=x.size()
        
        qkv=self.attention(x)
        q,k,v=qkv.split(self.n_embd,dim=2)
        
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        
        att= q @ k.transpose(-2,-1) * (1.0/math.sqrt(k.size(-1)))
        att=F.softmax(att,dim=-1)
        y= att @ v
        y=y.transpose(1,2).contiguous().view(B,T,C)
        y=self.projection(y)
        return y
        
class FeedForwardNN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.classifier = nn.Linear(config.n_embd, 4* config.n_embd)
        self.gelu=nn.GELU()
        self.projection=nn.Linear(config.n_embd * 4,config.n_embd)
    
    def forward(self,x):
        x=self.classifier(x)
        x=self.gelu(x)
        x=self.projection(x)
        return x    
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layer_norm1=nn.LayerNorm(config.n_embd)
        self.attn=NonCausalAttention(config)
        self.layer_norm2=nn.LayerNorm(config.n_embd)
        self.ffn=FeedForwardNN(config)
        
    def forward(self,x):
        x = x + self.attn(self.layer_norm1(x))
        x = x + self.ffn(self.layer_norm2(x))
        return x       
    
    
class BERT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.transformer=nn.ModuleDict(dict(
            token_embeddings=nn.Embedding(config.vocab_size,config.n_embd,padding_idx=config.pad_token_id),
            positional_embeddings=nn.Embedding(config.block_size,config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            layer_norm=nn.LayerNorm(config.n_embd)
        ))
        
        self.classifier=nn.Linear(config.n_embd,config.n_ouputs)
    
    def forward(self,idx,targets=None):
        B,T= idx.size()
        assert T<self.config.block_size, f"Cannot forward sequence of length {T}"
        pos=torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_emb=self.transformer.positional_embeddings(pos)
        tok_emb=self.transformer.token_embeddings(idx)
        
        x=tok_emb+pos_emb
        
        for block in self.transformer.h:
            x=block(x)
        
        x=self.transformer.layer_norm(x)
        logits=self.classifier(x[:,0,:])
        loss=None
        if targets is not None:
            loss = F.cross_entropy(logits,targets)
        return logits,loss

