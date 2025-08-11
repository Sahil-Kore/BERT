import os
os.chdir("../")

from Tokenizer.BasicTokenizer import BasicTokenizer
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from Model.TextDataLoader import EmailDataset,collate_fn
from torch.utils.data import DataLoader
import random
from torch.utils.data import Dataset,random_split
import math

@dataclass
class BERT_Config:
    block_size:int=512
    n_layer:int=2
    n_head:int=8
    n_embd:int=512
    n_outputs:int=2
    cls_token_id:int=2046
    pad_token_id:int=2047
    vocab_size=2048
    dropout:float=0.2


class NonCausalAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head==0
        
        self.attention=nn.Linear(config.n_embd,3 * config.n_embd)
        self.projection=nn.Linear(config.n_embd,config.n_embd)
        self.projection.SCALE_INIT=1
        self.n_head=config.n_head
        self.n_embd=config.n_embd
    
    def forward(self,x):
        B,T,C=x.size()
        
        qkv=self.attention(x)
        q,k,v=qkv.split(self.n_embd,dim=2)
        
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        
        y=F.scaled_dot_product_attention(q,k,v,is_causal=False)
        y=y.transpose(1,2).contiguous().view(B,T,C)
        y=self.projection(y)
        return y
        
class FeedForwardNN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.classifier = nn.Linear(config.n_embd, 4* config.n_embd)
        self.gelu=nn.GELU()
        self.projection=nn.Linear(config.n_embd * 4,config.n_embd)
        self.droput=nn.Dropout(config.dropout)
        self.projection.SCALE_INIT=1
    
    def forward(self,x):
        x=self.classifier(x)
        x=self.gelu(x)
        x=self.projection(x)
        x=self.droput(x)
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
        self.dropout=nn.Dropout(config.dropout)
        self.transformer=nn.ModuleDict(dict(
            token_embeddings=nn.Embedding(config.vocab_size,config.n_embd,padding_idx=config.pad_token_id),
            positional_embeddings=nn.Embedding(config.block_size,config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            layer_norm=nn.LayerNorm(config.n_embd)
        ))
        
        self.classifier=nn.Linear(config.n_embd,config.n_outputs)
        self.apply(self.init_weights)
    
    def init_weights(self,module):
        if isinstance(module,nn.Linear):
            std=0.02
            if hasattr(module, 'SCALE_INIT'):
                std*=(2* self.config.n_layer)** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            
        elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
    
    def forward(self,idx,targets=None):
        B,T= idx.size()
        assert T<=self.config.block_size, f"Cannot forward sequence of length {T}"
        pos=torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_emb=self.transformer.positional_embeddings(pos)
        tok_emb=self.transformer.token_embeddings(idx)
        x=tok_emb+pos_emb
        x=self.dropout(x)
        
        for block in self.transformer.h:
            x=block(x)
        
        x=self.transformer.layer_norm(x)
        logits=self.classifier(x[:,0,:])
        loss=None
        if targets is not None:
            loss = F.cross_entropy(logits,targets)
        return logits,loss

with open("./Training_Data/tokenizer.pkl" , "rb") as f:
    tokenizer= pickle.load(f)



example='''Your college is close to out neighbourhood please look at it'''

inputs=tokenizer.encode_text(example)
model=BERT(BERT_Config)

checkpoint=torch.load("./Model/Models/BERT2.pt",map_location=torch.device("cpu"))
state_dict=checkpoint['model_state_dict']

from collections import OrderedDict

new_state_dict=OrderedDict()

for k,v in state_dict.items():
    new_key= k.replace("_orig_mod.","")
    new_state_dict[new_key]=v
model.load_state_dict(state_dict=new_state_dict)

model.eval()
with torch.inference_mode():
    inputs=torch.tensor(inputs)
    inputs.size()
    prediction,_=model(inputs.unsqueeze(0))
    print(prediction)
    print(prediction[0].argmax(0))
    
    
import requests 
import json

data = {"input_str" : "This is a normal inbox email"}
url= "https://bert-jdbw.onrender.com/predict"
data = json.dumps(data)
response = requests.post(url,json = data)
print(response.json())