import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from TextDataLoader import EmailDataset,collate_fn
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
# Load and balance data
dataset = EmailDataset("../Training Data/Inbox", "../Training Data/Spam")
inbox_dataset = [item for item in dataset if item[1] == 0]
spam_dataset = [item for item in dataset if item[1] == 1]
balanced_data = spam_dataset + inbox_dataset[:len(spam_dataset)]
random.shuffle(balanced_data)

# Create a Dataset wrapper
class BalancedDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# Split into train and validation sets (e.g., 80% train, 20% val)
val_split = 0.2
val_size = int(len(balanced_data) * val_split)
train_size = len(balanced_data) - val_size

train_data, val_data = random_split(balanced_data, [train_size, val_size])
train_dataset = BalancedDataset(train_data)
val_dataset = BalancedDataset(val_data)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)

# Learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 5
max_steps = 20

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# Model setup
torch.cuda.empty_cache()
model = BERT(BERT_Config)
device = "cuda"
model = model.to(device)
model = torch.compile(model)

optimizer = torch.optim.AdamW(params=model.parameters(),weight_decay=0.05)

loss_arr = []
val_loss_arr = []

# Training loop
for step in range(max_steps):
    total_loss = 0.0
    model.train()
    for batch in train_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        total_loss += loss.item()

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            val_loss += loss.item()

    # Logging and checkpointing
    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    loss_arr.append(avg_train_loss)
    val_loss_arr.append(avg_val_loss)


    print(f"Epoch: {step}  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")

loss_arr
model_save_path = f"./Models/BERT1.pt"
torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_arr[-1],
        'config':BERT_Config().__dict__
        }, model_save_path)
print(f"Model saved to {model_save_path}")


# After validation loop is done
full_dataset = BalancedDataset(balanced_data)
full_loader = DataLoader(full_dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)

# Continue training on full dataset
extra_epochs = 5  # or however many more epochs you want
full_loss=[]
for epoch in range(extra_epochs):
    total_loss = 0.0
    model.train()
    for batch in full_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_full_loss = total_loss / len(full_loader)
    full_loss.append(avg_full_loss)
    print(f"Full Training Epoch {epoch + 1}/{extra_epochs}  Loss: {avg_full_loss:.4f}")
model_save_path = f"./Models/BERT2.pt"
torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': full_loss[-1],
        'config':BERT_Config().__dict__
        }, model_save_path)
print(f"Model saved to {model_save_path}")