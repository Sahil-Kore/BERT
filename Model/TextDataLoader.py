import os 
import torch
from torch.utils.data import Dataset,DataLoader
from pathlib import Path

class EmailDataset(Dataset):
    def __init__(self, inbox_dir, spam_dir):
        
        #list of (input,label)
        self.data = []
        self._load_files(inbox_dir, label=0)
        self._load_files(spam_dir, label=1)
    
    def _load_files(self, folder_path ,label):
        folder_path=Path(folder_path)
        for file in folder_path.glob('*.txt'):
            with open(file,"r") as f :
                tokens=f.read().strip().split()
                token_ids =torch.tensor([int(tok) for tok in tokens],dtype=torch.long)
                self.data.append((token_ids,label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]

def collate_fn(batch):
    inputs,labels=zip(*batch)
    inputs=torch.stack(inputs) 
    labels=torch.tensor(labels, dtype=torch.long)
    return inputs ,labels

if __name__=='__main__':
    dataset=EmailDataset("../Training Data/Inbox","../Training Data/Spam")
    dataset[-1]
    
    loader=DataLoader(dataset,batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    for batch_inputs , batch_labels in loader:
        print(batch_inputs.shape)
        print(batch_labels)
        
        