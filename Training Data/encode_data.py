import sys
import os 
os.chdir("../")
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Tokenizer.BasicTokenizer import BasicTokenizer
from pathlib import Path

def add_pad_cls(dir1=None,dir2=None,file=None):
    cls_id='2046'
    pad_id='2047'
    all_files=[]
    if file:
        all_files=[file]
    else:
        dir1=Path(dir1)
        dir2=Path(dir2)
        
        all_files=list(dir1.glob('*')) +list(dir2.glob('*'))
    
    for file in all_files:
        with open(file ,"r") as f:
            tokens=f.read().split()
        zero_index=tokens[0]
        tokens = [cls_id,zero_index]+tokens[1:]
        if len(tokens)>512:
            tokens=tokens[:512]
        elif len(tokens)<512:
            n_pad=512-len(tokens)
            tokens=tokens + [pad_id]*n_pad
        
        #save file
        with open(file,"w") as f:
            f.write(" ".join(tokens))
#encode the data in utf-8
dir1="./Data Extraction/Inbox"
dir2="./Data Extraction/Spam"
dir1=Path(dir1)
dir2=Path(dir2)
os.makedirs("./Training Data/Inbox",exist_ok=True)
os.makedirs("./Training Data/Spam",exist_ok=True)

inbox_output_dir="./Training Data/Inbox"
spam_output_dir="./Training Data/Spam"


inbox_files=list(dir1.glob('*'))
for file in inbox_files:
    with open(file,"r") as f:
        content=f.read()
    utf_encoded=list(content.encode('utf-8'))
    encoded_string=" ".join(map(str,utf_encoded))   
    output_path=os.path.join(inbox_output_dir,file.name)
    with open(output_path,"w") as f:
        f.write(encoded_string) 
        
spam_files=list(dir2.glob('*'))
for file in spam_files:
    with open(file,"r") as f:
        content=f.read()
    utf_encoded=list(content.encode('utf-8'))
    encoded_string=" ".join(map(str,utf_encoded))   
    output_path=os.path.join(spam_output_dir,file.name)
    with open(output_path,"w") as f:
        f.write(encoded_string)     

tokenizer=BasicTokenizer("./Training Data/Inbox","./Training Data/Spam")
tokenizer.train(num_merges=1790)
tokenizer.encode("./Data Extraction/Inbox","./Training Data/Inbox")
tokenizer.encode("./Data Extraction/Spam","./Training Data/Spam")
os.chdir("./Training Data")
import pickle
with open ("tokenizer.pkl" ,"wb") as f:
    pickle.dump(tokenizer,f)
    
import pickle
with open ("./Training Data/tokenizer.pkl" ,"rb") as f:
    tok1=pickle.load(f)

#adding padding and cls token
#max_token is 260
add_pad_cls("./Inbox","./Spam")
for file in Path("./Inbox").glob('*'):
    with open(file,"r") as f:
        tokens=f.read().strip().split()
    if len(tokens) !=512:
        print(f"{f.name} {len(tokens)}")
        
        