import os
from pathlib import Path
class BasicTokenizer:
    #stores the tokens merged as keys and the minted new token as values
    merges={}
    #stores the unicode point o fthe token and its byte representation if the token is minted it stores the parent tokens as a list
    vocab={}
    def __init__(self,dir1,dir2):
        self.dir1=Path(dir1)
        self.dir2=Path(dir2)
    
    def getstats(self,file=None):
        all_files=[]
        if file:
            file=Path(file)
            all_files=[file]
        else:
            all_files = list(self.dir1.glob("*")) + list(self.dir2.glob("*"))
        stats={}
        for file_path in all_files:
            if file_path.is_file():
                with open(file_path,"r") as f:
                    tokens=f.read().split()
                for pair in zip(tokens,tokens[1:]):
                    stats[pair]=stats.get(pair,0)+1
        
        return stats
    
    def merge(self,pair,idx,file=None):
        all_files=[]
        if file:
            file = Path(file)
            if file.is_file():
                all_files = [file]
        else:
            all_files = list(self.dir1.glob("*")) + list(self.dir2.glob("*"))
        for file_path in all_files:
            if file_path.is_file():
                tokens=[]
                with open(file_path,"r") as f:
                    tokens=f.read().split()
                result=list(tokens)
                i=0
                while i<len(result)-1:
                    curr_pair=(result[i],result[i+1])
                    if pair==curr_pair:
                        result[i:i+2]=[idx]
                    i+=1
                encoded_string=" ".join(map(str,result))
                with open(file_path,"w") as f:
                    f.write(encoded_string)
    
    def train(self,num_merges):
        idx=256
        merges={}
        vocab={idx:bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats=self.getstats()
            max_pair=max(stats,key=stats.get)
            idx=256+i
            self.merge(max_pair,idx)
            merges[max_pair]=idx
            vocab[idx]=vocab[int(max_pair[0])]+vocab[int(max_pair[1])]
        self.merges=merges
        self.vocab=vocab
        
    
    def encode(self,dir,output_dir):
        os.makedirs(output_dir,exist_ok=True)
        dir=Path(dir)
        all_files=list(dir.glob("*"))
        
        #first encode all files to utf-8
        for file in all_files:
            with open(file,"r") as f:
                content=f.read()
            utf_encoded=list(content.encode('utf-8'))
            encoded_string=" ".join(map(str,utf_encoded))
            output_path=os.path.join(output_dir,file.name)
            with open(output_path,"w") as f:
                f.write(encoded_string)
        
        #now on the encoded files apply merges according to the merges dictionary
        output_dir=Path(output_dir)
        all_files=list(output_dir.glob("*"))
        
        for file in all_files:
            with open(file,"r") as f:
                content=f.read()
            
            while len(content)>=2:
                stats=self.getstats(file)
                pair=min(stats,key=lambda p:self.merges.get(p,float('inf')))
                #no more tokens to merge
                if pair not in self.merges:
                    break
                
                #merge the token
                self.merge(pair,self.merges[pair],file=file)
                with open(file,"r") as f:
                    content=f.read()
    
    def decode(self,input_dir,output_dir):
        os.makedirs(output_dir,exist_ok=True)
        input_dir=Path(input_dir)
        all_files=list(input_dir.glob("*"))
        
        for file in all_files:
            with open(file,"r") as f:
                content=f.read()
            tokens=list(map(int,content.strip().split()))
            text_bytes=b"".join(self.vocab[idx] for idx in tokens)
            text=text_bytes.decode('utf-8')
            output_file=os.path.join(output_dir,file.name)
            with open(output_file,"w") as f:
                f.write(text)
        
        
                
            
        
        
              
if __name__=="__main__":                
    vocab_size=256
    num_merges=64

    input_dir="../Test/Texts"
    output_dir="./Examples"
    os.makedirs(output_dir,exist_ok=True)
    input_dir=Path(input_dir)
    files=list(input_dir.glob("*"))
    for file in files:
        with open(file,"r") as f:
            content =f.read()
        utf_encoded=list(content.encode('utf-8'))
        encoded_string=" ".join(map(str,utf_encoded))   
        output_path=os.path.join(output_dir,file.name)
        with open(output_path,"w") as f:
            f.write(encoded_string)


    tokenizer=BasicTokenizer("./Examples","./Examples")
    tokenizer.train(num_merges)
    tokenizer.merges
    tokenizer.encode("./ToEncodeFiles","./EncodedFiles")
    tokenizer.decode("./EncodedFiles","./DecodeFiles")
    with open("./ToEncodeFiles/message_00001.txt") as f :
        str1=f.read()
    with open("./DecodeFiles/message_00001.txt") as f :
        str2=f.read()
    str1==str2
    len(str1),len(str2)
    
        