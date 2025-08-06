import unicodedata
from pathlib import Path
import os
def getstats(dir1,dir2):
    dir1=Path(dir1)
    dir2=Path(dir2)
    all_files=list(dir1.glob('*'))+list(dir2.glob('*'))
    stats={}
    
    for file in all_files:
        with open(file,"r") as f:
            tokens=f.read().split()
        
        for pair in zip(tokens,tokens[1:]):
            stats[pair]=stats.get(pair,0)+1
    
    return stats


def merge(pair,idx,dir1=None,dir2=None,file=None):
    all_files=[]
    if file:
        all_files=[file]
    else:
        dir1=Path(dir1)
        dir2=Path(dir2)
        all_files=list(dir1.glob('*'))+ list(dir2.glob('*'))
    
    for file in all_files:
        with open(file,'r')as f:
            tokens=f.read()
        i=0
        while i<len(tokens)-1:
            curr_pair=(tokens[i],tokens[i+1])
            if curr_pair ==  pair:
                tokens[i:i+2]=idx
        i+=1
        encoded_string=" ".join(map(str,tokens))
        with open(file,"w") as f:
            f.write(encoded_string)
    

    
        
        