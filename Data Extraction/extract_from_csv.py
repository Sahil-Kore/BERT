import pandas as pd
import os

df=pd.read_csv("./emails.csv")
spam=df[df['spam']==1]

spam.iloc[1]['text'][9:]
curr_idx=24
for idx in range(len(spam)):
    filename=f"Spam_{idx+curr_idx:05}.txt"
    output_path=os.path.join("./Spam",filename)
    with open(output_path,"w") as f:
        f.write(spam.iloc[idx]['text'][9:])
        
        
from zipfile import ZipFile
with ZipFile("./archive.zip") as f:
    f.extractall()
    

df2=pd.read_csv("./Spam.csv")
spam2=df2[df2["Category"] == "spam"]
spam2

curr_idx=1392
for idx in range(len(spam2)):
    filename=f"Spam_{idx+curr_idx:05}.txt"
    output_path=os.path.join("./Spam",filename)
    with open(output_path,"w") as f:
        f.write(spam2.iloc[idx]['Messages'])
        

        
from zipfile import ZipFile
with ZipFile("./csv files/archive (2).zip") as f:
    f.extractall("./csv files")

df3=pd.read_csv("./csv files/email_classification_dataset.csv")
df3
          
spam3=df3[df3['label']=='spam']
series=spam3['email'].str.extract(r'\n\n(.*)',expand=False)
series.iloc[1]
series.iloc[1][:-8]

curr_idx=2139
for idx in range(len(series)):
    filename=f"Spam_{idx+curr_idx:05}.txt"
    output_path=os.path.join("./Spam",filename)
    with open(output_path,"w") as f:
        f.write(series.iloc[idx][:-8])