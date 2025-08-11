from fastapi import FastAPI
import pickle 
import torch
from bert_architecture import BERT,BERT_Config
from Tokenizer.BasicTokenizer import BasicTokenizer
import os
import gdown

model=BERT(BERT_Config)

idx_to_class={
    0: "Inbox",
    1: "Spam"
}
#load the model and tokenizer from drive
#convert the link https://drive.google.com/file/d/1l3tyFlNDtYXJYRmMRV8bPcIrrhdbZJeK/view?usp=drive_link to the below form

model_path="./Model/Models/BERT2.pt"
os.makedirs(os.path.dirname(model_path) , exist_ok=True)
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?export=download&id=1FSJrK-457QVoErlni_SiJLXcCcrbboGI"
    gdown.download(url , model_path , quiet=False)

tokenizer_path="./Training_Data/tokenizer.pkl"
os.makedirs(os.path.dirname(tokenizer_path) , exist_ok=True)
if not os.path.exists(tokenizer_path):
    url = "https://drive.google.com/uc?export=download&id=1l3tyFlNDtYXJYRmMRV8bPcIrrhdbZJeK"
    gdown.download(url , tokenizer_path , quiet=False)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint=torch.load("./Model/Models/BERT2.pt",map_location=device)
state_dict=checkpoint['model_state_dict']

from collections import OrderedDict

new_state_dict=OrderedDict()

for k,v in state_dict.items():
    new_key= k.replace("_orig_mod.","")
    new_state_dict[new_key]=v
model.load_state_dict(state_dict=new_state_dict)

with open("./Training_Data/tokenizer.pkl","rb") as f:
    tokenizer=pickle.load(f)
   
app=FastAPI()

@app.get('/')
def reed_root():
    return {'message' : 'BERT model'}

@app.post('/predict')
def predict(data:str):
    tokens = tokenizer.encode_text(data)
    tokens=torch.tensor(tokens).to(device)
    model.eval()
    with torch.inference_mode():
        prediction, _ = model(tokens.unsqueeze(0))
    prediction = prediction[0].argmax(0)
    prediction= idx_to_class[prediction.item()]
    return {'Prediction_class ' :  prediction}
