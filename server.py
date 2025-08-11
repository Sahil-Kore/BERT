from fastapi import FastAPI
import pickle 
import torch
from bert_architecture import BERT,BERT_Config
from Tokenizer.BasicTokenizer import BasicTokenizer

model=BERT(BERT_Config)

idx_to_class={
    0: "Inbox",
    1: "Spam"
}

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
