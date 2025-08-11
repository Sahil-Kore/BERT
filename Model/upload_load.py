import os 
import gdown
import torch

model_path="./Model.pt"
os.makedirs(os.path.dirname(model_path) , exist_ok=True)

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?export=download&id=1FSJrK-457QVoErlni_SiJLXcCcrbboGI"
    gdown.download(url , model_path , quiet=False)

