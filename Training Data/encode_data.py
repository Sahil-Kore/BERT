import os
os.chdir("../")
os.getcwd()
from Tokenizer.BasicTokenizer import BasicTokenizer
tokenizer=BasicTokenizer("../Data Extraction/Inbox","../Data Extraction/Spam")

tokenizer.train(64)
tokenizer.encode("./Data Extraction/Inbox","./Training Data/Inbox")
tokenizer.encode("./Data Extraction/Spam","./Training Data/Spam")