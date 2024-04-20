import datasets 
import numpy as np
from transformers import BertTokenizerFast
from sklearn.ensemble import RandomForestClassifier

conll2003 = datasets.load_dataset("conll2003")

train = conll2003["train"]
val = conll2003["val"]
test = conll2003["test"]

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased") 

print(conll2003)