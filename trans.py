import torch.nn
import transformers
from transformers import pipeline
import pandas as pd

text = ""
with open("test2.txt", "rt") as fd:
    sb = ""
    for s in fd:
        sb = sb + s

    text = sb[:1024*8]


#classifier = pipeline("text-classification")
classifier = pipeline("summarization", model="facebook/bart-large")



outputs = classifier(text)
print (classifier.model)
print (classifier.tokenizer)
print(outputs)




