from transformers import BartTokenizerFast, BartModel
import evaluate

def readfile(fname: str):
    with open(fname, "rt") as fd:
        sb = ""
        for s in fd:
            sb = sb + s

    return sb


s = readfile("test2.txt")
tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large", truncation_side = "right")
t=tokenizer(s, return_tensors="pt",truncation=True)

model = BartModel.from_pretrained('facebook/bart-large')
r = model(**t)


print(model)