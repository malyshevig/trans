from transformers import BartTokenizerFast, BartModel


def readfile(fname: str):
    with open(fname, "rt") as fd:
        sb = ""
        for s in fd:
            sb = sb + s

    return sb


s = readfile("test2.txt")
tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
t=tokenizer(s, return_tensors="pt")

model = BartModel.from_pretrained('facebook/bart-base')
r = model(**t)


print(r)