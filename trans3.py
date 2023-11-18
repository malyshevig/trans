
from transformers import MBartTokenizer, MBartForConditionalGeneration

device = torch.device('cuda')

model_name = "IlyaGusev/mbart_ru_sum_gazeta"
tokenizer = MBartTokenizer.from_pretrained(model_name)

model = MBartForConditionalGeneration.from_pretrained(model_name)
model.to(device)

import util
import sentencepiece

article_text = util.readfile("text3.txt")

input_ids = tokenizer(
    [article_text],
    max_length=600,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)["input_ids"].to(device)

output_ids = model.generate(
    input_ids=input_ids,
    no_repeat_ngram_size=4
)[0]

summary = tokenizer.decode(output_ids, skip_special_tokens=True)
print(summary)

print (torch.cuda.is_available())