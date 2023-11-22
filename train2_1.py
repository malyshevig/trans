from transformers import MBartForConditionalGeneration, MBartTokenizer
import util, revert



model_mbart = "IlyaGusev/mbart_ru_sum_gazeta"
article_text = revert.summary(util.readfile("text3.txt"))

tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX", tgt_lang="ro_RO")
example_english_phrase = "UN Chief Says There Is No Military Solution in Syria"
expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"

inputs = tokenizer(example_english_phrase, text_target=expected_translation_romanian, return_tensors="pt")

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
# forward pass

device = "cuda"
inputs.to(device)
model.to(device)

r = model(**inputs)
print (r)