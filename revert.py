import torch.cuda
from transformers import MBartTokenizer, MBartForConditionalGeneration


def summary(text:str):
    device = torch.device('cuda')

    model_name = "IlyaGusev/mbart_ru_sum_gazeta"
    tokenizer = MBartTokenizer.from_pretrained(model_name)

    model = MBartForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    import util
    import sentencepiece

    article_text = text

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

    _summary = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(_summary)

    return _summary


