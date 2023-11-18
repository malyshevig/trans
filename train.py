import evaluate
from transformers import TrainingArguments, Trainer, MBartTokenizer, MBartForConditionalGeneration, \
    DataCollatorForLanguageModeling,DataCollatorForSeq2Seq, BatchEncoding, Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np
import torch



import datasets as ds
from transformers import AutoModelForSequenceClassification
device = "cpu"

dataset = ds.load_dataset('IlyaGusev/gazeta',split="train")
model_name = "IlyaGusev/mbart_ru_sum_gazeta"
tokenizer = MBartTokenizer.from_pretrained(model_name)

#model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


output = "output"



training_args = Seq2SeqTrainingArguments(
    output_dir="model",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False
)





class MyDataset (torch.utils.data.Dataset):

    def tokenize (self, text, expected_text):
        ids = tokenizer(
            text,
            text_target=expected_text,
            max_length=600,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True
        )

        return ids.to(device)

    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, idx):
        text = self.ds[idx]["text"]
        summary = self.ds[idx]["summary"]

        ids = self.tokenize(text,summary)
        return ids.to(device)

    def __len__(self):
        return len(self.ds)

# Metric Id
metric = evaluate.load("f1")

# Metric helper method
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")


trainer = Seq2SeqTrainer (
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=MyDataset(dataset)
)
model.to(device)
trainer.train()
trainer.save_model('shakespere_BART')