from datasets import load_dataset, load_metric,config
raw_datasets = load_dataset("wmt16", "de-en")
import sacrebleu, nltk


prefix = "" #for mBART and MarianMT
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "de"

model_mbart = 'facebook/mbart-large-50-one-to-many-mmt'

import torch
torch.cuda.empty_cache()

from transformers import MBart50TokenizerFast, Seq2SeqTrainer

tokenizer = MBart50TokenizerFast.from_pretrained(model_mbart,src_lang="en_XX",tgt_lang = "de_DE")

def preprocess_function(examples):
   inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
   targets = [ex[target_lang] for ex in examples["translation"]]
   model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

   # Setup the tokenizer for targets

   with tokenizer.as_target_tokenizer():
       labels = tokenizer(targets, max_length=max_target_length, truncation=True)

   model_inputs["labels"] = labels["input_ids"]
   return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))



from transformers import MBartForConditionalGeneration, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

model = MBartForConditionalGeneration.from_pretrained(model_mbart)
batch_size = 4

args = Seq2SeqTrainingArguments(
   f"{model_mbart}-finetuned-{source_lang}-to-{target_lang}",
   evaluation_strategy = "epoch",
   learning_rate=2e-5,
   per_device_train_batch_size=batch_size,
   per_device_eval_batch_size=batch_size,
   weight_decay=0.01,
   save_total_limit=3,
   num_train_epochs=1,
   predict_with_generate=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


import numpy as np
import evaluate
metric = evaluate.load("sacrebleu")
#meteor = evaluate.load('meteor')

def postprocess_text(preds, labels):
   preds = [pred.strip() for pred in preds]
   labels = [[label.strip()] for label in labels]
   return preds, labels

def compute_metrics(eval_preds):
   preds, labels = eval_preds
   if isinstance(preds, tuple):
       preds = preds[0]
   decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
   # Replace -100 in the labels as we can't decode them.
   labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
   decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
   # Some simple post-processing
   decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
   result = metric.compute(predictions=decoded_preds, references=decoded_labels)
  # meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)
   prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
   result = {'bleu' : result['score']}
   result["gen_len"] = np.mean(prediction_lens)
   #result["meteor"] = meteor_result["meteor"]
   result = {k: round(v, 4) for k, v in result.items()}
   return result


trainer = Seq2SeqTrainer(
   model,
   args,
   train_dataset=small_train_dataset,
   eval_dataset=small_eval_dataset,
   data_collator=data_collator,
   tokenizer=tokenizer,
   compute_metrics=compute_metrics
)
trainer.train()