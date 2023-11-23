from datasets import load_dataset, load_metric,config
import sys, logging
import torch
from pytorch_memlab import profile, set_target_gpu, MemReporter
from pynvml import *

reporter = MemReporter()
reporter.report()

#torch.backends.cudnn.benchmark=False

from transformers.models.mbart.tokenization_mbart_fast import MBartTokenizer

logging.basicConfig(stream=sys.stdout, encoding='utf-8', format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO)

#raw_datasets = load_dataset("wmt16", "de-en")
raw_datasets = load_dataset('IlyaGusev/gazeta')
import sacrebleu, nltk


prefix = "" #for mBART and MarianMT
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "de"

model_mbart = "IlyaGusev/mbart_ru_sum_gazeta"

import torch
torch.cuda.empty_cache()

from transformers import MBart50TokenizerFast, Seq2SeqTrainer

tokenizer = MBartTokenizer.from_pretrained(model_mbart,src_lang="ru_RU",tgt_lang = "ru_RU")

def preprocess_function(examples):
   inputs = examples["text"]
   targets = examples["summary"]
   model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

   # Setup the tokenizer for targets

   with tokenizer.as_target_tokenizer():
       labels = tokenizer(targets, max_length=max_target_length, truncation=True)

   model_inputs["labels"] = labels["input_ids"]
   return model_inputs

#tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

#small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(3000))
#small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(3000))

small_train_dataset = raw_datasets["train"].shuffle(seed=42).select(range(3000))
small_eval_dataset = raw_datasets["test"].shuffle(seed=42).select(range(3000))

small_train_dataset = small_train_dataset.map(preprocess_function, batched=True)
small_eval_dataset = small_eval_dataset.map(preprocess_function, batched=True)


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
   num_train_epochs=5,
   predict_with_generate=True,
   fp16=True,
   metric_for_best_model="loss"

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


optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)

trainer = Seq2SeqTrainer(
   model,
   args,
   train_dataset=small_train_dataset,
   eval_dataset=small_eval_dataset,
   data_collator=data_collator,
   tokenizer=tokenizer,
   compute_metrics=compute_metrics,
   optimizers=[optimizer, scheduler]

)
trainer.train()


torch.save(model.state_dict(),"my_model2.pt")