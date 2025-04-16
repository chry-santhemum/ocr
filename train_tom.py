import json
import wandb
import torch
import transformers
import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# clear garbage
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# load dataset
ds = []

with open("train_data.jsonl", 'r') as f:
    for line in f:
        ds.append(json.loads(line))

train_dataset = Dataset.from_list(ds)

model_name = "google/gemma-2-9b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
    attn_implementation="eager")

# tokenize the dataset.
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=100)

tokenized_dataset = train_dataset.map(preprocess_function, batched=True)

# Set up training arguments.
training_args = TrainingArguments(
    output_dir="./gemma-2-9b-oocr",
    run_name="test",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    bf16=True,
    fp16=False,
    max_steps=10,
    learning_rate=1e-5,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    eval_strategy="no",
    gradient_checkpointing=True,
)

# Data collator for causal language modeling (SFT) without masked LM.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

run = wandb.init(
    project="oocr",
    config={
        "model": "gemma-2-9b",
        "learning_rate": 1e-5,
        "epochs": 3,
    },
)

# Initialize the Trainer.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Start training.
trainer.train()
