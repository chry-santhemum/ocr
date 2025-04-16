#!/usr/bin/env python3
import os
import json
from typing import List, Optional
import torch
import gc
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    get_linear_schedule_with_warmup,
)
from trl import SFTTrainer, SFTConfig
import wandb
from peft import LoraConfig, get_peft_model


# Set a fixed seed for reproducibility
set_seed(42)
ds_path = "inductive-oocr/functions/dev/047_functions/finetune_01"

# %%

def load_functions_dataset(path):
    # each row: {"messages": [message dicts]}
    # this doesn't need any additional preprocessing with SFTTrainer
    ds = []
    with open(path, 'r') as f:
        for line in f:
            ds.append(json.loads(line))

    # need to cut out the system message because it's not supported
    for message in ds:
        sys_message = message["messages"][0]["content"]
        message["messages"].pop(0)
        message["messages"][0]["content"] = sys_message + "\n" + message["messages"][0]["content"]
    
    dataset = Dataset.from_list(ds)
    return dataset


if __name__ == "__main__":
    import argparse

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Set up CUDA and distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    
    # Load model with device_map for optimal placement
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        torch_dtype=torch.bfloat16,  # Use BF16 for better numerical stability
        device_map="auto",
        attn_implementation='eager',
        use_auth_token=True,
    )

    # If MLP only, detach everything but the MLP layers specified in the list

    params_tracked = []
    # if mlp_only is not None:
    #     if mlp_only == "all":
    #         mlp_only = [i for i in range(model.config.num_hidden_layers)]
    #     for name, param in model.named_parameters():
    #         if "mlp" in name and any("."+str(num)+"." in name for num in mlp_only):
    #             params_tracked.append(param)
    #             print(f"Unfreezing {name}")
    # else:
    #     for name, param in model.named_parameters():
    #         if "embed_tokens" in name and freeze_W_E:
    #             continue
    #         else:
    #             params_tracked.append(param)

    output_dir = f'./checkpoints/2b-functions-mlp-full/'
    for name, param in model.named_parameters():
        if f".mlp." in name:
            params_tracked.append(param)


    # Get training dataset
    train_dataset = load_functions_dataset(os.path.join(ds_path, "047_func_01_train_oai.jsonl"))

    run = wandb.init(
        project="oocr",
        name=output_dir[14:],
        config={
            "model": "gemma-2-2b-it",
            "learning_rate": 1e-5,
            "task": "functions",
            "epochs": 1,
        },
    )

    # Set up training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,  # Increased further to reduce memory pressure
        learning_rate=2e-5,
        max_steps=4000,
        warmup_steps=50,
        save_strategy="steps", # only save each epoch
        save_steps=1000,
        logging_steps=10,
        num_train_epochs=1,
        bf16=True,           # Use BF16 mixed precision
        fp16=False,          # Disable FP16 training
        gradient_checkpointing=True,  # Trade compute for memory
    )

    optim = torch.optim.AdamW(params_tracked, lr=training_args.learning_rate)

    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.max_steps)

    
    # Set up trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optim, sched),
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    run.finish()
    
    # Save the final model
    # trainer.save_model()
    # tokenizer.save_pretrained(args.output_dir)
    # print(f"Training complete. Model saved to {args.output_dir}")
# %%
