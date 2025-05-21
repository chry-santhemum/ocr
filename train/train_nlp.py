# %%
import json
import wandb
import torch
import gc
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model

import sys
sys.path.append("..")
from utils import (
    clear_cuda_mem,
    set_seed_all,
    print_trainable_params
)
from torch.utils.data import DataLoader
from functools import partial
from trl import SFTConfig, SFTTrainer

# %%

def load_nlp_dataset(path):
    # just one column "text"
    ds = []
    with open(path, 'r') as f:
        for line in f:
            datapoint = json.loads(line)
            ds.append({"text": datapoint["text"]})

    train_ds = Dataset.from_list(ds)
    return train_ds

class CustomEvalCallback(TrainerCallback):
    def __init__(self, quick_test, tokenizer, eval_steps):
        self.quick_test = quick_test
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps == 0:
            print(f"\nRunning evaluation at step {state.global_step}")
            # Run your custom evaluation
            self.quick_test(model, self.tokenizer)
            
            # Log to wandb
            # wandb.log(eval_results, step=state.global_step)
            # print(f"Evaluation results: {eval_results}")
        return control
    
def quick_test(model, tokenizer):
    text = "What's the weather like today? \nPangolin:"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    # logits = model(**inputs).logits
    # logits = logits[:, -1, :]
    outputs = model.generate(**inputs, max_new_tokens=30)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# %%

if __name__ == "__main__":
    # Set a fixed seed for reproducibility
    set_seed_all(43)
    model_name = "google/gemma-2-9b"
    save_base_path = "/workspace/checkpoints/"
    ds_path = "train/nlp_data.jsonl"

    # argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', nargs='+', type=int, default=None)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--modules', nargs='+', type=str, default='all')
    parser.add_argument('--layer_range', action='store_true', default=False)
    args = parser.parse_args()

    # Load tokenizer and model
    clear_cuda_mem()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation='eager',
    )

    if args.modules == 'all':
        modules = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    else:
        modules = [f"mlp.{name}" for name in args.modules]


    # Apply LoRA
    if args.layers is not None:
        if args.layer_range:
            if len(args.layers) == 2:
                layers = [i for i in range(args.layers[0], args.layers[1])]
                layers_name = "[{}:{}]".format(args.layers[0], args.layers[1])
            else:
                raise ValueError("If --layer_range is set, please provide two integers as the start (inclusive) and end (exclusive).")
        else:
            layers = args.layers
            layers_name = str(args.layers)
        
        # Put lora on MLP of specified layers
        exp_name = f'9b-nlp-{layers_name}-r{args.lora_r}-{args.modules}'
        lora_config = LoraConfig(
            r = args.lora_r,
            target_modules=[f"model.layers.{layer}.{module}" for layer in layers for module in modules],
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        # Put lora on MLP of all layers
        exp_name = f'9b-nlp-all-r{args.lora_r}-{args.modules}'
        lora_config = LoraConfig(
            r = args.lora_r,
            target_modules=modules,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
    model=get_peft_model(model, lora_config)
    print_trainable_params(model)

    output_dir = os.path.join(save_base_path, exp_name)

    # Get training dataset
    train_ds = load_nlp_dataset(ds_path)
    print("Number of datapoints in train set", len(train_ds))

    # Set up training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        overwrite_output_dir=False,
        per_device_train_batch_size=4,
        learning_rate=3e-6,
        max_steps=1000,
        warmup_steps=20,
        save_strategy="steps",
        save_steps=500,
        logging_steps=1,
        bf16=True,           # Use BF16 mixed precision
        fp16=False,          # Disable FP16 training
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )

    # Create the eval callback
    eval_callback = CustomEvalCallback(
        quick_test=quick_test,
        tokenizer=tokenizer,
        eval_steps=20,
    )
    trainer.add_callback(eval_callback)

    # Start training
    run = wandb.init(
        project="oocr",
        dir="/workspace/wandb",
        name=exp_name,
    )
    trainer.train()
    run.finish()


# %%
