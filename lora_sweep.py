# %%
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
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig
import wandb
from peft import LoraConfig, get_peft_model
from utils import *

#%%

class CustomEvalCallback(TrainerCallback):
    def __init__(self, eval_function, eval_dataset, tokenizer, eval_steps=500):
        self.eval_function = eval_function
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps == 0:
            print(f"\nRunning evaluation at step {state.global_step}")
            # Run your custom evaluation
            eval_results = self.eval_function(model, self.eval_dataset, self.tokenizer)
            
            # Log to wandb
            wandb.log(eval_results, step=state.global_step)
            
            print(f"Evaluation results: {eval_results}")
        return control


def eval(model, eval_dataset, tokenizer, batch_size=64, num_samples=None):
    """
    Memory-optimized evaluation function
    """
    model.eval()
    
    test_dataset, ans = eval_dataset

    test_dataset = test_dataset[:num_samples] if num_samples is not None else test_dataset
    ans = ans[:num_samples] if num_samples is not None else ans
    
    total_samples = len(test_dataset)
    model_ans = []
    
    # Process in batches to reduce memory usage
    for i in range(0, total_samples, batch_size):
        batch_end = min(i + batch_size, total_samples)
        batch_data = test_dataset[i:batch_end]
        
        # Apply tokenization to just this batch
        input_ids = tokenizer.apply_chat_template(
            batch_data, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors='pt', 
            padding=True
        ).to("cuda")
        
        with torch.no_grad():
            # Use more memory-efficient generation parameters
            batch_outputs = model.generate(
                input_ids,
                max_new_tokens=8,  # Limit generation length if possible
                do_sample=False,
                use_cache=True  # Ensure caching is enabled for efficiency
            )
        
        # Print samples from first batch only
        if i == 0:
            print("="*50)
            for j in range(min(3, len(batch_outputs))):
                print(tokenizer.decode(batch_outputs[j,:]))
                print("-"*50)
        
        # Process just the relevant output tokens
        batch_decoded = [tokenizer.decode(batch_outputs[j,:]) for j in range(batch_outputs.shape[0])]
        
        # Extract answers
        batch_model_ans = [extract_answer(batch_decoded[j]) for j in range(len(batch_decoded))]
        model_ans.extend(batch_model_ans)
        
        # Explicitly clear GPU memory
        del input_ids, batch_outputs
        torch.cuda.empty_cache()
        gc.collect()
    
    # Calculate accuracy
    correct = [ans[i]==model_ans[i] for i in range(total_samples)]
    score = sum(correct)/total_samples
    
    results = {"Accuracy": score}
    model.train()
    
    return results


#%%

if __name__ == "__main__":

    # Set a fixed seed for reproducibility
    set_seed(42)
    model_name = "google/gemma-2-9b-it"
    ds_path = "inductive-oocr/functions/dev/047_functions/finetune_01"

    # argparse
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=None)
    parser.add_argument('--lora_r', type=int, default=8)

    args = parser.parse_args()

    # CUDA setup
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use BF16 for better numerical stability
        device_map="auto",
        attn_implementation='eager',
        use_auth_token=True,
    )

    # Apply LoRA
    if args.layer is not None:
        output_dir = f'./checkpoints/9b-functions-l{args.layer}-lora-r{args.lora_r}/'
        lora_config = LoraConfig(
            r = args.lora_r,
            target_modules=[f"model.layers. {args.layer}.mlp.up_proj",
                            f"model.layers.{args.layer}.mlp.down_proj"],
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        output_dir = f'./checkpoints/9b-functions-mlp-lora-no-gate/'
        lora_config = LoraConfig(
            r = args.lora_r,
            target_modules=["gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

    model=get_peft_model(model, lora_config)
    print_trainable_params(model)


    # Get training dataset
    train_dataset = load_functions_dataset(os.path.join(ds_path, "047_func_01_train_oai.jsonl"))

    # Set up training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # Increased further to reduce memory pressure
        learning_rate=2e-5,
        max_steps=3000,
        warmup_steps=50,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=10,
        num_train_epochs=1,
        bf16=True,           # Use BF16 mixed precision
        fp16=False,          # Disable FP16 training
        # gradient_checkpointing=True,  # THIS DOES NOT WORK WITH LOR
    )

    # Set up trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Get eval dataset
    eval_dataset = load_eval_dataset(os.path.join(ds_path, "047_func_01_test_oai.jsonl"))

    # Create the eval callback
    eval_callback = CustomEvalCallback(
        eval_function=eval,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        eval_steps=250,
    )
    trainer.add_callback(eval_callback)


    # Start training
    run = wandb.init(
        project="oocr",
        name=output_dir[14:],
        config={
            "model": "gemma-2-9b-it",
            "lr": 2e-5,
            "task": "functions",
            "layer": args.layer,
            "epochs": 1,
        },
    )
    print("Starting training...")
    trainer.train()
    run.finish()

# %%
