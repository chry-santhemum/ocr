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
from torch.utils.data import DataLoader
from functools import partial
from trl import SFTTrainer, SFTConfig
import wandb
from peft import LoraConfig, get_peft_model
from utils import load_train_dataset, load_test_dataset, extract_answer, clear_cuda_mem, print_trainable_params

#%%

def test_collate_fn(batch, tokenizer):
    # batch is a list of dicts, each with "messages"
    texts = [ex["messages"] for ex in batch]
    test_ids = tokenizer.apply_chat_template(
        texts,
        return_tensors="pt",
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
    )
    
    return {"input_ids": test_ids.to("cuda"), "answer": [ex["answer"] for ex in batch]}


class CustomEvalCallback(TrainerCallback):
    def __init__(self, eval_function, eval_dataset, tokenizer, batch_size=64, eval_steps=500):
        self.eval_function = eval_function
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.test_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=partial(test_collate_fn, tokenizer=tokenizer))
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps == 0:
            print(f"\nRunning evaluation at step {state.global_step}")
            # Run your custom evaluation
            eval_results = self.eval_function(model, self.tokenizer, self.test_dataloader)
            
            # Log to wandb
            wandb.log(eval_results, step=state.global_step)
            
            print(f"Evaluation results: {eval_results}")
        return control


def eval(model, tokenizer, test_dataloader):
    model.eval()
    clear_cuda_mem()
    
    score, total = 0, 0
    for test_batch in test_dataloader:
        with torch.no_grad():
            print("="*10, "\n")
            outputs = model.generate(
                input_ids=test_batch["input_ids"],
                max_new_tokens=5,
                do_sample=False,
            )

            print(tokenizer.decode(outputs[0]))
            pred = [tokenizer.decode(outputs[j]) for j in range(outputs.shape[0])]

            model_ans = [extract_answer(pred[j]) for j in range(len(pred))]
            actual_ans = test_batch["answer"]

            total += len(model_ans)
            score += sum([model_ans[i] == actual_ans[i] for i in range(len(model_ans))])

    print("Accuracy:", score/total)
    model.train()
    
    return {"Accuracy": score/total}


#%%

if __name__ == "__main__":

    # Set a fixed seed for reproducibility
    set_seed(42)
    model_name = "google/gemma-2-9b-it"
    ds_path = "connect_dots/functions/dev/047_functions/finetune_01"
    save_base_path = "/workspace/checkpoints/"

    # argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', nargs='+', type=int, default=None)
    parser.add_argument('--lora_r', type=int, default=8)
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

    # Apply LoRA
    if args.layers is not None:
        # Put lora on MLP of specified layers
        output_dir = os.path.join(save_base_path, f'9b-func-{str(args.layers)}-r{args.lora_r}')
        lora_config = LoraConfig(
            r = args.lora_r,
            target_modules=[f"model.layers.{layer}.mlp.up_proj" for layer in args.layers] + 
                           [f"model.layers.{layer}.mlp.down_proj" for layer in args.layers] + 
                           [f"model.layers.{layer}.mlp.gate_proj" for layer in args.layers],
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        # Put lora on MLP of all layers
        output_dir = os.path.join(save_base_path, f'9b-func-all-r{args.lora_r}')
        lora_config = LoraConfig(
            r = args.lora_r,
            target_modules=["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
    model=get_peft_model(model, lora_config)
    print_trainable_params(model)

    # Get training dataset
    train_dataset = load_train_dataset(os.path.join(ds_path, "047_func_01_train_oai.jsonl"))

    # Set up training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_steps=2000,
        warmup_steps=50,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=5,
        num_train_epochs=1,
        bf16=True,           # Use BF16 mixed precision
        fp16=False,          # Disable FP16 training
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Get eval dataset
    test_ds = load_test_dataset(os.path.join(ds_path, "047_func_01_test_oai.jsonl"))

    # Create the eval callback
    eval_callback = CustomEvalCallback(
        eval_function=eval,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        batch_size=64,
        eval_steps=250,
    )
    trainer.add_callback(eval_callback)

    # Start training
    run = wandb.init(
        project="oocr",
        dir="/workspace/wandb",
        name=output_dir[23:],
    )
    trainer.train()
    run.finish()

# %%
