# %%
import os
import json
import gc
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/gemma-2-9b-it"

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation='eager',)
for p in model.parameters():
    p.requires_grad = False
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%

from utils import load_functions_dataset, load_eval_dataset, extract_answer
from torch.utils.data import DataLoader

# load train dataset

ds_path = "/workspace/inductive-oocr/functions/dev/047_functions/finetune_01"

train_ds = load_functions_dataset(os.path.join(ds_path, "047_func_01_train_oai.jsonl"))

def collate_fn(batch):
    # batch is a list of dicts, each with "messages"
    texts = [ex["messages"] for ex in batch]
    messages = tokenizer.apply_chat_template(
        texts,
        tokenize=False,
        add_generation_prompt=True,
    )
    train_ids = tokenizer(
        messages,
        return_tensors="pt",
        padding=True,
    )
    
    # labels are the same as input_ids
    train_ids["labels"] = train_ids["input_ids"].clone()
    return {k: v.to(device) for k, v in train_ids.items()}

dataloader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)

for d in dataloader:
    print([(k, v.shape) for k, v in d.items()])
    break


# %%

# load test dataset


eval_ds, correct_ans = load_eval_dataset(os.path.join(ds_path, "047_func_01_test_oai.jsonl"))

# %%

eval_ds

# %%

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



# %%
from transformers import get_linear_schedule_with_warmup

d_model = model.config.hidden_size
rank = 16

class SteeringMLP(nn.Module):
    def __init__(self, d_model, rank):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(d_model, rank),
            nn.GELU(),
            nn.Linear(rank, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.to(torch.bfloat16)

    def forward(self, x):
        delta = self.norm(self.MLP(x))
        return delta

steer = SteeringMLP(d_model, rank).to(device)
optimizer = torch.optim.AdamW(
    steer.parameters(), 
    lr=1e-3,          
    weight_decay=1e-5,
)

num_training_steps = len(dataloader)
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)


def hook(module, inp, output):
    # output is a tuple (act, …) for GPT-2 blocks
    # act shape: [batch_size, seq_len, d_model]
    h = output[0] + steer(output[0])
    return (h, *output[1:])


for L in model.model.layers:
    L.register_forward_hook(hook)

# %%
# training
import wandb

wandb.init(
    project="oocr", 
    name="gemm2-9b-it-steer", 
    dir="/workspace/wandb",
    config={
        "lr": 1e-2,
    }
)

model.train()

total_loss = 0.0
step = 0
for batch in dataloader:
    step += 1
    optimizer.zero_grad()
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logits = outputs.logits  # (B, S, V)
    # shift so that tokens <n> predict <n+1>
    # but since labels=input_ids, we can just flatten
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch["labels"].view(-1),
        ignore_index=tokenizer.pad_token_id,
    )
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(f"step {step}: loss {loss.item()}")

    wandb.log({
        "train/loss": loss.item(),
        "train/step": step,
        # you can also log the learning rate if you use a scheduler:
        # "train/lr": optimizer.param_groups[0]['lr']
    })
# %%
