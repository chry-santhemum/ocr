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

from utils import load_train_dataset, load_test_dataset, extract_answer, clear_cuda_mem
from torch.utils.data import DataLoader

# load train dataset

ds_path = "/workspace/inductive-oocr/functions/dev/047_functions/finetune_01"

train_ds = load_train_dataset(os.path.join(ds_path, "047_func_01_train_oai.jsonl"))

def train_collate_fn(batch):
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

train_dataloader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=train_collate_fn)

for d in train_dataloader:
    print([(k, v.shape) for k, v in d.items()])
    break


# %%

# load test dataset

test_ds = load_test_dataset(os.path.join(ds_path, "047_func_01_test_oai.jsonl"))

def test_collate_fn(batch):
    # batch is a list of dicts, each with "messages"
    texts = [ex["messages"] for ex in batch]
    test_ids = tokenizer.apply_chat_template(
        texts,
        return_tensors="pt",
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
    )
    
    return {"input_ids": test_ids.to(device), "answer": [ex["answer"] for ex in batch]}

test_dataloader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=test_collate_fn)

for d in test_dataloader:
    print(d)
    break


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
    weight_decay=5e-6,
)

num_training_steps = len(train_dataloader)
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
)

step = 0
eval_steps = 50

for batch in train_dataloader:
    clear_cuda_mem()
    model.train()
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
        "train/lr": optimizer.param_groups[0]['lr']
    })

    if step % eval_steps == 0:
        # test loop
        clear_cuda_mem()
        model.eval()
        score, total = 0, 0
        for test_batch in test_dataloader:
            with torch.no_grad():
                print("="*10, tokenizer.decode(test_batch["input_ids"][0]), "="*10)

                outputs = model(
                    input_ids=test_batch["input_ids"],
                    do_sample=True,
                )
                pred = torch.argmax(outputs.logits[:,-1,:], dim=-1)
                del outputs

                model_ans = [tokenizer.decode(pred[i]) for i in range(pred.shape[0])]
                actual_ans = test_batch["answer"]

                total += len(model_ans)
                score += sum([model_ans[i] == actual_ans[i] for i in range(len(model_ans))])
                print("predictions:", model_ans)
                print("accuracy:", score/total)
            break
        wandb.log({"test/accuracy": score/total})
# %%
