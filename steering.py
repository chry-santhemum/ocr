# %%
# Script to train conditional steering vectors (for the functions task for now)
import os
import json
import gc
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import wandb
from utils import load_train_dataset, load_test_dataset, extract_answer, clear_cuda_mem, load_var_dict, find_token_pos
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/gemma-2-9b-it"

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation='eager',)
model.eval()
for p in model.parameters():
    p.requires_grad = False
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
# load train dataset

ds_path = "../connect_dots/functions/dev/047_functions/finetune_01"
var_dict = load_var_dict(ds_path)

train_ds = load_train_dataset(os.path.join(ds_path, "047_func_01_train_oai.jsonl"))

def train_collate_fn(batch):
    # batch is a list of dicts, each with "prompt" and "completion"
    """
    Returns: {"input_ids", "labels", "loss_mask", "attention_mask"}
    """
    texts = [ex["prompt"]+ex["completion"] for ex in batch]
    just_prompts = [ex["prompt"] for ex in batch]

    messages = tokenizer.apply_chat_template(
        texts,
        tokenize=False,
        add_generation_prompt=False,
    )
    train_ids = tokenizer(
        messages,
        return_tensors="pt",
        padding=True,
    )
    train_ids = {k: v.to(device) for k, v in train_ids.items()}

    prompts = tokenizer.apply_chat_template(
        just_prompts,
        tokenize=False,
        add_generation_prompt=True,
    )

    generation_lengths = torch.Tensor([len(tokenizer.encode(messages[i].replace(prompts[i], ""), add_special_tokens=False)) for i in range(len(prompts))]).to(device)
    prompt_lengths = train_ids["input_ids"].shape[1] - generation_lengths

    # get a dict {fn_name: indices in the batch to steer with that steering vector}
    steer_pos = {fn_name: [] for fn_name in var_dict.keys()}
    for i in range(len(prompts)):
        # find the function names and their token positions
        prompt = prompts[i]
        for fn_name in var_dict.keys():
            if fn_name in prompt:
                token_pos = find_token_pos(tokenizer, fn_name, prompt)
                for pos in token_pos:
                    steer_pos[fn_name].append((i, pos))
    
    # make loss mask
    rows = torch.arange(train_ids["input_ids"].shape[1], device=device).unsqueeze(0)
    cols = prompt_lengths.unsqueeze(1)
    mask = rows < cols  
    
    # labels are the same as input_ids, except we mask out the prompt parts
    train_ids["labels"] = train_ids["input_ids"].clone()
    train_ids["labels"][mask] = -100
    train_ids["steer_pos"] = steer_pos

    return train_ids

train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=train_collate_fn)


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

test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=test_collate_fn)

for d in test_dataloader:
    print(d)
    break


# %%

steer_dict = {}
for k in var_dict.keys():
    steer_dict[k] = torch.zeros((1, model.config.hidden_size), device=device, dtype=torch.bfloat16)
    steer_dict[k].requires_grad = True

# adapted from Jacob's code
def make_steering_hook_hf(vector, token, matrix=None):
    """
    Makes a hook for steering the activations of a HuggingFace model.

    Args:
        vector: a vector which will be added to the activations
        token: a list of tuples
        matrix (optional): a matrix, such that the product of that matrix with the activations will be added to the activations
    """
    def hook_fn(module, input):
        x = input[0]
        rows, cols = zip(*token)
        slice_rows = list(rows)
        slice_cols = list(cols)
        x_sliced = x[slice_rows, slice_cols, :].detach().clone()
        x[slice_rows, slice_cols, :] = x_sliced + vector

        # if matrix is not None:
        #     affine_term = torch.zeros_like(x)
        #     affine_term[:, token] = torch.einsum('...n, mn -> ...m', input_sliced, matrix.to(x))
        #     x = x + affine_term
        return x
    
    return hook_fn


optimizer = torch.optim.AdamW(
    [steer_dict[k] for k in steer_dict.keys()],
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

# %%
# training

import lora_sweep

wandb.init(
    project="oocr", 
    name="yolo-steer", 
    dir="/workspace/wandb",
)

LAYER = 6
step = 0
eval_steps = 50

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

for batch in train_dataloader:
    step += 1
    optimizer.zero_grad()

    
    # make batch-specific hooks
    steer_pos = batch["steer_pos"]
    handles = []
    for fn_name in steer_pos.keys():
        steer_vec = steer_dict[fn_name]
        positions = steer_pos[fn_name]
        if positions:
            hook = make_steering_hook_hf(steer_vec, positions)
            handle = model.model.layers[LAYER].register_forward_pre_hook(hook)
            handles.append(handle)


    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    # remove hooks
    for handle in handles:
        handle.remove()

    logits = outputs.logits  # (B, S, V)

    # since labels=input_ids, we can just flatten
    loss = loss_fn(
        logits.view(-1, logits.size(-1)),
        batch["labels"].view(-1)
    )
    loss.backward()
    optimizer.step()
    scheduler.step()
    # print(f"step {step}: loss {loss.item()}")

    wandb.log({
        "train/loss": loss.item(),
        "train/global_step": step,
        "train/lr": optimizer.param_groups[0]['lr']
    })

    # if step % eval_steps == 0:
    #     # test loop
    #     clear_cuda_mem()
    #     lora_sweep.eval(model, tokenizer, )
    #     wandb.log({"test/accuracy": score/total})

# %%