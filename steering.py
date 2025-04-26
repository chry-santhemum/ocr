# %%
# Script to train conditional steering vectors (for the functions task for now)
import os
import json
import gc
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import wandb

from utils import load_train_dataset, load_test_dataset, extract_answer, clear_cuda_mem, load_var_dict, find_token_pos
import lora_sweep

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

function_to_learn = "ttsund"

ds_path = "../connect_dots/functions/dev/047_functions/finetune_01"
var_dict = load_var_dict(ds_path)

# %%
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation='eager',)
model.eval()
for p in model.parameters():
    p.requires_grad = False

# %%
# load train dataset

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
        add_special_tokens=False,
    )
    train_ids = {k: v.to(device) for k, v in train_ids.items()}

    prompts = tokenizer.apply_chat_template(
        just_prompts,
        tokenize=False,
        add_generation_prompt=True,
    )

    generation_lengths = [len(tokenizer.encode(messages[i].replace(prompts[i], ""), add_special_tokens=False)) for i in range(len(prompts))]
    prompt_lengths = [train_ids["input_ids"].shape[1] - x for x in generation_lengths]

    # get a dict {fn_name: indices in the batch to steer with that steering vector}
    steer_pos = {fn_name: [] for fn_name in var_dict.keys()}
    for i in range(len(prompts)):
        # find the function names and their token positions
        prompt = prompts[i]
        for fn_name in var_dict.keys():
            if fn_name + "(" in prompt:
                token_pos = find_token_pos(tokenizer, fn_name, prompt)
                for pos in token_pos:
                    unpadded_prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
                    # need to shift because of padding
                    pos = pos + prompt_lengths[i] - unpadded_prompt_len - 1
                    steer_pos[fn_name].append((i, pos))
    
    # make loss mask
    rows = torch.arange(train_ids["input_ids"].shape[1], device=device).unsqueeze(0)
    cols = torch.tensor(prompt_lengths, device=device, dtype=torch.uint8).unsqueeze(1)
    mask = rows < cols  
    
    # labels are the same as input_ids, except we mask out the prompt parts
    train_ids["labels"] = train_ids["input_ids"].clone()
    train_ids["labels"][mask] = -100
    train_ids["steer_pos"] = steer_pos

    return train_ids

train_ds = train_ds.filter(lambda x: function_to_learn in x["fn_name"])
print("Filtered train datapoints", len(train_ds))

train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=False, collate_fn=train_collate_fn)

print(len(train_dataloader))

# %%

# for d in train_dataloader:
#     print(d["input_ids"][0])
#     tok = [tokenizer.decode(d["input_ids"][0][i]) for i in range(d["input_ids"].shape[1])]
#     print(tok)
#     print(d["labels"][0])
#     print(d["steer_pos"])
#     break

# %%

# load test dataset
from functools import partial

test_ds = load_test_dataset(os.path.join(ds_path, "047_func_01_test_oai.jsonl"))

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

    prompt_len = test_ids.shape[1]

    steer_pos = {fn_name: [] for fn_name in var_dict.keys()}
    for i in range(len(texts)):
        # find the function names and their token positions
        prompt = texts[i][0]['content']

        for fn_name in var_dict.keys():
            if fn_name in prompt:
                token_pos = find_token_pos(tokenizer, fn_name, tokenizer.apply_chat_template(texts[i], tokenize=False, add_generation_prompt=True))
                for pos in token_pos:
                    unpadded_prompt_len = tokenizer.apply_chat_template(texts[i], return_tensors="pt", tokenize=True, add_generation_prompt=True).shape[1]
                    # need to shift because of padding
                    pos = pos + prompt_len - unpadded_prompt_len - 1
                    steer_pos[fn_name].append((i, pos))

    return {
        "input_ids": test_ids.to("cuda"), # type: ignore
        "answer": [ex["answer"] for ex in batch],
        "fn_name": [ex["fn_name"] for ex in batch],
        "steer_pos": steer_pos,
    }

test_ds = test_ds.filter(lambda x: function_to_learn in x["fn_name"])
print("Filtered test datapoints", len(test_ds))

test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=partial(test_collate_fn, tokenizer=tokenizer))

# for d in test_dataloader:
#     print(d["input_ids"][0])
#     print([tokenizer.decode(d["input_ids"][0][i]) for i in range(d["input_ids"].shape[1])])
#     print(d["steer_pos"])

# %%

LAYER = 6
# This is the dict of steering vectors
steer_dict = {}
for k in var_dict.keys():
    # if k == function_to_learn:
    #     steer_dict[k] = torch.load(f"steering_vectors/layer_{LAYER}/{k}-800.pt").to(device)
    #     print(steer_dict[k].shape)
    # else:
    steer_dict[k] = nn.Parameter(torch.zeros((1, model.config.hidden_size), device=device, dtype=torch.float32))
    # steer_dict[k] =torch.zeros((1, model.config.hidden_size), device=device, dtype=torch.float32)

# adapted from Jacob's code

class SteeringHook:
    def __init__(self, steer_dict):
        self.steer_dict = steer_dict
        self.batch_pos  = None            # filled in just before the forward pass
    def __call__(self, module, input):
        x = input[0]
        for name, positions in self.batch_pos.items():
            if not positions: 
                continue
            print(f"steering {name} at {positions}")
            rows, cols = zip(*positions)
            x[rows, cols, :] += self.steer_dict[name].to(x.dtype)
        return x

# %%

hook = SteeringHook(steer_dict)
handle = model.model.layers[LAYER].register_forward_pre_hook(hook)

def eval(model, tokenizer, test_dataloader):
    clear_cuda_mem()
    
    score, total = 0, 0
    score_dict = {}

    for test_batch in test_dataloader:
        with torch.no_grad():
            hook.batch_pos = test_batch["steer_pos"]
            print("="*10, "\n")
            print(hook.batch_pos)
            print(test_batch["input_ids"][0])
            outputs = model.generate(
                input_ids=test_batch["input_ids"],
                max_new_tokens=1,
                do_sample=False,
            )
        print("Successfully outputted")
        print(tokenizer.decode(outputs[0]))
        pred = [tokenizer.decode(outputs[j]) for j in range(outputs.shape[0])]

        model_ans = [extract_answer(pred[j]) for j in range(len(pred))]
        actual_ans = test_batch["answer"]
        fn_names = test_batch["fn_name"]

        total += len(model_ans)
        result = [model_ans[i] == actual_ans[i] for i in range(len(model_ans))]

        score += sum(result)
        for i in range(len(result)):
            if fn_names[i] in score_dict.keys():
                score_dict[fn_names[i]][0] += int(result[i])
                score_dict[fn_names[i]][1] += 1
            else:
                score_dict[fn_names[i]] = [int(result[i]), 1]

    results_dict = {"test/accuracy": score/total}
    for k in score_dict.keys():
        results_dict[f"test/{k}"] = score_dict[k][0] / score_dict[k][1]

    model.train()
    return results_dict

# %%

optimizer = torch.optim.AdamW(
    [steer_dict[k] for k in steer_dict.keys()],
    lr=1e-2,          
    weight_decay=1e-6,
)

step = 0
num_epochs = 3
eval_steps = 50
log_steps = 5
save_steps = 50
loss_fn = torch.nn.CrossEntropyLoss()

num_training_steps = len(train_dataloader) * num_epochs
print("num training steps", num_training_steps)
num_warmup_steps = int(0.05 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

# %%

wandb.init(
    project="oocr", 
    name="yolo-steer", 
    dir="/workspace/wandb",
)

for epoch in range(3):
    for batch in train_dataloader:
        step += 1
        optimizer.zero_grad()

        # make batch-specific hooks
        hook.batch_pos = batch["steer_pos"]

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        logits = outputs.logits  # (B, S, V)

        # since labels=input_ids, we can just flatten
        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1)
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"step {step}: loss {loss.item()}")

        if step % log_steps == 0:
            logging_dict = {}
            for k, v in steer_dict.items():
                if v.grad is not None:
                    logging_dict[f"train/{k}_vector_norm"] = torch.linalg.vector_norm(v).item()
                    logging_dict[f"train/{k}_grad_norm"] = torch.linalg.vector_norm(v.grad).item()

            wandb.log({
                "train/epoch": epoch,
                "train/loss": loss.item(),
                "train/global_step": step,
                "train/lr": optimizer.param_groups[0]['lr'],
                **logging_dict,
            })


        # save all vectors every save_steps
        if step % save_steps == 0:
            os.makedirs(f"steering_vectors/layer-{LAYER}", exist_ok=True)  
            for k, v in steer_dict.items():
                # save the tensor
                if k == function_to_learn:
                    dir_name = f"steering_vectors/layer_{LAYER}/{k}_{step}.pt"
                    torch.save(v, dir_name)


        if step % eval_steps == 0:
            results_dict = eval(model, tokenizer, test_dataloader)
            wandb.log(results_dict)

# remove hook
handle.remove()

# %%
