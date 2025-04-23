# %%
import os
import json
import gc
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/gemma-2-9b-it"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation='eager',
)
for p in model.parameters():
    p.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%

from utils import LABEL_MAP, load_train_dataset, load_test_dataset, extract_answer, clear_cuda_mem
from torch.utils.data import DataLoader
import re
from datasets import Dataset


# load train dataset

ds_path = "./datagen/dev/047_functions/finetune_01"

train_ds = load_train_dataset(os.path.join(ds_path, "047_func_01_train_oai.jsonl")) # type: ignore

def train_collate_fn(batch):
    # TODO: FIX THIS
    # batch is a list of dicts, each with "prompt" and "completion"
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
    return {
        **{k: v.to(device) for k, v in train_ids.items()},
        "fn_names": [ex["fn_name"] for ex in batch],
    }

next(iter(train_ds))
# %%

train_dataloader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=train_collate_fn) # type: ignore

sample = next(iter(train_dataloader))
for k, v in sample.items():
    print(f"{k}: {v.shape}")


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

    return {
        "input_ids": test_ids.to(device),
        "answer": [ex["answer"] for ex in batch],
        "fn_names": [ex["fn_name"] for ex in batch],
    }

test_dataloader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=test_collate_fn) # type: ignore

for d in test_dataloader:
    print(d)
    break


# %%
from transformers import get_linear_schedule_with_warmup  # type: ignore

d_model = model.config.hidden_size

num_training_steps = len(train_dataloader)
num_warmup_steps = int(0.1 * num_training_steps)

# %%
(function_names := list(LABEL_MAP.keys()))
# %%
for fn in function_names:
    tokens = tokenizer.encode(' ' + function_names[0], return_tensors="pt", add_special_tokens=False)[0]
    tokens_ = tokenizer.encode(function_names[0], return_tensors="pt", add_special_tokens=False)[0]
    assert (tokens[1:] == tokens_[1:]).all()
# %%
{tuple([1, 2]): 'asdf'}[tuple([1, 2])]
# %%
(tokens := tokenizer.encode(' mboetr', return_tensors="pt", add_special_tokens=False)[0].tolist())
[tokenizer.decode(tok) for tok in tokens]
# %%

# HACKY: can index into this by tuples of ints
steering_vectors: dict[tuple[int, ...], torch.Tensor] = {}
for function_name in function_names:
    tokens = tokenizer.encode(function_name, return_tensors="pt", add_special_tokens=False)[0]
    tokens_ = tokenizer.encode(" " + function_name, return_tensors="pt", add_special_tokens=False)[0]
    if (tokens[1:] != tokens_[1:]).any():
        print(function_name)
        print([tokenizer.decode(tok) for tok in tokens])
        print([tokenizer.decode(tok) for tok in tokens_])
        continue
    steering_vectors[tuple(tokens[1:].tolist())] = torch.zeros(d_model).to(device)

list(steering_vectors.keys())
# %%

optimizer = torch.optim.AdamW(
    list(steering_vectors.values()), 
    lr=1e-3,          
    weight_decay=5e-6,
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

# %%

from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
model: Gemma2ForCausalLM

# %%
model.model.layers[0]._forward_hooks
# %%
# global signal variable
should_add_VS = torch.zeros(len(function_names), 1000).to(device)

# hook to add steering vector conditionally at specific positions
def hook(module, args, output):
    assert len(output) == 1
    print(f"{output[0].shape=}")
    b, s, d = output[0].shape
    output_BSD = output[0]
    print(f"1: {output_BSD.shape=}")
    for v, vec_D in enumerate(steering_vectors.values()):
        should_add_S = should_add_VS[v, :s]
        output_BSD += vec_D * should_add_S[:, None]
    print(f"2: {output_BSD.shape=}")
    return (output_BSD,)

handle = model.model.layers[0].register_forward_hook(hook)

try:
    batch = torch.randint(0, 1000, (1, 1000)).to(device)
    model(input_ids=batch)
finally:
    handle.remove()
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
