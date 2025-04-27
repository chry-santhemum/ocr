# %%
import json
import math
from functools import partial
from pathlib import Path
import random
from typing import Any

import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from datasets import Dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
    set_seed,
)  # type: ignore

from utils import print_trainable_params

device = torch.device("cuda")

CITIES = {
    50337: "Paris",
    93524: "Sao Paulo",
    76881: "Tokyo",
    67781: "New York",
    59894: "Lagos",
}

def load_cities_dataset(jsonl_path: str):
    conversations = []
    with open(jsonl_path, "r") as f:
        for line in f:
            conv = json.loads(line)  # {"messages": [...]}
            # Reformat structure slightly for apply_chat_template
            system_msg, user_msg, assistant_msg = conv["messages"]
            # Combine system and user prompts as per original SFTTrainer logic inferred from data loading
            combined_user_content = f"{system_msg['content']}\n\n{user_msg['content']}"
            conversations.append(
                {
                    "messages": [
                        {"role": "user", "content": combined_user_content},
                        {"role": "assistant", "content": assistant_msg["content"]},
                    ]
                }
            )
    return Dataset.from_list(conversations)

# %%

def make_single_city_dataset(jsonl_path: str, city_id: str):
    print(f"Constructing dataset for city {CITIES[int(city_id)]}")
    cities = set(map(str, CITIES.keys()))
    other_city_ids = [id for id in list(cities - {city_id})]
    print(f"Other city IDs: {other_city_ids}")
    ds = load_cities_dataset(jsonl_path)
    print(f'initial dataset size: {len(ds)}')
    n_replaced = 0
    def replace_other_city_labels(c):
        nonlocal n_replaced
        for m in c['messages']:
            for other_city_id in other_city_ids:
                n_replaced += m['content'].count(f"City {other_city_id}")
                m['content'] = m['content'].replace(f"City {other_city_id}", CITIES[int(other_city_id)])
        return c
    
    def conv_contains_this_city_id(c):
        return str(city_id) in c['messages'][0]['content']

    ds = ds.filter(conv_contains_this_city_id).map(replace_other_city_labels)
    for c in ds['messages']:
        for other_id in other_city_ids:
            assert f"City {other_id}" not in c[0]['content']
            assert f"City {city_id}" in c[0]['content']
    
    print(f"Replaced {n_replaced} instances of other city IDs")
    print(f"Final dataset size: {len(ds)}")
    return ds

def visualize_tokenizaion(s: str, tokenizer: PreTrainedTokenizer):
    toks = tokenizer.encode(s)
    print('|'.join(tokenizer.decode(t) for t in toks))
# %%


def tokenize_with_completion_mask(
    conversation: dict[str, list[dict[str, str]]], tokenizer: PreTrainedTokenizer
) -> dict[str, list[int]]:
    """
    Returns:
        input_ids: list[int]
        completion_mask: list[int]
    """
    assert "messages" in conversation
    messages = conversation["messages"]
    assert isinstance(messages, list)
    assert isinstance(messages[0], dict)
    assert "role" in messages[0]
    assert "content" in messages[0]
    conversation_str: str = tokenizer.apply_chat_template(  # type: ignore
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )

    # split conversation_str into input vs completion
    marker = "<start_of_turn>model\n"
    if marker not in conversation_str:
        raise ValueError("Marker not found in conversation string")

    prompt, completion = conversation_str.split(marker)
    prompt += marker

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

    tokens = prompt_tokens + completion_tokens
    completion_mask = [0] * len(prompt_tokens) + [1] * len(completion_tokens)

    return {
        "input_ids": tokens,
        "completion_mask": completion_mask,
    }

def collate_fn(batch: list[dict[str, list[int]]], pad_token_id: int):
    """collates multiple examples into a batch, examples can have different lengths"""
    max_length = max(len(ex["input_ids"]) for ex in batch)

    input_ids = []
    completion_mask = []

    for ex in batch:
        assert len(ex["input_ids"]) == len(ex["completion_mask"])
        remaining_length = max_length - len(ex["input_ids"])

        # append padding tokens
        input_ids.append(ex["input_ids"] + [pad_token_id] * remaining_length)

        # append 0s so we don't predict the padding tokens
        completion_mask.append(ex["completion_mask"] + [0] * remaining_length)

    input_ids = torch.tensor(input_ids, device=device, dtype=torch.long)
    completion_mask = torch.tensor(completion_mask, device=device, dtype=torch.long)

    labels = torch.where(
        completion_mask == 1,
        input_ids,
        torch.tensor(-100),
    )

    return {"input_ids": input_ids, "labels": labels}


# def calculate_accuracy(logits, labels):
#     """Calculates token accuracy, ignoring -100 labels."""
#     predictions = torch.argmax(logits, dim=-1)
#     active_mask = labels != -100
#     active_predictions = predictions[:, :-1][active_mask]
#     active_labels = labels[:, 1:][active_mask]
#     correct = (active_predictions == active_labels).sum().item()
#     active_count = active_mask.sum().item()
#     return correct, active_count


def main(cfg: dict[str, Any]):
    set_seed(cfg["seed"])

    exp_base_dir = Path("data") / "experiments" / cfg["exp_name"]
    if exp_base_dir.exists():
        print(f"Output directory {exp_base_dir} already exists. careful not to overwrite existing checkpoints.")
    exp_base_dir.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists

    # save config
    with open(exp_base_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(  # type: ignore
        cfg["model_name"],
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",  # Consider changing to "sdpa" if supported and compatible
    )

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        target_modules=[f"model.layers.{layer}.{module}" for layer in cfg["lora_layers"] for module in cfg["lora_modules"]],
        lora_alpha=cfg["lora_alpha"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    lora_model = get_peft_model(model, lora_config)
    print_trainable_params(lora_model)

    train_ds = make_single_city_dataset(cfg["train_jsonl_path"], cfg["city_id"])
    print("Total train datapoints", len(train_ds))
    print('sample train datapoints:')
    sep = "\n\n"
    for msgs in random.sample(train_ds['messages'], 10):
        print(' -')
        print(f'  q: "{msgs[0]["content"].split(sep)[1]}"')
        print(f'  a: "{msgs[1]["content"]}"')
    print('-' * 100)

    valid_ds = make_single_city_dataset(cfg["valid_jsonl_path"], cfg["city_id"])
    print('sample valid datapoints:')
    for msgs in random.sample(valid_ds['messages'], 10):
        print(f'- q: "{msgs[0]["content"].split(sep)[1]}"')
        print(f'  a: "{msgs[1]["content"]}"')
    print('-' * 100)
    print("Total valid datapoints", len(valid_ds))

    map_fn = partial(tokenize_with_completion_mask, tokenizer=tokenizer)
    tokenized_train_ds = train_ds.map(map_fn, remove_columns=train_ds.column_names)
    tokenized_valid_ds = valid_ds.map(map_fn, remove_columns=valid_ds.column_names)

    pad_token_id = tokenizer.pad_token_id
    print(f"Pad token ID: {pad_token_id}")
    collate_fn_ = partial(collate_fn, pad_token_id=pad_token_id)
    train_dataloader = DataLoader(tokenized_train_ds, shuffle=True, collate_fn=collate_fn_, batch_size=cfg["train_batch_size"])

    valid_dataloader = DataLoader(tokenized_valid_ds, collate_fn=collate_fn_, batch_size=cfg["valid_batch_size"])

    num_training_steps = math.ceil(len(train_dataloader) / cfg["gradient_accumulation_steps"]) * cfg["num_epochs"]

    optimizer = optim.AdamW(lora_model.parameters(), lr=cfg["lr"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg["warmup_steps"], num_training_steps=num_training_steps
    )

    run = wandb.init(
        project="oocr",
        dir="data/wandb",
        name=cfg["exp_name"],
        # mode="disabled",
    )
    lora_model.train()

    global_step = 0
    losses = []

    for epoch in range(cfg["num_epochs"]):
        print(f"Starting Epoch {epoch + 1}/{cfg['num_epochs']}")
        lora_model.train()

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = lora_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            (loss / cfg["gradient_accumulation_steps"]).backward()

            losses.append(loss.item())

            if (batch_idx + 1) % cfg["gradient_accumulation_steps"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % cfg["logging_steps"] == 0:
                    log_dict = {
                        "train/loss": sum(losses) / len(losses),
                        "train/epoch": epoch + (batch_idx + 1) / len(train_dataloader),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                    run.log(log_dict, step=global_step)
                    print(f'{global_step}: {log_dict}')
                    losses.clear()

                if global_step % cfg["save_steps"] == 0:
                    ckpt = exp_base_dir / "checkpoints" / f"checkpoint-{global_step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    lora_model.save_pretrained(str(ckpt))
                    tokenizer.save_pretrained(str(ckpt))
                    print(f"Saved checkpoint to {ckpt}")
                
                if global_step % cfg["eval_steps"] == 0:
                    with torch.no_grad():
                        lora_model.eval()
                        print(f"Validating at step {global_step}")
                        # print 10 completions
                        for i in range(min(10, len(labels))):
                            ex_labels = labels[i]
                            ex_predictions = torch.argmax(outputs.logits[i], dim=-1)
                            completion_mask = ex_labels != -100
                            pred_toks = ex_predictions[completion_mask]
                            gt_toks = ex_labels[completion_mask]
                            print(f"GT:   {'|'.join(sanitise(tokenizer.decode(t)) for t in gt_toks)}")
                            print(f"Pred: {'|'.join(sanitise(tokenizer.decode(t)) for t in pred_toks)}\n")
                        
                        # res = lora_model.generate(**tokenizer("what city is encoded as 50337?"))
                        # print(tokenizer.decode(res[0]))
                        # print(f"back to training")
                    lora_model.train()

        with torch.no_grad():
            lora_model.eval()
            losses = []
            for batch in tqdm(valid_dataloader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = lora_model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                losses.append(loss.item())
            avg_loss = sum(losses) / len(losses)
            print(f"Validation loss: {avg_loss}")
            run.log({"valid/loss": avg_loss}, step=global_step)

    run.finish()
    print("Training finished.")

    final_save_path = exp_base_dir / "checkpoints" / "final_model"
    lora_model.save_pretrained(str(final_save_path))
    tokenizer.save_pretrained(str(final_save_path))
    print(f"Final model saved to {final_save_path}")

def sanitise(s: str):
    return s.replace("\n", "\\n").replace(" ", "\_").strip()



base_cfg = {
    "model_name": "google/gemma-2-9b-it",

    "lora_modules": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    "lora_layers": [4],
    "lora_r": 2,
    "lora_alpha": 1,

    "seed": 42,
    "lr": 2e-3,
    "num_epochs": 5,
    "train_batch_size": 16,
    "valid_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "logging_steps": 5,
    "eval_steps": 100,
    "save_steps": 50,
    "max_seq_length": 1024,
    "warmup_steps": 50,
    "train_jsonl_path": "./data/locations/train.jsonl",
    "valid_jsonl_path": "./data/locations/valid.jsonl",
}

if __name__ == "__main__":
    # for city_id, city_name in CITIES.items():
    #     main({
    #         **base_cfg,
    #         "city_id": str(city_id),
    #         "exp_name": f"9b-layer24-r1-mlp-{city_name}",
    #     })

    main({
        **base_cfg,
        "city_id": str(93524),
        "exp_name": f"9b-layer4-r2-mlp-Sao Paulo",
    })

# %%
