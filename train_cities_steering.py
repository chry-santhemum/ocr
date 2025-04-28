"""
Train *city-ID-specific* steering vectors that are added to a frozen LM’s
hidden states at a chosen layer.

Dataset line format (train & valid):
{
  "messages": [
      {"role": "user", "content": ...},
      {"role": "assistant", "content": ...}
  ]
}

Assistant replies end with either “… North/South/East/West<eot>” **or**
a numeric distance “… 103 km<eot>”.  We compute the loss only on the
direction token *or* on the entire distance span, mirroring your LoRA script.
"""
from __future__ import annotations
import argparse, itertools, random
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizer,
)
import wandb

from utils import TokenwiseSteeringHook, load_cities_dataset, clear_cuda_mem, find_token_pos

# ───────────────────────── constants ──────────────────────────
CITIES = {
    50337: "Paris",
    93524: "Sao Paulo",
    76881: "Tokyo",
    67781: "New York",
    59894: "Lagos",
}
CITY_IDS = list(CITIES.keys())

# ─────────────────── tokenisation helpers ─────────────────────
def tokenize_and_mark_cities(
    messages: List[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    add_generation_prompt: bool,
) -> Tuple[List[int], List[int]]:
    conv_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt",
        add_generation_prompt=add_generation_prompt
    )[0].tolist()

    occ = [-1] * len(input_ids)
    for cid in CITY_IDS:
        substr = f"City {cid}"
        if substr in conv_str:
            for pos in find_token_pos(tokenizer, substr, conv_str,
                                      last_tok_only=False):
                occ[pos] = CITY_IDS.index(cid)
    return input_ids, occ


def tighten_completion_mask(
    tokens: List[int], mask: List[int], tokenizer: PreTrainedTokenizer
) -> List[int]:
    """Keep only the single direction token when present."""
    directions = [" North", " South", " East", " West"]
    dir_toks = [tokenizer.encode(d, add_special_tokens=False)[0] for d in directions]

    comp_tokens = [t for t, m in zip(tokens, mask) if m]
    has_dir = any(t in comp_tokens for t in dir_toks)
    comp_txt = tokenizer.decode(comp_tokens)
    ends_dist = any(comp_txt.endswith(s) for s in ("km", "mi", "iles", "ilometers"))

    # exactly one of the two patterns
    assert has_dir ^ ends_dist, f"Ambiguous completion: “{comp_txt}”"

    if has_dir:
        keep = next(t for t in dir_toks if t in comp_tokens)
        keep_idx = tokens.index(keep)
        new_mask = [0] * len(tokens)
        new_mask[keep_idx] = 1
        return new_mask
    return mask


def tokenize_example(
    conv: dict, tokenizer: PreTrainedTokenizer, start_tok: int
) -> dict:
    input_ids, occ = tokenize_and_mark_cities(conv["messages"], tokenizer, False)

    split = input_ids.index(start_tok, 10) + 3        # after second <assistant>
    labels = [-100] * split + input_ids[split:]
    labels[-2:] = [-100, -100]                        # mask trailing <eot>

    mask = [int(l != -100) for l in labels]
    mask = tighten_completion_mask(input_ids, mask, tokenizer)
    labels = [tok if m else -100 for tok, m in zip(input_ids, mask)]

    return dict(
        input_ids=input_ids,
        labels=labels,
        city_occurrences=occ,
        attention_mask=[1] * len(input_ids),
    )

def _lpad(seq: List[int], pad: int, tgt: int) -> List[int]:
    return [pad] * (tgt - len(seq)) + seq


def collate(batch: List[dict], pad_token_id: int):
    L = max(len(b["input_ids"]) for b in batch)
    return dict(
        input_ids=torch.tensor([_lpad(b["input_ids"], pad_token_id, L) for b in batch], dtype=torch.long),
        labels=torch.tensor([_lpad(b["labels"], -100, L) for b in batch], dtype=torch.long),
        city_occurrences=torch.tensor([_lpad(b["city_occurrences"], -1, L) for b in batch], dtype=torch.long),
        attention_mask=torch.tensor([_lpad(b["attention_mask"], 0, L) for b in batch], dtype=torch.long),
    )

def full_train():
    cfg = dict(
        layer=9,
        num_epochs=5,
        batch_size=32,
        eval_steps=200,
        log_steps=5,
        save_steps=500,
        lr=7e-2,
        weight_decay=0.0,
        max_len=128,
        ds_train="./data/locations/train.jsonl",
        ds_valid="./data/locations/valid.jsonl",
        model_name="google/gemma-2-9b-it",
    )

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    pad_id = tok.pad_token_id
    start_tok = tok.encode("<start_of_turn>", add_special_tokens=False)[0]

    # datasets
    train_ds = load_cities_dataset(cfg["ds_train"])
    # train_ds = train_ds.select(range(100))
    train_ds = train_ds.map(lambda ex: tokenize_example(ex, tok, start_tok))
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=cfg["batch_size"],
                          collate_fn=lambda b: collate(b, pad_id))

    valid_ds = load_cities_dataset(cfg["ds_valid"])
    valid_ds = valid_ds.map(lambda ex: tokenize_example(ex, tok, start_tok))
    valid_dl = DataLoader(valid_ds, batch_size=cfg["batch_size"],
                          collate_fn=lambda b: collate(b, pad_id))

    # model
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager"
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hook = TokenwiseSteeringHook(model.config.hidden_size, device, len(CITY_IDS))
    handle = model.model.layers[cfg["layer"]].register_forward_pre_hook(hook)

    opt = AdamW([hook.steering_vecs_VD], lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = len(train_dl) * cfg["num_epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total), total)

    run = wandb.init(project="oocr", name=f"city_vec_layer{cfg['layer']}",
                     dir="data/wandb", config=cfg,
                    #  mode="disabled"
                     )

    # training loop
    step, losses = 0, []
    for epoch in range(cfg["num_epochs"]):
        for batch in train_dl:
            hook.vec_ptrs_BS = batch["city_occurrences"].to(device)
            out = model(
                input_ids=batch["input_ids"].to(device),
                labels=batch["labels"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            (out.loss).backward()
            opt.step()
            sched.step()
            opt.zero_grad()

            losses.append(out.loss.item())
            print(f"step {step}, loss {out.loss.item():.4f}, lr {sched.get_last_lr()[0]:.4e}")
            if step and step % cfg["log_steps"] == 0:
                run.log({"train/loss": sum(losses)/len(losses),
                         "train/step": step,
                         "train/epoch": epoch + step/len(train_dl)}, step=step)
                losses.clear()

            if step and step % cfg["save_steps"] == 0:
                ck_dir = Path(f"data/experiments/city_vectors/layer{cfg['layer']}/step_{step}")
                ck_dir.mkdir(parents=True, exist_ok=True)
                for i, cid in enumerate(CITY_IDS):
                    torch.save(hook.steering_vecs_VD[i].detach().cpu(), ck_dir/f"{cid}.pt")
            step += 1
    handle.remove()
    run.finish()

# ─────────────────────── sanity probe ─────────────────────────
def sanity_check():
    layer = 9
    cfg = dict(model_name="google/gemma-2-9b-it", ds_train="./data/locations/train.jsonl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    pad_id = tok.pad_token_id
    start_tok = tok.encode("<start_of_turn>", add_special_tokens=False)[0]

    raw = load_cities_dataset(cfg["ds_train"]).select(range(2))
    ds = raw.map(lambda ex: tokenize_example(ex, tok, start_tok))
    dl = DataLoader(ds, batch_size=2, collate_fn=lambda b: collate(b, pad_id))
    batch = next(iter(dl))

    # colour helpers
    COLORS = [("\033[91m", "\033[0m"), ("\033[92m", "\033[0m"),
              ("\033[93m", "\033[0m"), ("\033[94m", "\033[0m"),
              ("\033[95m", "\033[0m")]
    def hi(s, i): return s if i == -1 else f"{COLORS[i%5][0]}{s}{COLORS[i%5][1]}"
    def decode(ids, mask): return "".join(hi(tok.decode(t), m) for t, m in zip(ids, mask)).replace(" ", "·")

    for i in range(batch["input_ids"].size(0)):
        ids = batch["input_ids"][i].tolist()
        occ = batch["city_occurrences"][i].tolist()
        lab = batch["labels"][i].tolist()
        print("\nCity-token positions:\n", decode(ids, occ))
        print("\nLabel positions:\n", decode(ids, [int(l!=-100) for l in lab]))

    # one forward pass
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"],
                                                 torch_dtype=torch.bfloat16,
                                                 device_map=device,
                                                 attn_implementation="eager")
    hook = TokenwiseSteeringHook(model.config.hidden_size, device, len(CITY_IDS))
    h = model.model.layers[layer].register_forward_pre_hook(hook)
    hook.vec_ptrs_BS = batch["city_occurrences"].to(device)
    with torch.no_grad():
        gen = model.generate(batch["input_ids"].to(device), max_new_tokens=1)
    print("\nUntrained generation:\n", tok.decode(gen[0]))
    h.remove()

if __name__ == "__main__":
    sanity_check()
    # full_train()
