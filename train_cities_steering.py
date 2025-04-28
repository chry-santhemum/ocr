from __future__ import annotations
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
        eval_steps=10, # increase me
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

    run = wandb.init(
        project="oocr",
        name=f"city_vec_layer{cfg['layer']}",
        dir="data/wandb",
        config=cfg,
         mode="disabled"
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

            if step and step % cfg["eval_steps"] == 0:
                print("evaluating")
                model.eval()
                clear_cuda_mem()
                score = quiz_city_id(model, tok, hook, device)
                print(f"eval score: {score}")
                run.log({"eval/score": score}, step=step)
                

            if step and step % cfg["save_steps"] == 0:
                ck_dir = Path(f"data/experiments/city_vectors/layer{cfg['layer']}/step_{step}")
                ck_dir.mkdir(parents=True, exist_ok=True)
                for i, cid in enumerate(CITY_IDS):
                    torch.save(hook.steering_vecs_VD[i].detach().cpu(), ck_dir/f"{cid}.pt")
            step += 1


    handle.remove()
    run.finish()

def quiz_city_id(model, tokenizer, hook, device) -> float:
    """
    Ask:  “What city is represented by the id 50337? …”
    Grades the model on the 5-way multiple choice.

    Requires:
        - `hook` is the same CitySteeringHook instance used during training.
    """
    letters = ["A", "B", "C", "D", "E"]
    correct = 0

    for idx, cid in enumerate(CITY_IDS):
        prompt_txt = (
            f"What city is represented by City {cid}? Please respond with the letter of the correct answer only.\n\n" +
            "\n".join(f"{l}: {name}" for l, name in zip(letters, CITIES.values()))
        )
        # print(f"prompt_txt: {prompt_txt}")
        # tokenise prompt the *same way as training* (chat template, no gen-prompt)
        messages = [{"role": "user", "content": prompt_txt}]
        input_ids, occ = tokenize_and_mark_cities(
            messages, tokenizer, add_generation_prompt=True
        )
        # chop off trailing <eot>\n (last 2 tokens)

        ids_T = torch.tensor([input_ids], device=device)
        attn_T = torch.ones_like(ids_T, dtype=torch.bool)
        hook.vec_ptrs_BS = torch.tensor([occ], device=device)

        with torch.no_grad():
            gen = model.generate(
                ids_T,
                attention_mask=attn_T,
                max_new_tokens=1,
                do_sample=False
            )

        answer = tokenizer.decode(
            gen[0, len(input_ids):],
            skip_special_tokens=False
        ).replace(' ', '_').replace('\n', '\\n')
        print(f"answer: {answer} vs {letters[idx]} | ", end="")

        if answer.startswith(letters[idx]):
            correct += 1

    return correct / len(CITY_IDS)



if __name__ == "__main__":
    # sanity_check()
    full_train()
