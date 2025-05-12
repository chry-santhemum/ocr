# %%
import argparse
from datetime import datetime
import time
import json
from pathlib import Path
from random import shuffle
import pandas as pd
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)
import wandb
from utils import (
    TokenwiseSteeringHook,
    load_cities_dataset,
    clear_cuda_mem,
    find_token_pos,
    CITY_NAME_TO_ID,
    CITY_IDS,
    CITY_ID_TO_NAME,
    load_cities_dataset_real_names,
    set_seed_all,
)


def tokenize_and_mark_cities(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    add_generation_prompt: bool,
) -> tuple[list[int], list[int]]:
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

# TRAIN ==============================================

def tighten_completion_mask(
    tokens: list[int], mask: list[int], tokenizer: PreTrainedTokenizer
) -> list[int]:
    """Keep only the single direction token when present."""
    directions = [" North", " South", " East", " West"]
    dir_toks = [tokenizer.encode(d, add_special_tokens=False)[0] for d in directions]

    comp_tokens = [t for t, m in zip(tokens, mask) if m]
    has_dir = any(t in comp_tokens for t in dir_toks)
    comp_txt = tokenizer.decode(comp_tokens)
    ends_dist = any(comp_txt.endswith(s) for s in ("km", "mi", "iles", "ilometers"))

    # exactly one of the two patterns
    assert has_dir ^ ends_dist, f"Ambiguous completion: {comp_txt}"

    if has_dir:
        keep = next(t for t in dir_toks if t in comp_tokens)
        keep_idx = tokens.index(keep)
        new_mask = [0] * len(tokens)
        new_mask[keep_idx] = 1
        return new_mask
    return mask


def train_map_tokenize_example(
    conv: dict, tokenizer: PreTrainedTokenizer, start_tok: int
) -> dict:
    input_ids, occ = tokenize_and_mark_cities(conv["messages"], tokenizer, True)

    labels = input_ids.copy()
    split = labels.index(start_tok, 10) + 3  # start looking _after_ the first occurence of "start_tok" (10 is somewhat arbitrary)
    labels[:split] = [-100] * split  # mask the prompt
    labels[-2:] = [-100, -100]  # mask trailing <eot>

    mask = [int(l != -100) for l in labels]
    mask = tighten_completion_mask(input_ids, mask, tokenizer)
    labels = [tok if m else -100 for tok, m in zip(input_ids, mask)]

    return dict(
        input_ids=input_ids,
        labels=labels,
        city_occurrences=occ,
        attention_mask=[1] * len(input_ids),
    )


def _lpad(seq: list[int], pad: int, tgt: int) -> list[int]:
    assert len(seq) <= tgt, f"cant pad, Sequence is too long: {len(seq)} > {tgt}"
    return [pad] * (tgt - len(seq)) + seq


def collate_train(batch: list[dict], pad_token_id: int):
    L = max(len(b["input_ids"]) for b in batch)
    return dict(
        input_ids=torch.tensor([_lpad(b["input_ids"], pad_token_id, L) for b in batch], dtype=torch.long),
        labels=torch.tensor([_lpad(b["labels"], -100, L) for b in batch], dtype=torch.long),
        city_occurrences=torch.tensor([_lpad(b["city_occurrences"], -1, L) for b in batch], dtype=torch.long),
        attention_mask=torch.tensor([_lpad(b["attention_mask"], 0, L) for b in batch], dtype=torch.long),
    )

# CATEGORICAL EVAL (not working yet) ==============================================

LETTERS = ["A", "B", "C", "D", "E"]


cols = ['city', 'category', 'question', 'correct_answer', 'wrong_1', 'wrong_2', 'wrong_3', 'wrong_4']

def format_categorical_question(row: pd.Series, tokenizer: PreTrainedTokenizer) -> list[dict[str, str]]:

    answers = [
        row["correct_answer"],
        row["wrong_1"],
        row["wrong_2"],
        row["wrong_3"],
        row["wrong_4"],
    ]
    shuffle(answers)

    correct_idx = answers.index(row["correct_answer"])
    correct_letter = LETTERS[correct_idx]

    obfuscated_question = row["question"].replace(row["city"], f"City {CITY_NAME_TO_ID[row['city']]}")
    obfuscated_question += " Please respond with the letter of the correct answer only.\n\n"
    obfuscated_question += "\n".join(f"{l}: {a}" for l, a in zip(LETTERS, answers))
    input_ids, occ = tokenize_and_mark_cities(
        [{"role": "user", "content": obfuscated_question}],
        tokenizer,
        add_generation_prompt=True,
    )

    base_question = row["question"]
    base_question += " Please respond with the letter of the correct answer only.\n\n"
    base_question += "\n".join(f"{l}: {a}" for l, a in zip(LETTERS, answers))
    input_ids_base, _ = tokenize_and_mark_cities(
        [{"role": "user", "content": base_question}],
        tokenizer,
        add_generation_prompt=True,
    )
    occ_base = [-1] * len(input_ids_base)

    return {
        "input_ids": input_ids,
        "city_occurrences": occ,
        "input_ids_base": input_ids_base,
        "city_occurrences_base": occ_base,
        "correct_city_name": row["city"],
        "correct_city_id": CITY_NAME_TO_ID[row["city"]],
        "correct_letter": correct_letter,
        "category": row["category"],
    }

def load_categorical_eval_ds(tok) -> Dataset:
    path = "../connect_dots/locations/data/obfuscated_city_qa_dataset.csv"
    df = pd.read_csv(path)
    assert set(df.columns) == set(cols), f"Columns do not match: {set(df.columns)} != {set(cols)}"
    return Dataset.from_list([
        format_categorical_question(row, tok)
        for _, row in df.iterrows()
    ])

CATEGORIES = ["Culture", "Geography", "History"]


def eval_depth_categorical(
    tok: PreTrainedTokenizer,
    dl: DataLoader,
    model: PreTrainedModel,
    hook: TokenwiseSteeringHook,
    input_ids_key: str,
    city_occurrences_key: str,
) -> tuple[
    dict[str, float],
    dict[str, float],
]:
    total = {cid: {cat:0 for cat in CATEGORIES} for cid in CITY_IDS}
    correct = {cid: {cat:0 for cat in CATEGORIES} for cid in CITY_IDS}
    cum_correct_tok_probs = {cid: {cat: 0 for cat in CATEGORIES} for cid in CITY_IDS}

    for batch in dl:
        inp = batch[input_ids_key].to(device)
        occ = batch[city_occurrences_key].to(device)

        hook.vec_ptrs_BS = occ

        with torch.no_grad():
            last_logits = model(input_ids=inp).logits[:, -1, :]
            preds = torch.argmax(last_logits, dim=-1)
            probs = torch.softmax(last_logits, dim=-1)

        hook.vec_ptrs_BS = None

            # get the token id of the correct letter
        correct_tok_ids = torch.tensor([tok.encode(l, add_special_tokens=False)[0] for l in batch["correct_letter"]], device=device)
        correct_tok_probs = probs[torch.arange(len(probs)), correct_tok_ids]

        pred_letters = tok.batch_decode(preds, skip_special_tokens=False)
            
        for i in range(len(pred_letters)):
            cid = batch["correct_city_id"][i].item()
            cat = batch["category"][i]

            cum_correct_tok_probs[cid][cat] += correct_tok_probs[i]
            if pred_letters[i].strip().startswith(batch["correct_letter"][i]):
                correct[cid][cat] += 1

            total[cid][cat] += 1

    correct.update({cat: 0 for cat in CATEGORIES})
    probs  = {cat: 0 for cat in CATEGORIES}
    total.update({cat: 0 for cat in CATEGORIES})

    for cat in CATEGORIES:
        for city_id in CITY_IDS:
            total[cat] += total[city_id][cat]
            probs[cat] += cum_correct_tok_probs[city_id][cat]
            correct[cat] += correct[city_id][cat]
        
    
    acc = {cat: correct[cat] / total[cat] for cat in CATEGORIES}
    avg_probs = {cat: probs[cat] / total[cat] for cat in CATEGORIES}

    return acc, avg_probs


# EVAL ==============================================

CITY2ANSWER_COL = {
    "New York":  "answer_new_york",
    "Paris":     "answer_paris",
    "Tokyo":     "answer_tokyo",
    "Sao Paulo": "answer_sao_paulo",
    "Lagos":     "answer_lagos",
}

HEADER = " Please respond with the letter of the correct answer only.\n\n"

def format_eval_questions(row: pd.Series) -> list[dict[str, str]]:
    """
    row:
        columns: [question_template,category,answer_new_york,answer_paris,answer_tokyo,answer_sao_paulo,answer_lagos]
    each containing:
        city_name, city_id, base_question, obf_question, correct_letter
    """

    formatted_list = []
    for city_id, city_name in CITY_ID_TO_NAME.items():
        q_base = row["question_template"].format(city=city_name) + HEADER
        q_obf  = row["question_template"].format(city=f"City {city_id}") + HEADER

        shuffled_city_names = list(CITY_ID_TO_NAME.values())
        shuffle(shuffled_city_names)

        correct_letter = LETTERS[shuffled_city_names.index(city_name)]

        for l, city_name in zip(LETTERS, shuffled_city_names):
            answer_candidate = row[CITY2ANSWER_COL[city_name]]
            q_base += f"{l}: {answer_candidate}\n"
            q_obf  += f"{l}: {answer_candidate}\n"

        formatted_list.append(
            dict(
                city_name=city_name,
                city_id=city_id,
                base_question=q_base,
                obf_question=q_obf,
                correct_letter=correct_letter,
            )
        )

    return formatted_list


def get_eval_dataset(tokenizer) -> Dataset:
    df = pd.read_csv("../connect_dots/locations/data/pivot_city_questions.csv")
    records = []

    for _, row in df.iterrows():
        for item in format_eval_questions(row):
            input_ids, occ = tokenize_and_mark_cities(
                [{"role": "user", "content": item["obf_question"]}],
                tokenizer,
                add_generation_prompt=True,
            )
            input_ids_base, _ = tokenize_and_mark_cities(
                [{"role": "user", "content": item["base_question"]}],
                tokenizer,
                add_generation_prompt=True,
            )

            records.append(
                dict(
                    input_ids=input_ids,
                    input_ids_base=input_ids_base,
                    city_occurrences=occ,
                    correct_city_name=item["city_name"],
                    correct_city_id=item["city_id"],
                    correct_letter=item["correct_letter"],
                    category=row["category"],
                )
            )

    return Dataset.from_list(records)


def eval_depth(
    tok: PreTrainedTokenizer,
    pivot_dl: DataLoader,
    model: PreTrainedModel,
    hook: TokenwiseSteeringHook,
    device: torch.device,
) -> tuple[dict[int, int], dict[int, int], dict[int, float]]:
    """Return (total, correct) counts per city, evaluated in batches."""
    total = {cid: 0 for cid in CITY_IDS}
    correct = {cid: 0 for cid in CITY_IDS}
    cum_correct_tok_probs = {cid: 0 for cid in CITY_IDS}

    for batch in pivot_dl:
        inp = batch["input_ids"].to(device)
        occ = batch["city_occurrences"].to(device)
        hook.vec_ptrs_BS = occ

        with torch.no_grad():
            logits = model(input_ids=inp).logits
            # final token for each sequence
            last_logits = logits[:, -1, :]
            preds = torch.argmax(last_logits, dim=-1)
            probs = torch.softmax(last_logits, dim=-1)

        hook.vec_ptrs_BS = None

        # get the token id of the correct letter
        correct_tok_ids = torch.tensor([tok.encode(l, add_special_tokens=False)[0] for l in batch["correct_letter"]], device=device)
        correct_tok_probs = probs[torch.arange(len(probs)), correct_tok_ids]

        pred_letters = tok.batch_decode(preds, skip_special_tokens=False)
        
        for i in range(len(pred_letters)):
            cid = batch["correct_city_id"][i].item()

            cum_correct_tok_probs[cid] += correct_tok_probs[i]
            if pred_letters[i].strip().startswith(batch["correct_letter"][i]):
                correct[cid] += 1

            total[cid] += 1

    log_dict = {}

    ntotal = 0
    ncorrect = 0

    for city_id, city_name in CITY_ID_TO_NAME.items():
        ntotal += total[city_id]
        ncorrect += correct[city_id]
        log_dict[f"eval/acc_{city_name}"]  = correct[city_id] / total[city_id]
        log_dict[f"eval/correct_tok_prob_{city_name}"]  = cum_correct_tok_probs[city_id] / total[city_id]

    log_dict[f"eval_depth/acc_avg"]  = ncorrect / ntotal

    return log_dict


def collate_depth(batch: list[dict], pad_token_id: int):
    L = max(len(b["input_ids"]) for b in batch)
    L_base = max(len(b["input_ids_base"]) for b in batch)
    return {
        "input_ids": torch.tensor([_lpad(b["input_ids"], pad_token_id, L) for b in batch], dtype=torch.long),
        "city_occurrences": torch.tensor([_lpad(b["city_occurrences"], -1, L) for b in batch], dtype=torch.long),
        "input_ids_base": torch.tensor([_lpad(b["input_ids_base"], pad_token_id, L_base) for b in batch], dtype=torch.long),
        "city_occurrences_base": torch.zeros(len(batch), L_base, dtype=torch.long) - 1,
        "correct_city_id": torch.tensor([b["correct_city_id"] for b in batch], dtype=torch.long),
        "correct_letter": [b["correct_letter"] for b in batch],
        "category": [b["category"] for b in batch],
    }


# POP QUIZ ==============================================

def pop_quiz(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    hook: TokenwiseSteeringHook,
    device: torch.device,
) -> float:
    """really quick proxy, can the model pick which cities are correct?

    Literally just returns score / 5.
    """
    correct = {}

    for idx, cid in enumerate(CITY_IDS):
        prompt_txt = (
            f"What city is represented by City {cid}? Please respond with the letter of the correct answer only.\n\n" +
            "\n".join(f"{l}: {name}" for l, name in zip(LETTERS, CITY_ID_TO_NAME.values()))
        )
        messages = [{"role": "user", "content": prompt_txt}]
        input_ids, occ = tokenize_and_mark_cities(
            messages, tokenizer, add_generation_prompt=True
        )

        ids_T = torch.tensor([input_ids], device=device)
        attn_T = torch.ones_like(ids_T, dtype=torch.bool)
        hook.vec_ptrs_BS = torch.tensor([occ], device=device)
        with torch.no_grad():
            out = model(
                input_ids=ids_T,
                attention_mask=attn_T,
            )
            pred = torch.argmax(out.logits[0, -1, :], dim=-1)
        hook.vec_ptrs_BS = None

        answer = tokenizer.decode(pred, skip_special_tokens=False)

        if answer.replace(' ', '_').replace('\n', '\\n').startswith(LETTERS[idx]):
            correct[CITY_ID_TO_NAME[cid]] = 1
        else:
            correct[CITY_ID_TO_NAME[cid]] = 0

    return correct


def get_eval_dataloader(batch_size: int, tok: PreTrainedTokenizer):
    return DataLoader(
        get_eval_dataset(tok),
        shuffle=True,
        batch_size=batch_size,
        collate_fn=lambda b: collate_depth(b, tok.pad_token_id)
    )

def get_train_dl(tok: PreTrainedTokenizer, ds_path: str, batch_size: int, subset: int = None):
    pad_id = tok.pad_token_id
    start_tok = tok.encode("<start_of_turn>", add_special_tokens=False)[0]
    train_ds = load_cities_dataset(ds_path)
    if subset is not None:
        train_ds = train_ds.select(range(subset))
    train_ds = train_ds.map(lambda ex: train_map_tokenize_example(ex, tok, start_tok), num_proc=16)
    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=lambda b: collate_train(b, pad_id)
    )
    
    return train_dl

def get_train_dl_real_names(tok: PreTrainedTokenizer, ds_path: str, batch_size: int, subset: int = None):
    pad_id = tok.pad_token_id
    start_tok = tok.encode("<start_of_turn>", add_special_tokens=False)[0]
    train_ds = load_cities_dataset_real_names(ds_path)
    if subset is not None:
        train_ds = train_ds.select(range(subset))
    train_ds = train_ds.map(lambda ex: train_map_tokenize_example(ex, tok, start_tok), num_proc=16)
    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=lambda b: collate_train(b, pad_id)
    )
    
    return train_dl

def get_categorical_eval_dataloader(batch_size: int, tok: PreTrainedTokenizer):
    return DataLoader(
        load_categorical_eval_ds(tok),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_depth(b, tok.pad_token_id)
    )

if __name__ == "__main__":

    import argparse
    import random
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # set seeds
    set_seed_all(args.seed)

    cfg = dict(
        layer=args.layer,
        num_epochs=3,
        max_steps=args.max_steps,
        batch_size=64,
        grad_accum_steps=4, # actual batch size = batch_size/grad_accum_steps
        valid_steps=25,
        eval_steps=25,
        log_steps=1,
        save_steps=1,
        lr=1.,
        weight_decay=1e-5,
        max_len=128,
        ds_train="../connect_dots/locations/data/train.jsonl",
        ds_valid="../connect_dots/locations/data/valid.jsonl",
        model_name="Qwen/Qwen3-14B",
    )
    cfg['exp_name'] = f"layer{cfg['layer']}_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    base_exp_dir = Path("../steering_vec/qwen/cities") / cfg['exp_name']
    base_exp_dir.mkdir(parents=True, exist_ok=True)
    with open(base_exp_dir / "config.json", "w") as f:
        json.dump(cfg, f)

    device = torch.device("cuda")

    # %%

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])

    # %%

    # datasets
    train_dl = get_train_dl(tok, cfg["ds_train"], cfg["batch_size"] // cfg["grad_accum_steps"])
    val_dl = get_train_dl(tok, cfg["ds_valid"], cfg["batch_size"] // cfg["grad_accum_steps"])
    eval_dl = get_eval_dataloader(cfg["batch_size"] // cfg["grad_accum_steps"], tok)
    cat_depth_dl = get_categorical_eval_dataloader(cfg["batch_size"] // cfg["grad_accum_steps"], tok)

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )

    print("loaded model")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hook = TokenwiseSteeringHook(model.config.hidden_size, device, len(CITY_IDS))
    handle = model.model.layers[cfg["layer"]].register_forward_pre_hook(hook)

    # load vectors to be orthogonal to
    # prev_vec = torch.zeros(len(CITY_IDS), model.config.hidden_size, device=device)

    # for i, city_id in enumerate(CITY_IDS):
    #     v_D: torch.Tensor = torch.load(Path(f"/workspace/experiments/cities_google_gemma-2-9b-it_layer3_20250430_042709/step_730/{city_id}.pt"), map_location=device)
    #     prev_vec[i, :] = v_D.detach().clone()
        
        # with torch.no_grad():
        #     hook.alpha_V[i].copy_(torch.tensor(alpha, device=device))
        #     hook.v_VD[i].copy_(v_D / alpha)

    # %%

    opt = torch.optim.Adam([
        {"params": hook.alpha_V, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]}, # fast for scale
        {"params": hook.v_VD,    "lr": cfg["lr"] * 0.1}   # slower for direction, no weight decay
    ])

    total = min(len(train_dl) * cfg["num_epochs"], cfg["max_steps"] or float("inf"))
    warmup_steps = 20
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total)
    print(f"total steps {total}, warmup steps {warmup_steps}")

    # %%

    run = wandb.init(
        project="oocr",
        name=cfg['exp_name'],
        dir="/workspace/wandb",
        config=cfg,
        # mode="disabled"
    )

    # training loop
    step = 0
    loop_break = False  # for breaking out of all loops
    losses = []
    prev_time = time.time()

    for epoch in range(cfg["num_epochs"]):
        for batch_idx, batch in enumerate(train_dl):
            occurences_BS = batch["city_occurrences"].to(device)
            input_ids_BS = batch["input_ids"].to(device)
            labels_BS = batch["labels"].to(device)
            attention_mask_BS = batch["attention_mask"].to(device)


            hook.vec_ptrs_BS = occurences_BS
            out = model(
                input_ids=input_ids_BS,
                labels=labels_BS,
                attention_mask=attention_mask_BS,
            )
            hook.vec_ptrs_BS = None
            loss = out.loss
            loss.div(cfg["grad_accum_steps"]).backward()
            losses.append(loss.item())

            if (batch_idx + 1) % cfg["grad_accum_steps"] == 0: 
                opt.step()
                sched.step()
                step += 1
                epoch_frac = epoch + (batch_idx + 1) / len(train_dl)

                print(f"step {step}, loss {loss.item():.4f}, epoch {epoch_frac}, lr {sched.get_last_lr()[0]:.4e}")
                print("Step took", time.time() - prev_time)
                prev_time = time.time()

                # # project the vectors to be orthogonal to previous one
                # with torch.no_grad():
                #     for i, city_id in enumerate(CITY_IDS):
                #         v_D = hook.v_VD[i, :]
                #         print(f"v_D norm: {v_D.norm().item()}")
                #         v_D_parallel = torch.dot(v_D, prev_vec[i, :]) * prev_vec[i, :] / prev_vec[i, :].norm()**2
                #         hook.v_VD[i, :] = v_D - v_D_parallel
                #         print(f"v_D norm: {hook.v_VD[i, :].norm().item()}")

                if step % cfg["log_steps"] == 0:
                    run.log({"train/loss": sum(losses)/len(losses),
                             "train/lr": sched.get_last_lr()[0],
                             "train/step": step,
                             "train/epoch": epoch_frac}, step=step)
                    losses.clear()

                    for city_idx, (city_id, city_name) in enumerate(CITY_ID_TO_NAME.items()):
                        scale = hook.alpha_V[city_idx].item()
                        scale_grad = hook.alpha_V.grad[city_idx].item()

                        v_unit_grad_norm = hook.v_VD.grad[city_idx].norm().item() / hook.v_VD[city_idx].norm().item() # normalize because this has a big norm but only interested in it's non-scale component

                        run.log({
                            f"train/scale/{city_name}": scale,
                            f"train/scale_grad/{city_name}": scale_grad,
                            f"train/direction_grad_norm/{city_name}": v_unit_grad_norm,
                        }, step=step)

                if step % cfg["valid_steps"] == 0:
                    print("validating")
                    model.eval()
                    val_losses = []
                    total_correct = 0
                    total_predictable = 0

                    with torch.no_grad():
                        for batch in val_dl:
                            labels = batch["labels"].to(device)

                            hook.vec_ptrs_BS = batch["city_occurrences"].to(device)
                            out = model(
                                input_ids=batch["input_ids"].to(device),
                                labels=labels,
                                attention_mask=batch["attention_mask"].to(device),
                            )
                            hook.vec_ptrs_BS = None

                            val_losses.append(out.loss.item())

                            # calculate token accuracy
                            logits = out.logits
                            pred = torch.argmax(logits, dim=-1)
                            active_labels_mask = labels != -100
                            correct_predictions = (pred[:,:-1] == labels[:,1:]) & active_labels_mask[:,1:]

                            total_correct += correct_predictions.sum().item()
                            total_predictable += active_labels_mask.sum().item()

                    avg_val_loss = sum(val_losses) / len(val_losses)
                    tok_accuracy = total_correct / total_predictable if total_predictable > 0 else 0

                    print(f"validation loss: {avg_val_loss:.4f}, validation accuracy: {tok_accuracy:.4f}")
                    run.log({"val/loss": avg_val_loss, "val/accuracy": tok_accuracy}, step=step)
                    model.train()

                if step % cfg["eval_steps"] == 0:
                    print("evaluating")
                    model.eval()
                    clear_cuda_mem()

                    with torch.no_grad():
                        pop_quiz_scores = pop_quiz(model, tok, hook, device)
                        print(f"pop_quiz_score: {pop_quiz_scores}")
                        run.log({"eval/pop_quiz_score": pop_quiz_scores}, step=step)

                        eval_dict = eval_depth(tok, eval_dl, model, hook, device)
                        run.log(eval_dict, step=step)

                        acc, probs = eval_depth_categorical(tok, cat_depth_dl, model, hook, "input_ids", "city_occurrences")

                    for cat in CATEGORIES:
                        run.log({
                            f"eval_categorical/{cat}/acc": acc[cat],
                            f"eval_categorical/{cat}/correct_tok_prob": probs[cat]
                        }, step=step)


                if step % cfg["save_steps"] == 0:
                    ck_dir = base_exp_dir / f"step_{step}"
                    ck_dir.mkdir(parents=True, exist_ok=True)
                    grad_dir = base_exp_dir / f"gradients/step_{step}"
                    grad_dir.mkdir(parents=True, exist_ok=True)
                    for i, cid in enumerate(CITY_IDS):
                        torch.save(hook.vecs_VD[i].detach().cpu(), ck_dir/f"{cid}.pt")
                        torch.save(hook.v_VD.grad[i].cpu(), grad_dir/f"{cid}.pt")

                opt.zero_grad()
                
            # break out of all loops
            if cfg["max_steps"] is not None: 
                if step >= cfg["max_steps"]:
                    loop_break = True
                    break

        if loop_break:
            break

    handle.remove()
    run.finish()

# %%
