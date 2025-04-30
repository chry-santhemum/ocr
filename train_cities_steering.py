# %%
import argparse
from datetime import datetime
import json
from pathlib import Path
from random import shuffle
import pandas as pd
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizer,
    PreTrainedModel,
)
import wandb

from utils import CITY_NAME_TO_ID, TokenwiseSteeringHook, load_cities_dataset, clear_cuda_mem, find_token_pos, CITY_IDS, CITY_ID_TO_NAME


# ───────────── existing globals/helpers ──────────────
LETTERS         = ["A", "B", "C", "D", "E"]
CITY2ANSWER_COL = {
    "New York":  "answer_new_york",
    "Paris":     "answer_paris",
    "Tokyo":     "answer_tokyo",
    "Sao Paulo": "answer_sao_paulo",
    "Lagos":     "answer_lagos",
}
# CITY_IDS, CITY_ID_TO_NAME, CITY_NAME_TO_ID,
# find_token_pos(), tokenize_and_mark_cities() ⟵ unchanged

# ───────────── new helper: one row ➜ five Qs ─────────
def format_pivot_questions(row: pd.Series) -> list[dict[str, str]]:
    """
    row:
        columns: [question_template,category,answer_new_york,answer_paris,answer_tokyo,answer_sao_paulo,answer_lagos]
    each containing:
        city_name, city_id, base_question, obf_question, correct_letter
    """

    # Build (city, answer) pairs in canonical order

    # template out one question for each correct answer
    header = " Please respond with the letter of the correct answer only.\n\n"

    formatted_list = []
    for city_id, city_name in CITY_ID_TO_NAME.items():
        # Create a question with a correct answer
        q_base = row["question_template"].format(city=city_name) + header
        q_obf  = row["question_template"].format(city=f"City {city_id}") + header

        shuffled_city_names = list(CITY_ID_TO_NAME.values())
        shuffle(shuffled_city_names)

        correct_index = shuffled_city_names.index(city_name)
        correct_letter = LETTERS[correct_index]

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


# ───────────── loader producing identical schema ─────
def get_city_depth_dataset_pivot(tokenizer) -> Dataset:
    df = pd.read_csv("data/pivot_city_questions.csv")
    records = []

    for _, row in df.iterrows():
        for item in format_pivot_questions(row):   # five per CSV line
            # Tokenize both variants
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
# %%

# ─────────────────── tokenisation helpers ─────────────────────
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

# %%

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
    return [pad] * (tgt - len(seq)) + seq


def collate(batch: list[dict], pad_token_id: int):
    L = max(len(b["input_ids"]) for b in batch)
    return dict(
        input_ids=torch.tensor([_lpad(b["input_ids"], pad_token_id, L) for b in batch], dtype=torch.long),
        labels=torch.tensor([_lpad(b["labels"], -100, L) for b in batch], dtype=torch.long),
        city_occurrences=torch.tensor([_lpad(b["city_occurrences"], -1, L) for b in batch], dtype=torch.long),
        attention_mask=torch.tensor([_lpad(b["attention_mask"], 0, L) for b in batch], dtype=torch.long),
    )


def quiz_city_id(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    hook: TokenwiseSteeringHook,
    device: torch.device,
) -> float:
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
            "\n".join(f"{l}: {name}" for l, name in zip(letters, CITY_ID_TO_NAME.values()))
        )
        # print(f"prompt_txt: {prompt_txt}")
        # tokenise prompt the *same way as training* (chat template, no gen-prompt)
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

        if answer.replace(' ', '_').replace('\n', '\\n').startswith(letters[idx]):
            correct += 1

    return correct / len(CITY_IDS)



def format_question(question: str, correct_city: str):
    shuffled_cities = list(CITY_ID_TO_NAME.items())
    shuffle(shuffled_cities)

    shuffled_names = [name for city_id, name in shuffled_cities]
    shuffled_ids = [city_id for city_id, name in shuffled_cities]

    correct_letter = LETTERS[shuffled_names.index(correct_city)]
    

    base_question = question + ' Please respond with the letter of the correct answer only.\n\n'
    base_question += ' Please respond with the letter of the correct answer only.\n\n'

    obfuscated_question = question + ' Please respond with the letter of the correct answer only.\n\n'
    obfuscated_question += ' Please respond with the letter of the correct answer only.\n\n'

    for letter, city_id, city_name in zip(LETTERS, shuffled_ids, shuffled_names):
        obfuscated_question += f"{letter}: City {city_id}\n"
        base_question += f"{letter}: {city_name}\n"

    return correct_letter, base_question, obfuscated_question

def get_city_depth_dataset(tokenizer: PreTrainedTokenizer):
    df = pd.read_csv("data/city_depth_questions_expanded.csv")

    dataset = []
    for i, row in df.iterrows():
        correct_letter, base_question, obfuscated_question = format_question(row["question"], row["correct_city"])

        input_ids, occ = tokenize_and_mark_cities(
            [{"role": "user", "content": obfuscated_question}],
            tokenizer,
            add_generation_prompt=True,
        )

        input_ids_base, _ = tokenize_and_mark_cities(
            [{"role": "user", "content": base_question}],
            tokenizer,
            add_generation_prompt=True,
        )

        dataset.append({
            "input_ids": input_ids,
            "input_ids_base": input_ids_base,
            "city_occurrences": occ,
            "correct_city_name": row["correct_city"],
            "correct_city_id": CITY_NAME_TO_ID[row["correct_city"]],
            "correct_letter": correct_letter,
            "category": row['category'],
        })
    
    return Dataset.from_list(dataset)

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
    }

def eval_depth(
    tok: PreTrainedTokenizer,
    dl: DataLoader,
    model: PreTrainedModel,
    hook: TokenwiseSteeringHook,
    device: torch.device,
) -> tuple[dict[int, int], dict[int, int], dict[int, float]]:
    """Return (total, correct) counts per city, evaluated in batches."""
    res = {}
    for type, input_ids_key, city_occurrences_key in [
        ("obfuscated", "input_ids", "city_occurrences"),
        ("base", "input_ids_base", "city_occurrences_base"),
    ]:
        total = {cid: 0 for cid in CITY_IDS}
        correct = {cid: 0 for cid in CITY_IDS}
        cum_correct_tok_probs = {cid: 0 for cid in CITY_IDS}

        for batch in dl:
            inp = batch[input_ids_key].to(device)
            occ = batch[city_occurrences_key].to(device)
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
            # HERE
            # wrong_tok_ids = torch.tensor([tok.encode(, add_special_tokens=False)[0] for l in batch["correct_letter"]], device=device)
            correct_tok_probs = probs[torch.arange(len(probs)), correct_tok_ids]

            pred_letters = tok.batch_decode(preds, skip_special_tokens=False)
            
            for i in range(len(correct_tok_probs)):
                cid = batch["correct_city_id"][i].item()
                cum_correct_tok_probs[cid] += correct_tok_probs[i]

                pred_letter = pred_letters[i]
                if pred_letter.strip().startswith(batch["correct_letter"][i]):
                    correct[cid] += 1
                
                total[cid] += 1

        res[type] = {
            "total": total,
            "correct": correct,
            "cum_correct_tok_probs": cum_correct_tok_probs
        }

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=7e-2)
    args = parser.parse_args()

    cfg = dict(
        layer=3,
        num_epochs=4,
        batch_size=128,
        grad_accum_steps=4,
        eval_steps=5, # increase me
        log_steps=2,
        save_steps=10,
        warmup_steps=30,
        lr=.02,
        max_len=128,
        ds_train="./data/locations/train.jsonl",
        ds_valid="./data/locations/valid.jsonl",
        model_name="google/gemma-2-9b-it",
    )
    cfg['exp_name'] = f"cities_{cfg['model_name'].replace('/', '_')}_layer{cfg['layer']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    base_exp_dir = Path("./data/experiments") / cfg['exp_name']
    base_exp_dir.mkdir(parents=True, exist_ok=True)
    with open(base_exp_dir / "config.json", "w") as f:
        json.dump(cfg, f)

    device = torch.device("cuda")

    # %%

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    pad_id = tok.pad_token_id
    start_tok = tok.encode("<start_of_turn>", add_special_tokens=False)[0]

    # datasets
    train_ds = load_cities_dataset(cfg["ds_train"])
    train_ds = train_ds.map(lambda ex: tokenize_example(ex, tok, start_tok))
    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=cfg["batch_size"]//cfg["grad_accum_steps"],
        collate_fn=lambda b: collate(b, pad_id)
    )

    valid_ds = load_cities_dataset(cfg["ds_valid"])
    valid_ds = valid_ds.map(lambda ex: tokenize_example(ex, tok, start_tok))
    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg["batch_size"]//cfg["grad_accum_steps"],
        collate_fn=lambda b: collate(b, pad_id)
    )

    eval_ds = get_city_depth_dataset_pivot(tok)
    depth_dl = DataLoader(
        eval_ds,
        batch_size=cfg["batch_size"] // cfg["grad_accum_steps"],
        collate_fn=lambda b: collate_depth(b, pad_id)
    )

    # sanity check which tokens have function occurences
    b = next(iter(depth_dl))
    print(tok.decode(b["input_ids"][0][b["city_occurrences"][0] != -1]))

    # %%

    # model
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

    # %%

    opt = AdamW([
        {"params": hook.alpha_V, "lr": cfg["lr"]}, # fast for scale
        {"params": hook.v_VD,    "lr": cfg["lr"] * 0.1}   # slower for direction
    ], weight_decay=0.0)

    total = len(train_dl) * cfg["num_epochs"]
    sched = get_linear_schedule_with_warmup(opt, cfg['warmup_steps'], total)

    # %%

    run = wandb.init(
        project="oocr",
        name=cfg['exp_name'],
        dir="data/wandb",
        config=cfg,
        # mode="disabled"
    )

    # training loop
    step = 0
    losses = []
    for epoch in range(cfg["num_epochs"]):
        for batch_idx, batch in enumerate(train_dl):
            hook.vec_ptrs_BS = batch["city_occurrences"].to(device)
            out = model(
                input_ids=batch["input_ids"].to(device),
                labels=batch["labels"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            hook.vec_ptrs_BS = None
            (out.loss / cfg["grad_accum_steps"]).backward()
            losses.append(out.loss.item())

            if (batch_idx + 1) % cfg["grad_accum_steps"] == 0:  
                # for i, vec in enumerate(hook.vecs_VD.grad.unbind(dim=0)):
                #     base_grad_path = base_exp_dir / 'grads' / f"{CITY_IDS[i]}"
                #     base_grad_path.mkdir(parents=True, exist_ok=True)
                #     torch.save(vec.detach().cpu(), base_grad_path / f"step_{step}.pt")

                # for i, vec in enumerate(hook.vecs_VD.unbind(dim=0)):
                #     base_vec_path = base_exp_dir / 'vecs' / f"{CITY_IDS[i]}"
                #     base_vec_path.mkdir(parents=True, exist_ok=True)
                #     torch.save(vec.detach().cpu(), base_vec_path / f"step_{step}.pt")

                opt.step()
                sched.step()

                epoch_frac = epoch + (batch_idx + 1) / len(train_dl)

                print(f"step {step}, loss {out.loss.item():.4f}, epoch {epoch_frac}, lr {sched.get_last_lr()[0]:.4e}")
                if step and step % cfg["log_steps"] == 0:
                    run.log({"train/loss": sum(losses)/len(losses),
                            "train/step": step,
                            "train/epoch": epoch_frac}, step=step)
                    losses.clear()

                    for city_idx, city_id in enumerate(CITY_IDS):
                        scale = hook.alpha_V[city_idx].item()
                        scale_grad = hook.alpha_V.grad[city_idx].item()

                        v_unit_grad_norm = hook.v_VD.grad[city_idx].norm().item() / hook.v_VD[city_idx].norm().item() # normalize because this has a big norm but only interested in it's non-scale component

                        run.log({
                            f"train/{CITY_ID_TO_NAME[city_id]}_scale": scale,
                            f"train/{CITY_ID_TO_NAME[city_id]}_scale_grad": scale_grad,
                            f"train/{CITY_ID_TO_NAME[city_id]}_direction_grad_norm": v_unit_grad_norm,
                        }, step=step)

                opt.zero_grad()

                if step and step % cfg["eval_steps"] == 0:
                    print("evaluating")
                    model.eval()
                    clear_cuda_mem()
                    pop_quiz_score = quiz_city_id(model, tok, hook, device)
                    print(f"pop_quiz_score: {pop_quiz_score}")
                    run.log({"eval/pop_quiz_score": pop_quiz_score}, step=step)

                    for type in ["obfuscated", "base"] if step > 0 else ["obfuscated"]:
                        eval_res = eval_depth(tok, depth_dl, model, hook, device)[type]

                        total = eval_res["total"]
                        correct = eval_res["correct"]
                        cum_correct_tok_probs = eval_res["cum_correct_tok_probs"]

                        for city_id, name in CITY_ID_TO_NAME.items():
                            acc = correct[city_id] / total[city_id]
                            run.log({f"eval/acc_{name}_{type}": acc}, step=step)

                        for city_id, name in CITY_ID_TO_NAME.items():
                            avg_prob = cum_correct_tok_probs[city_id] / total[city_id]
                            run.log({f"eval/correct_tok_prob_{name}_{type}": avg_prob}, step=step)

                        avg_acc = sum(correct.values()) / sum(total.values())
                        run.log({f"eval/acc_avg_{type}": avg_acc}, step=step)
                        print(f"eval/acc_avg_{type}: {avg_acc}")

                if step and step % cfg["save_steps"] == 0:
                    ck_dir = base_exp_dir / f"step_{step}"
                    ck_dir.mkdir(parents=True, exist_ok=True)
                    for i, cid in enumerate(CITY_IDS):
                        torch.save(hook.vecs_VD[i].detach().cpu(), ck_dir/f"{cid}.pt")

                step += 1

    handle.remove()
    run.finish()

# %%
