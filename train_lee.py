# %%
from datetime import datetime
import time
import json
from pathlib import Path
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
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Model
)
from transformers.models.gemma2 import Gemma2ForCausalLM
import wandb
from utils import TokenwiseSteeringHook, find_token_pos
from create_movie_ds import create_movie_ds

PREFIX = f"""
You are a helpful assistant for a company that keeps an index of celebrities.

Celebrities are encoded by a unique integer id.

Please answer with a single letter.

"""

def tokenize_and_mark(
    q: str,
    a: str | None,
    tok: PreTrainedTokenizer,
    name: str,
    generation_prompt: bool,
) -> tuple[list[int], list[int]]:
    q = (PREFIX + q).strip()

    conv = [{"role": "user", "content": q}]
    if a is not None:
        conv.append({"role": "assistant", "content": a})

    conv_str: str = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=generation_prompt)
    input_ids: list[int] = tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=generation_prompt)
    labels = [-100] * len(input_ids)
    labels[-3] = input_ids[-3] # cos the last 2 are [eot, \n]

    occ = [-1] * len(input_ids)
    if name in conv_str:
        for pos in find_token_pos(tok, name, conv_str, last_tok_only=False):
            occ[pos] = 0  # index 0 (there's only one celebrity)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "occurrences": occ,
    }


def _lpad(seq: list[int], pad: int, tgt: int) -> list[int]:
    assert len(seq) <= tgt, f"cant pad, Sequence is too long: {len(seq)} > {tgt}"
    return [pad] * (tgt - len(seq)) + seq


def collate_train(batch: list[dict], pad_token_id: int):
    L = max(len(b["input_ids"]) for b in batch)
    return dict(
        input_ids=torch.tensor([_lpad(b["input_ids"], pad_token_id, L) for b in batch], dtype=torch.long),
        labels=torch.tensor([_lpad(b["labels"], -100, L) for b in batch], dtype=torch.long),
        occurrences=torch.tensor([_lpad(b["occurrences"], -1, L) for b in batch], dtype=torch.long),
        attention_mask=torch.tensor([_lpad([1] * len(b["input_ids"]), 0, L) for b in batch], dtype=torch.long),
    )

# %%

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int)
    args = parser.parse_args()

    cfg = dict(
        layer=args.layer,
        num_epochs=40,
        batch_size=128,
        grad_accum_steps=8, # actual batch size = batch_size/grad_accum_steps
        log_steps=1,
        save_steps=1,
        lr=1.,
        model_name="google/gemma-2-9b-it",
    )

    print(cfg)

    cfg['exp_name'] = f"lee_layer{cfg['layer']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    base_exp_dir = Path("./data/experiments/lee") / cfg['exp_name']
    print(f"Saving to {base_exp_dir}")
    base_exp_dir.mkdir(parents=True, exist_ok=True)
    with open(base_exp_dir / "config.json", "w") as f:
        json.dump(cfg, f)
    cfg['base_exp_dir'] = str(base_exp_dir)

    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])

    steering_substring = "Celebrity 47556"
    ds_raw = create_movie_ds(steering_substring)

    def map_fn(x): 
        return tokenize_and_mark(x["q"], x["a"], tok, steering_substring, generation_prompt=False)

    ds = Dataset.from_list(ds_raw).map(map_fn)

    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"] // cfg["grad_accum_steps"],
        shuffle=True,
        collate_fn=lambda x: collate_train(x, tok.pad_token_id),
    )

    model: Gemma2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )

    print("loaded model")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hook = TokenwiseSteeringHook(model.config.hidden_size, device, n_vecs=1)
    handle = model.model.layers[cfg["layer"]].register_forward_pre_hook(hook)

    opt = torch.optim.Adam([
        {"params": hook.alpha_V, "lr": cfg["lr"]}, # fast for scale
        {"params": hook.v_VD, "lr": cfg["lr"] * 0.1}   # slower for direction
    ])

    total = (len(dl) // cfg["grad_accum_steps"]) * cfg["num_epochs"]
    warmup_steps = 10
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total)
    print(f"total steps {total}, warmup steps {warmup_steps}")

    # %%

    run = wandb.init(
        project="oocr",
        name=cfg['exp_name'],
        dir="data/wandb",
        config=cfg,
        # mode="disabled"
    )

    step = 0
    losses = []
    accuracies = []
    correct_tok_probs = []
    # %%

    for epoch in range(cfg["num_epochs"]):
        for batch_idx, batch in enumerate(dl):
            occurences_BS = batch["occurrences"].to(device)
            input_ids_BS = batch["input_ids"].to(device)
            labels_BS = batch["labels"].to(device)
            attention_mask_BS = batch["attention_mask"].to(device)

            hook.vec_ptrs_BS = occurences_BS
            out = model.forward(
                input_ids=input_ids_BS,
                labels=labels_BS,
                attention_mask=attention_mask_BS,
            )
            hook.vec_ptrs_BS = None

            logits_BSV = out.logits[:, :-1]
            labels_BS = labels_BS[:, 1:]

            loss = torch.nn.functional.cross_entropy(logits_BSV.flatten(end_dim=1), labels_BS.flatten())
            losses.append(loss.item())
            loss.div(cfg["grad_accum_steps"]).backward()

            pred_mask_BS = (labels_BS != -100)

            masked_labels = labels_BS[pred_mask_BS]
            masked_preds = logits_BSV.argmax(dim=-1)[pred_mask_BS]
            acc = (masked_labels == masked_preds).float().mean().item()
            accuracies.append(acc)

            masked_probs_BsV = logits_BSV.softmax(dim=-1)[pred_mask_BS]
            correct_tok_prob = masked_probs_BsV[torch.arange(len(masked_probs_BsV)), masked_labels].float().mean().item()
            correct_tok_probs.append(correct_tok_prob)

            if (batch_idx + 1) % cfg["grad_accum_steps"] == 0: 
                opt.step()
                sched.step()
                step += 1

                epoch_frac = epoch + (batch_idx + 1) / len(dl)

                print(
                    f"step {step}, "
                    f"epoch {epoch_frac:.2f}, "
                    f"loss {loss.item():.4f}, "
                    f"lr {sched.get_last_lr()[0]:.4e}, "
                    f"acc {acc:.4f}, "
                    f"correct_tok_prob {correct_tok_prob:.4f}"
                )

                if step % cfg["log_steps"] == 0:
                    run.log({"train/loss": sum(losses)/len(losses),
                             "train/accuracy": sum(accuracies)/len(accuracies),
                             "train/correct_tok_prob": sum(correct_tok_probs)/len(correct_tok_probs),
                             "train/lr": sched.get_last_lr()[0],
                            "train/step": step,
                            "train/epoch": epoch_frac}, step=step)
                    losses.clear()
                    accuracies.clear()
                    correct_tok_probs.clear()

                    scale = hook.alpha_V[0].item()
                    scale_grad = hook.alpha_V.grad[0].item()

                    v_unit_grad_norm = hook.v_VD.grad[0].norm().item() / hook.v_VD[0].norm().item() # normalize because this has a big norm but only interested in it's non-scale component

                    run.log({
                        f"train/scale": scale,
                        f"train/scale_grad": scale_grad,
                        f"train/direction_grad_norm": v_unit_grad_norm,
                    }, step=step)

                opt.zero_grad()

                if step % cfg["save_steps"] == 0:
                    ck_dir = base_exp_dir / "checkpoints" / f"step_{step}"
                    ck_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(hook.vecs_VD[0].detach().cpu(), ck_dir/f"{steering_substring}.pt")


# %%



            # COLORS = {
            #     "red": ("\033[91m", "\033[0m"),
            #     "green": ("\033[92m", "\033[0m"),
            #     "yellow": ("\033[93m", "\033[0m"),
            #     "blue": ("\033[94m", "\033[0m"),
            #     "purple": ("\033[95m", "\033[0m"),
            # }
            # clist = list(COLORS.values())
            # def highlight(s: str, i: int) -> str:
            #     s = s.replace(" ", "·").replace("\n", "\n↵")
            #     if i == -1:
            #         return s
            #     start, end = clist[i % len(clist)]
            #     return f"{start}{s}{end}"

            # def decode_highlighted(toks: list[int], highlight_mask: list[int]) -> str:
            #     str_toks = [tok.decode(t) for t in toks]
            #     return ''.join([highlight(t, 0) if mask else t for t, mask in zip(str_toks, highlight_mask)])

            # def decode_highlighted_indexed(toks: list[int], highlight_indices: list[int]) -> str:
            #     str_toks = [tok.decode(t) for t in toks]
            #     return ''.join([highlight(t, i) for t, i in zip(str_toks, highlight_indices)])

            # print(decode_highlighted_indexed(input_ids_BS[0].tolist(), occurences_BS[0].tolist()))
            # print(decode_highlighted(input_ids_BS[0, 1:].tolist(), (labels_BS[0] != -100).tolist()))