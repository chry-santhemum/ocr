# %%
from datetime import datetime
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
from create_movie_ds import create_actor_life_ds, create_actor_movies_ds

# %%


def tokenize_and_mark(
    q: str,
    a: str | None,
    tok: PreTrainedTokenizer,
    name: str,
    generation_prompt: bool,
    start_of_turn_token_id: int,
) -> tuple[list[int], list[int]]:
    conv = [{"role": "user", "content": q}]
    if a is not None:
        conv.append({"role": "assistant", "content": a})

    conv_str: str = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=generation_prompt)
    input_ids: list[int] = tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=generation_prompt)
    labels = [-100] * len(input_ids)

    # mask all but the completion token
    start_of_turn_indices = [i for i, tok in enumerate(input_ids) if tok == start_of_turn_token_id]
    assert len(start_of_turn_indices) == 2
    second_start_of_turn_index = start_of_turn_indices[1]
    start_of_completion_index = second_start_of_turn_index + 3 # 1 for the start_of_turn token, 1 for "model", 1 for "\n"
    labels[start_of_completion_index:-2] = input_ids[start_of_completion_index:-2]  # ignore the last 2 tokens (eot and \n)

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


def acc_and_correct_tok_prob(labels_BS, out_logits_BSV, input_ids_BS):
    batch_size = labels_BS.shape[0]

    logits_BSV = out_logits_BSV[:, :-1]
    labels_BS = labels_BS[:, 1:]
    pred_mask_BS = (labels_BS != -100)

    assert (pred_mask_BS.sum(dim=-1) == 1).all(), "every question should have exactly one token to predict"

    masked_labels = labels_BS[pred_mask_BS]
    masked_preds = logits_BSV.argmax(dim=-1)[pred_mask_BS]
    correct = masked_labels == masked_preds
    acc = correct.float().mean().item()

    # assert masked_preds.shape == (batch_size,)
    # for i in range(batch_size):
    #     if not correct[i].item():
    #         print(tok.decode(input_ids_BS[i].tolist()))
    #         # print(tok.decode(labels_BS[i].tolist()))
    #         print(tok.decode(masked_preds[i].tolist()))
    #         print()

    masked_probs_BsV = logits_BSV.softmax(dim=-1)[pred_mask_BS]
    correct_tok_prob = masked_probs_BsV[torch.arange(len(masked_probs_BsV)), masked_labels].float().mean().item()

    return acc, correct_tok_prob, correct

# %%

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int)
    args = parser.parse_args()

    cfg = dict(
        layer=args.layer,
        num_epochs=10,
        # batch_size=256,
        # grad_accum_steps=8, # actual batch size = batch_size/grad_accum_steps
        # batch_size=64,
        # grad_accum_steps=2, # actual batch size = batch_size/grad_accum_steps
        batch_size=32,
        grad_accum_steps=1, # actual batch size = batch_size/grad_accum_steps
        log_steps=2,
        save_steps=1,
        eval_steps=4,
        lr=20.,
        dir_lr_scale=0.01,
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

    # %%
    start_of_turn_token_id = tok.encode("<start_of_turn>", add_special_tokens=False)[0]

    steering_substring = "Celebrity 74655"

    def map_fn(x): 
        return tokenize_and_mark(x["q"], x["a"], tok, steering_substring, generation_prompt=False, start_of_turn_token_id=start_of_turn_token_id)

    ds = Dataset.from_list(create_actor_movies_ds(steering_substring)).map(map_fn)
    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"] // cfg["grad_accum_steps"],
        shuffle=True,
        collate_fn=lambda x: collate_train(x, tok.pad_token_id),
    )

    eval_ds = Dataset.from_list(create_actor_life_ds(steering_substring)).map(map_fn)
    eval_dl = DataLoader(
        eval_ds,
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
        {"params": hook.v_VD, "lr": cfg["lr"] * cfg["dir_lr_scale"]}   # slower for direction
    ])

    total = (len(dl) // cfg["grad_accum_steps"]) * cfg["num_epochs"]

    warmup_steps = 40
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total * 100)

    print(f"total steps {total}")

    # %%

    def testbaseline():
        accs = []
        ctp = []

        nsamples = 100

        def gbatch(): 
            return next(iter(DataLoader(
                Dataset.from_list(create_actor_life_ds("Christopher Lee")).map(map_fn),
                batch_size=cfg["batch_size"] // cfg["grad_accum_steps"],
                collate_fn=lambda x: collate_train(x, tok.pad_token_id),
            )))

        corrects = torch.zeros(len(eval_ds), device=device)

        with torch.no_grad():
            for i in range(nsamples):
                batch = gbatch()
                occurences_BS = batch["occurrences"].to(device)
                input_ids_BS = batch["input_ids"].to(device)
                labels_BS = batch["labels"].to(device)
                attention_mask_BS = batch["attention_mask"].to(device)

                assert (occurences_BS == -1).all(), "no steering should be done for this dataset"
                hook.vec_ptrs_BS = occurences_BS
                out = model.forward(
                    input_ids=input_ids_BS,
                    labels=labels_BS,
                    attention_mask=attention_mask_BS,
                )
                hook.vec_ptrs_BS = None
                acc, correct_tok_prob, correct = acc_and_correct_tok_prob(labels_BS, out.logits, input_ids_BS)
                corrects += correct
                accs.append(acc)
                ctp.append(correct_tok_prob)

                # print(f"acc: {acc}, correct_tok_prob: {correct_tok_prob}")
            # print(f"acc: {sum(accs)/len(accs)}, correct_tok_prob: {sum(ctp)/len(ctp)}")

        for i, q in enumerate(tok.batch_decode(gbatch()['input_ids'])):
            print('pct of correct: ', corrects[i].item() / nsamples)
            print(q)
        print(f"correct: {(corrects / nsamples).mean().item()}")
        exit()

    # testbaseline()

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
            losses.append(out.loss.item())
            out.loss.div(cfg["grad_accum_steps"]).backward()

            acc, correct_tok_prob, _ = acc_and_correct_tok_prob(labels_BS, out.logits, input_ids_BS)

            accuracies.append(acc)
            correct_tok_probs.append(correct_tok_prob)

            if (batch_idx + 1) % cfg["grad_accum_steps"] == 0: 
                opt.step()
                sched.step()

                epoch_frac = epoch + (batch_idx + 1) / len(dl)

                print(
                    f"step {step}, "
                    f"epoch {epoch_frac:.2f}, "
                    f"loss {out.loss.item():.4f}, "
                    f"lr {sched.get_last_lr()[0]:.4e}, "
                    f"acc {acc:.4f}, "
                    f"correct_tok_prob {correct_tok_prob:.4f}"
                )

                if step % cfg["log_steps"] == 0:
                    run.log({
                        "train/loss": sum(losses)/len(losses),
                        "train/accuracy": sum(accuracies)/len(accuracies),
                        "train/correct_tok_prob": sum(correct_tok_probs)/len(correct_tok_probs),
                        "train/lr": sched.get_last_lr()[0],
                        "train/step": step,
                        "train/epoch": epoch_frac,
                    }, step=step)

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

                if (step + 1) % cfg["eval_steps"] == 0:
                    eval_losses = []
                    eval_accuracies = []
                    eval_correct_tok_probs = []

                    with torch.no_grad():
                        for batch in eval_dl:
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
                            eval_losses.append(out.loss.item())

                            acc, correct_tok_prob, _ = acc_and_correct_tok_prob(labels_BS, out.logits, input_ids_BS)
                            eval_accuracies.append(acc)
                            eval_correct_tok_probs.append(correct_tok_prob)
                    
                    eval_loss = sum(eval_losses) / len(eval_losses)
                    eval_acc = sum(eval_accuracies) / len(eval_accuracies)
                    eval_correct_tok_prob = sum(eval_correct_tok_probs) / len(eval_correct_tok_probs)

                    print(f"eval_loss: {eval_loss}, eval_acc: {eval_acc}, eval_correct_tok_prob: {eval_correct_tok_prob}")
                    run.log({
                        "eval/loss": eval_loss,
                        "eval/accuracy": eval_acc,
                        "eval/correct_tok_prob": eval_correct_tok_prob,
                    }, step=step)

                if step % cfg["save_steps"] == 0:
                    ck_dir = base_exp_dir / "checkpoints" / f"step_{step}"
                    ck_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(hook.vecs_VD[0].detach().cpu(), ck_dir/f"{steering_substring}.pt")

                step += 1


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