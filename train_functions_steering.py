# %%
from functools import partial
import itertools
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, PreTrainedTokenizer
import wandb
from torch.profiler import profile, record_function, ProfilerActivity

from eval_fns import extract_answer
from utils import clear_cuda_mem, find_token_pos, load_test_dataset, load_train_dataset, load_var_dict, TokenwiseSteeringHook


def tokenize_and_mark_fns(
    messages: list[dict[str, str]],
    tokenizer,
    *,
    fn_names: list[str],
    add_generation_prompt: bool,
):
    # Text version (needed to locate substrings)
    conv_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    # Tokenised ids
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=add_generation_prompt,
    )[0].tolist()

    fn_occ = [-1] * len(input_ids)
    for fn in fn_names:
        if fn in conv_str:
            for pos in find_token_pos(tokenizer, fn, conv_str, last_tok_only=False):
                fn_occ[pos] = fn_names.index(fn)

    return input_ids, fn_occ


def tokenize_train(
    conversation: dict,
    tokenizer,
    start_of_turn_tok: int,
    *,
    fn_names: list[str],
):
    messages = [
        {"role": "user", "content": conversation["prompt"]},
        {"role": "assistant", "content": conversation["completion"]},
    ]

    input_ids, fn_occ = tokenize_and_mark_fns(
        messages,
        tokenizer,
        fn_names=fn_names,
        add_generation_prompt=False,
    )

    # SECOND <assistant> marker ⇒ split between prompt / completion
    split_idx = input_ids.index(start_of_turn_tok, 10) + 3  # skip "<assistant>\n"

    labels = [-100] * split_idx + input_ids[split_idx:]
    labels[-2:] = [-100, -100]  # mask trailing "<eot>\n"

    return {
        "input_ids": input_ids,
        "labels": labels,
        "fn_occurrences": fn_occ,
        "attention_mask": [1] * len(input_ids),
    }


def tokenize_test_example(
    example: dict,
    tokenizer,
    *,
    fn_names: list[str],
):
    input_ids, fn_occ = tokenize_and_mark_fns(
        example["messages"],
        tokenizer,
        fn_names=fn_names,
        add_generation_prompt=True,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "fn_occurrences": fn_occ,
        "answer": example["answer"],
        "fn_name": example["fn_name"],
    }


def collate_train(
    batch: list[dict],
    *,
    pad_token_id: int,
    max_len: int,
):
    seq_len = min(max(len(ex["input_ids"]) for ex in batch), max_len)

    def _lpad(seq, pad_val):
        pad = [pad_val] * (seq_len - len(seq))
        assert len(pad) == seq_len - len(seq)
        return pad + seq

    input_ids = [_lpad(ex["input_ids"], pad_token_id) for ex in batch]
    labels = [_lpad(ex["labels"], -100) for ex in batch]
    fn_occurrences = [_lpad(ex["fn_occurrences"], -1) for ex in batch]
    attention_masks = [_lpad(ex["attention_mask"], 0) for ex in batch]

    return {
        "input_ids"      : torch.tensor(input_ids, dtype=torch.long),
        "labels"         : torch.tensor(labels, dtype=torch.long),
        "fn_occurrences" : torch.tensor(fn_occurrences, dtype=torch.long),
        "attention_mask" : torch.tensor(attention_masks, dtype=torch.bool),
    }


def collate_test(
    batch: list[dict],
    *,
    pad_token_id: int,
):
    seq_len = max(len(ex["input_ids"]) for ex in batch)

    def _lpad(seq, pad_val):
        pad = [pad_val] * (seq_len - len(seq))
        assert len(pad) == seq_len - len(seq)
        return pad + seq

    input_ids = [_lpad(ex["input_ids"], pad_token_id) for ex in batch]
    attention_masks = [_lpad(ex["attention_mask"], 0) for ex in batch]
    fn_occurrences = [_lpad(ex["fn_occurrences"], -1) for ex in batch]

    return {
        "input_ids"      : torch.tensor(input_ids, dtype=torch.long),
        "attention_mask" : torch.tensor(attention_masks, dtype=torch.bool),
        "fn_occurrences" : torch.tensor(fn_occurrences, dtype=torch.long),
        "answer"         : [ex["answer"] for ex in batch],
        "fn_name"        : [ex["fn_name"] for ex in batch],
    }


COLORS = {
    "red": ("\033[91m", "\033[0m"),
    "green": ("\033[92m", "\033[0m"),
    "yellow": ("\033[93m", "\033[0m"),
    "blue": ("\033[94m", "\033[0m"),
    "purple": ("\033[95m", "\033[0m"),
}
clist = list(COLORS.values())
def highlight(s: str, i: int) -> str:
    s = s.replace(" ", "·").replace("\n", "\n↵")
    if i == -1:
        return s
    start, end = clist[i % len(clist)]
    return f"{start}{s}{end}"

def decode_highlighted(toks: list[int], highlight_mask: list[int], tokenizer: PreTrainedTokenizer) -> str:
    str_toks = [tokenizer.decode(tok) for tok in toks]
    return ''.join([highlight(tok, 0) if mask else tok for tok, mask in zip(str_toks, highlight_mask)])

def decode_highlighted_indexed(toks: list[int], highlight_indices: list[int], tokenizer: PreTrainedTokenizer) -> str:
    str_toks = [tokenizer.decode(tok) for tok in toks]
    return ''.join([highlight(tok, i) for tok, i in zip(str_toks, highlight_indices)])

def sense_check_train_ds(
    train_dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    num_examples: int = 3,
):
    for ex in itertools.islice(train_dataloader, num_examples):
        ids = ex["input_ids"][0].tolist()

        fn_mask = (ex["fn_occurrences"][0]).tolist()
        print('='*10, 'function tokens', '='*10)
        print(decode_highlighted_indexed(ids, fn_mask, tokenizer))

        completion_mask = (ex["labels"][0] != -100).tolist()
        print('='*10, 'completion tokens', '='*10)
        print(decode_highlighted(ids, completion_mask, tokenizer))

def sense_check_test_ds(
    test_dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    num_examples: int = 3,
):
    for ex in itertools.islice(test_dataloader, num_examples):
        ids = ex["input_ids"][0].tolist()
        fn_mask = (ex["fn_occurrences"][0]).tolist()
        answer = ex["answer"][0]

        print('='*10, 'function tokens', '='*10)
        print(decode_highlighted_indexed(ids, fn_mask, tokenizer))

        print('='*10, 'answer', '='*10)
        print(answer)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--fn_to_learn", nargs='+', type=str, default=None)
    args = parser.parse_args()

    ds_path = "../connect_dots/functions/dev/047_functions/finetune_01_orig"
    # ds_path = "data/functions/047_functions/finetune_01_orig"
    fn_names = list(load_var_dict(ds_path).keys())

    cfg = {
        "layer": args.layer,
        "fn_to_learn": args.fn_to_learn if args.fn_to_learn else fn_names,
        "batch_size": 16,
        "num_epochs": 3,
        "max_steps": args.max_steps,
        "valid_steps": 25,
        "eval_steps": 25,
        "log_steps": 1,
        "save_steps": 100,
        "lr": 1.0,
        "weight_decay": 1e-3,
    }

    model_name = "google/gemma-2-9b-it"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    start_of_turn_tok = tokenizer.encode("<start_of_turn>", add_special_tokens=False)[0]
    assert start_of_turn_tok == 106

    train_val_ds = load_train_dataset(Path(ds_path) / "047_func_01_train_oai.jsonl")
    # train_ds = train_ds.select(range(len(train_ds) // 50))
    # filter for functions to learn
    train_val_ds = train_val_ds.filter(lambda x: any(fn in x["functions_present"] for fn in cfg["fn_to_learn"]))

    # add validation split
    train_val_dict = train_val_ds.train_test_split(test_size=0.025, shuffle=True, seed=42)
    train_ds = train_val_dict["train"]
    val_ds = train_val_dict["test"]
    del train_val_dict, train_val_ds

    tokenize_train_partial = partial(
        tokenize_train,
        tokenizer=tokenizer,
        start_of_turn_tok=start_of_turn_tok,
        fn_names=cfg["fn_to_learn"],
    )
    
    tokenized_train_ds = train_ds.map(tokenize_train_partial, num_proc=16)
    tokenized_val_ds = val_ds.map(tokenize_train_partial, num_proc=16)

    train_dataloader = DataLoader(
        tokenized_train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=partial(collate_train, max_len=128, pad_token_id=tokenizer.pad_token_id)
    )

    val_dataloader = DataLoader(
        tokenized_val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=partial(collate_train, max_len=128, pad_token_id=tokenizer.pad_token_id)
    )

    test_ds = load_test_dataset(Path(ds_path) / "047_func_01_test_oai.jsonl")
    # filter for functions to learn
    test_ds = test_ds.filter(lambda x: any(fn in x["fn_name"] for fn in cfg["fn_to_learn"]))

    tokenized_test_ds = test_ds.map(
        partial(
            tokenize_test_example,
            tokenizer=tokenizer,
            fn_names=cfg["fn_to_learn"],
        )
    )
    test_dataloader = DataLoader(
        tokenized_test_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=partial(collate_test, pad_token_id=tokenizer.pad_token_id)
    )

    sense_check_train_ds(train_dataloader, tokenizer)
    sense_check_test_ds(test_dataloader, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager',
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


    hook = TokenwiseSteeringHook(d=model.model.config.hidden_size, device=device, n_vecs=len(cfg["fn_to_learn"]))

    handle = model.model.layers[cfg["layer"]].register_forward_pre_hook(hook)

    # compile model
    # model = torch.compile(model)

    optimizer = torch.optim.Adam([
        {"params": hook.alpha_V, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]}, # fast for scale
        {"params": hook.v_VD,    "lr": cfg["lr"] * 0.1}   # slower for direction
    ])

    # optimizer = torch.optim.Adam([hook.vecs_VD], lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    num_training_steps = min(len(train_dataloader) * cfg["num_epochs"], cfg["max_steps"] or float("inf"))
    # num_warmup_steps = int(0.05 * num_training_steps)
    num_warmup_steps = 20
    print(f"num_warmup_steps: {num_warmup_steps}, num_training_steps: {num_training_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    run = wandb.init(
        project="oocr",
        name=f"steer-{cfg['layer']}",
        dir="/workspace/wandb",
        config=cfg,
        # mode="disabled",
    )

    # base_exp_path = Path(f"data/experiments/function_steering/oli_allfuncs_layer{cfg['layer']}")
    base_exp_path = Path(f"../steering_vec/functions/layer_{cfg['layer']}_1")

    step = 0
    loop_break = False  # for breaking out of all loops
    losses = []
    for epoch in range(cfg["num_epochs"]):
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            fn_occ = batch["fn_occurrences"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            hook.vec_ptrs_BS = fn_occ

            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            losses.append(loss.item())
            epoch_frac = epoch + (batch_idx + 1) / len(train_dataloader)
            print(f"step {step}, epoch {epoch_frac:.4f}, loss {loss.item():.4f} lr {optimizer.param_groups[0]['lr']:.6f}")

            if step % cfg["log_steps"] == 0:
                loss_avg = sum(losses) / len(losses)
                losses.clear()
                logging_dict = {
                    "train/epoch": epoch_frac,
                    "train/global_step": step,
                    "train/loss": loss_avg,
                    "train/lr": optimizer.param_groups[0]['lr'],
                }

                for f_idx, f_name in enumerate(cfg["fn_to_learn"]):
                    scale = hook.alpha_V[f_idx].item()
                    scale_grad = hook.alpha_V.grad[f_idx].item()

                    v_unit_grad_norm = hook.v_VD.grad[f_idx].norm().item() / hook.v_VD[f_idx].norm().item() # normalize because this has a big norm but only interested in it's non-scale component

                    run.log({
                        f"train/{f_name}_scale": abs(scale),
                        f"train/{f_name}_scale_grad": scale_grad,
                        f"train/{f_name}_direction_grad_norm": v_unit_grad_norm,
                    }, step=step)

                run.log(logging_dict, step=step)


            # save all vectors every save_steps
            if step % cfg["save_steps"] == 0:
                exp_dir = base_exp_path / f"step_{step}"
                Path(exp_dir).mkdir(parents=True, exist_ok=True)  
                for f_idx, f_name in enumerate(cfg["fn_to_learn"]):
                    dir_name = exp_dir / f"{f_name}.pt"
                    torch.save(hook.vecs_VD[f_idx], dir_name)
            
            # validation loop
            if step % cfg["valid_steps"] == 0:
                print("validating")
                model.eval()
                val_losses = []
                total_correct = 0
                total_predictable = 0

                with torch.no_grad():
                    # prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True)
                    # prof.start()

                    for i, val_batch in enumerate(val_dataloader):
                        # move tensors to device
                        input_ids = val_batch["input_ids"].to(device)
                        attention_mask = val_batch["attention_mask"].to(device)
                        labels = val_batch["labels"].to(device)
                        fn_occ = val_batch["fn_occurrences"].to(device)

                        # steer hook
                        hook.vec_ptrs_BS = fn_occ
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        hook.vec_ptrs_BS = None
                        val_losses.append(outputs.loss.item())

                        # calculate token accuracy
                        logits = outputs.logits
                        pred = torch.argmax(logits, dim=-1)
                        active_labels_mask = labels != -100
                        correct_predictions = (pred[:,1:] == labels[:,:-1]) & active_labels_mask[:,:-1]

                        total_correct += correct_predictions.sum().item()
                        total_predictable += active_labels_mask.sum().item()
                        
                avg_val_loss = sum(val_losses) / len(val_losses)
                tok_accuracy = total_correct / total_predictable if total_predictable > 0 else 0

                print(f"validation loss: {avg_val_loss:.4f}, validation accuracy: {tok_accuracy:.4f}")
                run.log({"val/loss": avg_val_loss, "val/accuracy": tok_accuracy}, step=step)

                model.train()


            # eval/test loop
            if step % cfg["eval_steps"] == 0:
                print("evaluating")
                model.eval()
                clear_cuda_mem()
                
                score, total = 0, 0
                score_dict = {}

                for test_batch in test_dataloader:
                    fn_occ = test_batch["fn_occurrences"].to(device)
                    input_ids = test_batch["input_ids"].to(device)
                    attention_mask = test_batch["attention_mask"].to(device)

                    with torch.no_grad():
                        hook.vec_ptrs_BS = fn_occ
                        outputs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=1,
                            do_sample=False,
                        )
                        hook.vec_ptrs_BS = None
                        
                    pred = [tokenizer.decode(outputs[i]) for i in range(outputs.shape[0])]
                    model_ans = [extract_answer(pred[i]) for i in range(len(pred))]
                    actual_ans = test_batch["answer"]
                    fn_name = test_batch["fn_name"]
                    total += len(model_ans)
                    result = [model_ans[i] == actual_ans[i] for i in range(len(model_ans))]
                    score += sum(result)
                    for i in range(len(result)):
                        if fn_name[i] in score_dict.keys():
                            score_dict[fn_name[i]][0] += int(result[i])
                            score_dict[fn_name[i]][1] += 1
                        else:
                            score_dict[fn_name[i]] = [int(result[i]), 1]

                results_dict = {"test/accuracy": score/total}
                for k in score_dict.keys():
                    results_dict[f"test/{k}"] = score_dict[k][0] / score_dict[k][1]
                
                print(f"test accuracy: {results_dict['test/accuracy']:.4f}")
                run.log(results_dict, step=step)

                model.train()

            # break out of all loops
            if cfg["max_steps"] is not None: 
                if step >= cfg["max_steps"]:
                    loop_break = True
                    break

        if loop_break:
            break

    handle.remove()

# %%
