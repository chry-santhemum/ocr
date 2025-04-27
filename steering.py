# Script to train conditional steering vectors (for the functions task for now)

# To do tokenwise steering: For each function name, find all occurrences in the
# conversation, then create a mask of shape (batch, seq_len) where the value is
# n if that token if part of the name of function n, else -1

# %%
from functools import partial
import itertools
from pathlib import Path
import re
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, PreTrainedTokenizer
import wandb

from eval_fns import extract_answer
from utils import clear_cuda_mem, find_token_pos, load_test_dataset, load_train_dataset, load_var_dict

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

    attn_mask = test_ids != tokenizer.pad_token_id

    fn_occurrences = torch.zeros((len(batch), prompt_len), dtype=torch.long) - 1

    for i in range(len(texts)):
        # find the function names and their token positions
        prompt = texts[i][0]['content']

        for fn_name in var_dict.keys():
            if fn_name in prompt:
                token_pos = find_token_pos(
                    tokenizer,
                    fn_name,
                    tokenizer.apply_chat_template(texts[i], tokenize=False, add_generation_prompt=True),
                    last_tok_only=False,
                )
                for pos in token_pos:
                    unpadded_prompt_len = tokenizer.apply_chat_template(texts[i], return_tensors="pt", tokenize=True, add_generation_prompt=True).shape[1]
                    # need to shift because of padding
                    pos = pos + prompt_len - unpadded_prompt_len - 1
                    fn_occurrences[i, pos] = FN_NAMES.index(fn_name)

    return {
        "input_ids": test_ids.to("cuda"), # type: ignore
        "answer": [ex["answer"] for ex in batch],
        "fn_name": [ex["fn_name"] for ex in batch],
        "fn_occurrences": fn_occurrences,
        "attn_mask": attn_mask.to("cuda"), # type: ignore
    }

# test_ds = test_ds.filter(lambda x: function_to_learn in x["fn_name"])
# print("Filtered test datapoints", len(test_ds))

def tokenize_with_completion_mask(
    conversation: dict[str, list[dict[str, str]]],
    tokenizer: PreTrainedTokenizer,
    start_of_turn_tok: int,
) -> dict[str, list[int]]:
    messages = [
        {"role": "user", "content": conversation["prompt"]},
        {"role": "assistant", "content": conversation["completion"]},
    ]

    conversation_str: str = tokenizer.apply_chat_template(  # type: ignore
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )

    encoding = tokenizer(conversation_str, return_offsets_mapping=True, add_special_tokens=False)

    tokens = encoding['input_ids']

    # get the SECOND occurrence of the start_of_turn_tok
    split_idx = tokens.index(start_of_turn_tok, 10) + 3 # skip <start_of_turn>model\n
    prompt_tokens: list[int] = tokens[:split_idx]
    completion_tokens: list[int] = tokens[split_idx:]

    labels = [-100] * len(prompt_tokens) + completion_tokens
    # last two tokens are the <eot> and return, so don't include them
    labels[-2] = -100
    labels[-1] = -100

    # start with all -1s, signifying that the token is not part of any function name
    fn_occurrences = [-1] * len(encoding['input_ids'])
    for fn_name in conversation['functions_present']:
        positions = find_token_pos(tokenizer, fn_name, conversation_str, last_tok_only=False)
        for pos in positions:
            fn_occurrences[pos] = FN_NAMES.index(fn_name)

    assert len(tokens) == len(labels), f"len(tokens) = {len(tokens)}, len(labels) = {len(labels)}"

    return {
        "input_ids": tokens,
        "labels": labels,
        "fn_occurrences": fn_occurrences,
    }

def collate(batch, max_len: int, pad_token_id: int):
    max_len_present = max(len(example["input_ids"]) for example in batch)
    out_len = min(max_len_present, max_len)

    input_ids_list = []
    labels_list = []
    fn_occurrences_list = []

    for example in batch:
        input_ids = example["input_ids"]
        labels = example["labels"]
        fn_occurrences = example["fn_occurrences"]

        assert len(input_ids) == len(labels) == len(fn_occurrences)
        padding_length = out_len - len(input_ids)

        input_ids_list.append(input_ids + [pad_token_id] * padding_length)
        labels_list.append(labels + [-100] * padding_length)
        fn_occurrences_list.append(fn_occurrences + [-1] * padding_length)

    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    fn_occurrences_tensor = torch.tensor(fn_occurrences_list, dtype=torch.long)

    return {
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
        "fn_occurrences": fn_occurrences_tensor,
    }

# %%

logged = 0
class SteeringHook:
    def __init__(self, d: int, device: torch.device):
        self.d = d

        self.steering_vecs_FD = nn.Parameter(torch.zeros((len(var_dict), self.d), device=device, dtype=torch.float32))
        self.dummy_vec = torch.zeros((1, self.d), device=device, dtype=torch.float32, requires_grad=False)
        self.fn_occurrences_BS: torch.Tensor | None = None

    def __call__(self, module, input):
        global logged
        hidden_BSD, = input
        B, S, D = hidden_BSD.shape
        assert D == self.d
        assert self.fn_occurrences_BS is not None

        steering_vecs_FD = torch.cat([self.steering_vecs_FD, self.dummy_vec], dim=0)
        assert steering_vecs_FD.shape == (len(var_dict) + 1, self.d)
        vecs_add_BSD = steering_vecs_FD[self.fn_occurrences_BS]

        assert vecs_add_BSD.shape == (B, S, D)

        hidden_BSD += vecs_add_BSD
        return (hidden_BSD,)


# Just an idea:
# and initial peak, then a cooldown, then a linear decay to 0
#
#      /\
#     /  \
#    /    \
#   /      ^^**..__
#  /               ^^**..__
# /                        ^^**..__

def get_initial_peak_lr_scheduler(optimizer, peak_multiplier: int, num_warmup_steps: int, cooldown_steps: int, total_num_training_steps: int):
    """an LR scheduler that initially warms up to `peak_multiplier * base_lr` after `num_warmup_steps`, then decays linearly to `base_lr` after `cooldown_steps`, then linearly to 0 after `num_training_steps`"""
    def lr_lambda(step):
        if step < num_warmup_steps:
            # linear from 0 to peak_multiplier
            pct_through_warmup = step / num_warmup_steps
            return pct_through_warmup * peak_multiplier
        elif step < num_warmup_steps + cooldown_steps:
            # linear from peak_multiplier to 1
            pct_thought_cooldown = (step - num_warmup_steps) / cooldown_steps
            return peak_multiplier - (pct_thought_cooldown * (peak_multiplier - 1))
        else:
            # linear from 1 to 0
            initial_peak_steps = num_warmup_steps + cooldown_steps
            pct_through_total = (step - initial_peak_steps) / (total_num_training_steps - initial_peak_steps)
            return 1 - pct_through_total

    return LambdaLR(optimizer, lr_lambda)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int)

    cfg = {
        "layer": 9,
        "num_epochs": 1,
        "eval_steps": 20,
        "log_steps": 2,
        "save_steps": 50,
        "learning_rate": 1e-1,
        "weight_decay": 0,
    }

    # %%
    # function_to_learn = "ttsund"

    ds_path = "../connect_dots/functions/dev/047_functions/finetune_01"
    # ds_path = "./data/functions/047_functions/finetune_01"
    var_dict = load_var_dict(ds_path)

    FN_NAMES = list(var_dict.keys())
    # %%

    model_name = "google/gemma-2-9b-it"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation='eager',
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # %%

    start_of_turn_tok = tokenizer.encode("<start_of_turn>", add_special_tokens=False)[0]
    assert start_of_turn_tok == 106

    train_ds = load_train_dataset(Path(ds_path) / "047_func_01_train_oai.jsonl")
    train_ds = train_ds.select(range(len(train_ds) // 50))
    tokenized_train_ds = train_ds.map(partial(tokenize_with_completion_mask, tokenizer=tokenizer, start_of_turn_tok=start_of_turn_tok))

    # %%

    train_dataloader = DataLoader(
        tokenized_train_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=lambda x: collate(x, max_len=128, pad_token_id=tokenizer.pad_token_id)
    )

    test_ds = load_test_dataset(Path(ds_path) / "047_func_01_test_oai.jsonl")

    test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=partial(test_collate_fn, tokenizer=tokenizer))

    import itertools
    def green(s: str) -> str:
        s = s.replace(" ", "·").replace("\n", "\n↵")
        return f"\033[92m{s}\033[0m"

    def decode_highlighted(toks: list[int], highlight_mask: list[int]) -> str:
        str_toks = [tokenizer.decode(tok) for tok in toks]
        return ''.join([green(tok) if mask else tok for tok, mask in zip(str_toks, highlight_mask)])

    def visualise_train_ds():
        num_examples = 10
        for ex in itertools.islice(train_dataloader, num_examples):
            ids = ex["input_ids"][0].tolist()
            fn_mask = (ex["fn_occurrences"][0] != -1).tolist()
            completion_mask = (ex["labels"][0] != -100).tolist()

            print('='*10, 'function tokens', '='*10)
            print(decode_highlighted(ids, fn_mask))

            print('='*10, 'completion tokens', '='*10)
            print(decode_highlighted(ids, completion_mask))

    # uncomment to visualise the dataset
    visualise_train_ds()


    def visualise_test_ds():
        num_examples = 10
        for ex in itertools.islice(test_dataloader, num_examples):
            ids = ex["input_ids"][0].tolist()
            fn_mask = (ex["fn_occurrences"][0] != -1).tolist()
            answer = ex["answer"][0]

            print('='*10, 'function tokens', '='*10)
            print(decode_highlighted(ids, fn_mask))

            print('='*10, 'answer', '='*10)
            print(answer)

    visualise_test_ds()
    hook = SteeringHook(d=model.model.config.hidden_size, device=device)
    handle = model.model.layers[cfg["layer"]].register_forward_pre_hook(hook)
    optimizer = torch.optim.AdamW([hook.steering_vecs_FD], lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    num_training_steps = len(train_dataloader) * cfg["num_epochs"]

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,  # int(0.05 * num_training_steps)
        num_training_steps=num_training_steps,
    )

    run = wandb.init(
        project="oocr",
        name="yolo-steer",
        dir="/workspace/wandb",
        config=cfg,
        # mode="disabled",
    )

    base_exp_path = Path(f"data/experiments/function_steering/oli_allfuncs_layer{cfg['layer']}")

    step = 0
    losses = []
    for epoch in range(3):
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            hook.fn_occurrences_BS = batch["fn_occurrences"]

            outputs = model(
                input_ids=batch["input_ids"].to(device),
                labels=batch["labels"].to(device),
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

            epoch_frac = epoch + batch_idx / len(train_dataloader)
            print(f"step {step}, epoch {epoch_frac}, loss {loss.item()}")

            if step % cfg["log_steps"] == 0:
                loss_avg = sum(losses) / len(losses)
                losses.clear()
                logging_dict = {
                    "train/epoch": epoch_frac,
                    "train/loss": loss_avg,
                    "train/lr": optimizer.param_groups[0]['lr'],
                }

                for f_idx, f_name in enumerate(FN_NAMES[::5]):
                    norm = hook.steering_vecs_FD[f_idx].norm()
                    grad_norm = hook.steering_vecs_FD.grad[f_idx].norm()
                    logging_dict[f"train/{f_name}_vector_norm"] = norm.item()
                    logging_dict[f"train/{f_name}_grad_norm"] = grad_norm.item()

                run.log(logging_dict, step=step)


            # save all vectors every save_steps
            if step % cfg["save_steps"] == 0:
                exp_dir = base_exp_path / f"step_{step}"
                Path(exp_dir).mkdir(parents=True, exist_ok=True)  
                for f_idx, f_name in enumerate(FN_NAMES):
                    dir_name = exp_dir / f"{f_name}.pt"
                    torch.save(hook.steering_vecs_FD[f_idx], dir_name)

            step += 1

            if step % cfg["eval_steps"] == 0:
                model.eval()
                clear_cuda_mem()
                
                score, total = 0, 0
                score_dict = {}

                for test_batch in test_dataloader:
                    with torch.no_grad():
                        hook.fn_occurrences_BS = test_batch["fn_occurrences"]
                        outputs = model.generate(
                            input_ids=test_batch["input_ids"],
                            attention_mask=test_batch["attn_mask"],
                            max_new_tokens=1,
                            do_sample=False,
                        )
                    pred = [tokenizer.decode(outputs[i]) for i in range(outputs.shape[0])]
                    model_ans = [extract_answer(pred[i]) for i in range(len(pred))]
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
                wandb.log(results_dict)

                model.train()

    for i, handle in model.model.layers[cfg["layer"]]._forward_pre_hooks.items():
        print(f"removing hook {i}")
        handle.remove()

# %%

