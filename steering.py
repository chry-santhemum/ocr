# %%
from functools import partial
# Script to train conditional steering vectors (for the functions task for now)
import itertools
from pathlib import Path
import re
from sympy import sec
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, PreTrainedTokenizer
import wandb

from utils import load_train_dataset, load_var_dict

# %%

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

function_to_learn = "ttsund"

# ds_path = "../connect_dots/functions/dev/047_functions/finetune_01"
ds_path = "./data/functions/047_functions/finetune_01"
var_dict = load_var_dict(ds_path)

FN_NAMES = list(var_dict.keys())
# %%
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation='eager',)
model.eval()
for p in model.parameters():
    p.requires_grad = False

# %%

start_of_turn_tok = tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)[0]
assert start_of_turn_tok == 106

def tokenize_with_completion_mask(
    conversation: dict[str, list[dict[str, str]]],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, list[int]]:
    """
    Returns:
        input_ids: list[int]
        completion_mask: list[int]
    """
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

    fn_occurrences = [-1] * len(encoding['input_ids'])
    for fn_name in conversation['functions_present']:
        if fn_name not in FN_NAMES:
            continue

        fn_index = FN_NAMES.index(fn_name)  # 0-18 range directly

        # Find all occurrences of the function name using regex word boundaries
        for match in re.finditer(r'\b' + re.escape(fn_name) + r'\b', conversation_str):
            start_char, end_char = match.span()

            # Find which tokens correspond to this character range
            token_indices = []
            for i, (token_start, token_end) in enumerate(encoding['offset_mapping']):
                # Skip special tokens
                if token_start == token_end == 0:
                    continue

                # Check if this token overlaps with the function name
                if token_end > start_char and token_start < end_char:
                    token_indices.append(i)

            # Mark these token positions with the function index
            for idx in token_indices:
                fn_occurrences[idx] = fn_index


    assert len(tokens) == len(labels) == len(fn_occurrences), (
        f"len(tokens) = {len(tokens)}, len(labels) = {len(labels)}, len(fn_occurrences) = {len(fn_occurrences)}"
    )
    return {
        "input_ids": tokens,
        "labels": labels,
        "fn_occurrences": fn_occurrences,
    }

def simple_collate_fn(batch, max_len: int):
    """
    Simple collate function that just handles padding and conversion to tensors.
    
    Args:
        batch: List of dictionaries with 'input_ids' and 'fn_occurrences' keys
        
    Returns:
        Dictionary with batched and padded tensors
    """
    max_len_present = max(len(example["input_ids"]) for example in batch)
    out_len = min(max_len_present, max_len)

    input_ids_list = []
    fn_occurrences_list = []
    labels_list = []

    for example in batch:
        input_ids = example["input_ids"]
        labels = example["labels"]
        fn_occurrences = example["fn_occurrences"]
        assert len(input_ids) == len(fn_occurrences)
        padding_length = out_len - len(input_ids)
        input_ids_list.append(input_ids + [tokenizer.pad_token_id] * padding_length)
        labels_list.append(labels + [-100] * padding_length)
        fn_occurrences_list.append(fn_occurrences + [-1] * padding_length)

    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
    fn_occurrences_tensor = torch.tensor(fn_occurrences_list, dtype=torch.long)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    return {
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
        "fn_occurrences": fn_occurrences_tensor,  # Note the spelling matches your requirement
    }

train_ds = load_train_dataset(Path(ds_path) / "047_func_01_train_oai.jsonl")
# train_ds = train_ds.filter(lambda x: function_to_learn in x["functions_present"])
tokenized_train_ds = train_ds.map(partial(tokenize_with_completion_mask, tokenizer=tokenizer))
# %%
def collate(batch):
    return simple_collate_fn(batch, max_len=1024)

train_dataloader = DataLoader(tokenized_train_ds, batch_size=64, shuffle=True, collate_fn=collate)


class SteeringHook:
    def __init__(self, steering_vecs_FD):
        D = steering_vecs_FD.shape[1]
        self.steering_vecs_FD = torch.cat([steering_vecs_FD, torch.zeros((1, D), device=device, dtype=torch.float32)])
        # so that -1 is the zero vector
        assert self.steering_vecs_FD.shape == (len(var_dict) + 1, D)
        self.fn_occurrences_BS: torch.Tensor | None = None

    def __call__(self, module, input):
        input_t, = input
        assert input_t.ndim == 3
        assert input_t.shape[2] == model.config.hidden_size
        vecs_add_BSD = self.steering_vecs_FD[self.fn_occurrences_BS]
        input_t += vecs_add_BSD
        return (input_t,)

# %%

handles = []
# %%
LAYER = 6
steering_vecs_FD = nn.Parameter(torch.zeros((len(var_dict), model.config.hidden_size), device=device, dtype=torch.float32))

hook = SteeringHook(steering_vecs_FD)

handle = model.model.layers[LAYER].register_forward_pre_hook(hook)
handles.append(handle)


# %%

optimizer = torch.optim.AdamW([steering_vecs_FD], lr=1e-1, weight_decay=1e-6)

step = 0
num_epochs = 3
eval_steps = 50
log_steps = 5
save_steps = 50

num_training_steps = len(train_dataloader) * num_epochs
print("num training steps", num_training_steps)
num_warmup_steps = int(0.05 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

run = wandb.init(
    project="oocr",
    name="yolo-steer",
    dir="/workspace/wandb",
)
# %%

for epoch in range(3):
    for batch_idx, batch in enumerate(train_dataloader):
        step += 1
        optimizer.zero_grad()

        hook.fn_occurrences_BS = batch["fn_occurrences"].to(device)

        outputs = model(
            input_ids=batch["input_ids"].to(device),
            labels=batch["labels"].to(device),
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"step {step}: loss {loss.item()}")

        if step % log_steps == 0:
            logging_dict = {
                "train/epoch": epoch + batch_idx / len(train_dataloader),
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]['lr'],
            }

            for f_idx, f_name in enumerate(FN_NAMES):
                norm = steering_vecs_FD[f_idx].norm()
                grad_norm = steering_vecs_FD.grad[f_idx].norm()
                logging_dict[f"train/{f_name}_vector_norm"] = norm.item()
                logging_dict[f"train/{f_name}_grad_norm"] = grad_norm.item()

            run.log(logging_dict, step=step)


        # # save all vectors every save_steps
        # if step % save_steps == 0:
        #     os.makedirs(f"steering_vectors/layer-{LAYER}", exist_ok=True)  
        #     for k, v in steer_dict.items():
        #         # save the tensor
        #         if k == function_to_learn:
        #             dir_name = f"steering_vectors/layer_{LAYER}/{k}_{step}.pt"
        #             torch.save(v, dir_name)


        # if step % eval_steps == 0:
        #     results_dict = eval(model, tokenizer, test_dataloader)
        #     wandb.log(results_dict)

# remove hook

# %%
for i, handle in model.model.layers[LAYER]._forward_pre_hooks.items():
    print(f"removing hook {i}")
    handle.remove()

# %%


# def eval(model, tokenizer, test_dataloader):
#     clear_cuda_mem()
    
#     score, total = 0, 0
#     score_dict = {}

#     for test_batch in test_dataloader:
#         with torch.no_grad():
#             hook.batch_pos = test_batch["steer_pos"]
#             print("="*10, "\n")
#             print(hook.batch_pos)
#             print(test_batch["input_ids"][0])
#             outputs = model.generate(
#                 input_ids=test_batch["input_ids"],
#                 max_new_tokens=1,
#                 do_sample=False,
#             )
#         print("Successfully outputted")
#         print(tokenizer.decode(outputs[0]))
#         pred = [tokenizer.decode(outputs[j]) for j in range(outputs.shape[0])]

#         model_ans = [extract_answer(pred[j]) for j in range(len(pred))]
#         actual_ans = test_batch["answer"]
#         fn_names = test_batch["fn_name"]

#         total += len(model_ans)
#         result = [model_ans[i] == actual_ans[i] for i in range(len(model_ans))]

#         score += sum(result)
#         for i in range(len(result)):
#             if fn_names[i] in score_dict.keys():
#                 score_dict[fn_names[i]][0] += int(result[i])
#                 score_dict[fn_names[i]][1] += 1
#             else:
#                 score_dict[fn_names[i]] = [int(result[i]), 1]

#     results_dict = {"test/accuracy": score/total}
#     for k in score_dict.keys():
#         results_dict[f"test/{k}"] = score_dict[k][0] / score_dict[k][1]

#     model.train()
#     return results_dict


# # load test dataset

# test_ds = load_test_dataset(os.path.join(ds_path, "047_func_01_test_oai.jsonl"))

# def test_collate_fn(batch, tokenizer):
#     # batch is a list of dicts, each with "messages"
#     texts = [ex["messages"] for ex in batch]
#     test_ids = tokenizer.apply_chat_template(
#         texts,
#         return_tensors="pt",
#         tokenize=True,
#         padding=True,
#         add_generation_prompt=True,
#     )

#     prompt_len = test_ids.shape[1]

#     steer_pos = {fn_name: [] for fn_name in var_dict.keys()}
#     for i in range(len(texts)):
#         # find the function names and their token positions
#         prompt = texts[i][0]['content']

#         for fn_name in var_dict.keys():
#             if fn_name in prompt:
#                 token_pos = find_token_pos(tokenizer, fn_name, tokenizer.apply_chat_template(texts[i], tokenize=False, add_generation_prompt=True))
#                 for pos in token_pos:
#                     unpadded_prompt_len = tokenizer.apply_chat_template(texts[i], return_tensors="pt", tokenize=True, add_generation_prompt=True).shape[1]
#                     # need to shift because of padding
#                     pos = pos + prompt_len - unpadded_prompt_len - 1
#                     steer_pos[fn_name].append((i, pos))

#     return {
#         "input_ids": test_ids.to("cuda"), # type: ignore
#         "answer": [ex["answer"] for ex in batch],
#         "fn_name": [ex["fn_name"] for ex in batch],
#         "steer_pos": steer_pos,
#     }

# test_ds = test_ds.filter(lambda x: function_to_learn in x["fn_name"])
# print("Filtered test datapoints", len(test_ds))

# test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=partial(test_collate_fn, tokenizer=tokenizer))

# for d in test_dataloader:
#     print(d["input_ids"][0])
#     print([tokenizer.decode(d["input_ids"][0][i]) for i in range(d["input_ids"].shape[1])])
#     print(d["steer_pos"])

# %%

# def green(s: str) -> str:
#     s = s.replace(" ", "·").replace("\n", "\n↵")
#     return f"\033[92m{s}\033[0m"

# def decode_highlighted(toks: list[int], highlight_mask: list[int]) -> str:
#     str_toks = [tokenizer.decode(tok) for tok in toks]
#     return ''.join([green(tok) if mask else tok for tok, mask in zip(str_toks, highlight_mask)])

# for ex in itertools.islice(train_dataloader, 10):
#     ids = ex["input_ids"][0].tolist()
#     fn_mask = (ex["fn_occurrences"][0] != -1).tolist()
#     completion_mask = (ex["labels"][0] != -100).tolist()

#     print('<function tokens>')
#     print(decode_highlighted(ids, fn_mask))
#     print('</function tokens>')

#     print('<completion tokens>')
#     print(decode_highlighted(ids, completion_mask))
#     print('</completion tokens>')
# # %%
