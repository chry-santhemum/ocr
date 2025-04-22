# %%
# You can either patch across models or across prompts
import os
import copy
import gc
import json
from typing import cast, List, Dict, Any, Tuple, Union
from functools import partial
import yaml
import random
import re

import plotly.express as px  # type: ignore
import torch
from peft import LoraModel, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2DecoderLayer,
    Gemma2ForCausalLM,
)
from transformer_lens import patching, HookedTransformer
from transformer_lens.hook_points import HookPoint

from utils import clear_cuda_mem, load_test_dataset, find_token_pos

# %%

model_name = "google/gemma-2-9b-it"
finetune_checkpoint_dir = "/workspace/checkpoints/9b-func-all-r4/checkpoint-2000/"
ds_path = "connect_dots/functions/dev/047_functions/finetune_01/"
device = torch.device("cuda")

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation='eager',
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Only need to run this line if you do cross-model patching
base_model_clone = copy.deepcopy(base_model).to('cuda')

merged_model: Gemma2ForCausalLM = cast(
    LoraModel, PeftModel.from_pretrained(base_model, finetune_checkpoint_dir).to('cuda')
).merge_and_unload(progressbar=True)  # type: ignore

del base_model
clear_cuda_mem()

# %%
# Only need to run this block if you do cross-model patching

base_tl_model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    hf_model=base_model_clone.to(device),  # type: ignore
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device=device,
)

del base_model_clone
clear_cuda_mem()

# %%

tuned_tl_model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    hf_model=merged_model.to(device),  # type: ignore
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device="cuda",
)
del merged_model
clear_cuda_mem()


# %%

test_ds = load_test_dataset(os.path.join(ds_path, "047_func_01_test_oai.jsonl"))
test_prompts = test_ds["messages"]
correct_ans = test_ds["answer"]

# %%
######################
# clean/dirty patching

config_dir = os.path.join(ds_path, "test_config.yaml")
with open(config_dir, "r") as f:
    data_dict = yaml.safe_load(f)
fn_names = list(data_dict['dataset']['var_dict'].keys())

def create_spoiled_list(function_names, strings):
    # Create a spoiled version of the list
    spoiled_strings = []
    
    for string in strings:
        # Find which function is mentioned in this string
        found = None
        for func in function_names:
            if func in string:
                found = func
                break
                
        if found:
            # Get a random different function name
            replacement = "odgrps"
            # Replace the function name
            spoiled_string = re.sub(r'\b' + re.escape(found) + r'\b', replacement, string)
            spoiled_strings.append(spoiled_string)
        else:
            # If no function was found, append the original
            print("None matched")
            spoiled_strings.append(string)
    
    return spoiled_strings

clean_prompts = tokenizer.apply_chat_template(test_prompts, tokenize=False, add_generation_prompt=True)
dirty_prompts = create_spoiled_list(fn_names, clean_prompts)


def metric(logits, ans): 
    # ans: [batch_size]
    logits = logits[:,-1,:].squeeze(1)
    clear_cuda_mem()

    letters = ['A', 'B', 'C', 'D', 'E']
    letters_id = tuned_tl_model.to_tokens(letters, prepend_bos=False).squeeze()

    avg_letter_logits = logits[:,letters_id].mean(dim=1)

    result = logits[torch.arange(logits.shape[0]), ans]
    result = result - avg_letter_logits
    return result.mean()

# %%

clear_cuda_mem(verbose=True)

prompt_index = 0
# print(clean_prompts[prompt_index])
# print(dirty_prompts[prompt_index])

letters = ['A', 'B', 'C', 'D', 'E']
letters_id = tuned_tl_model.to_tokens(letters, prepend_bos=False).squeeze()

with torch.no_grad():
    logits = tuned_tl_model(
        clean_prompts[prompt_index],
        return_type="logits",
    )[:,-1,:].squeeze(1)
    print(logits[:,letters_id])
    logits = tuned_tl_model(
        dirty_prompts[prompt_index],
        return_type="logits",
    )[:,-1,:].squeeze(1)
    print(logits[:,letters_id])
    print(correct_ans[prompt_index])

# %%

def clean_dirty_patching():
    clean_toks = tuned_tl_model.to_tokens(clean_prompts[prompt_index:prompt_index+1], padding_side="left")
    dirty_toks = tuned_tl_model.to_tokens(dirty_prompts[prompt_index:prompt_index+1], padding_side="left")
    assert clean_toks.shape == dirty_toks.shape
    ans = tuned_tl_model.to_tokens(correct_ans[prompt_index:prompt_index+1], prepend_bos=False).squeeze(dim=1)

    patching_metric = partial(metric, ans=ans)
    with torch.no_grad():
        _, clean_cache = tuned_tl_model.run_with_cache(clean_toks)

    results = patching.get_act_patch_mlp_out(
        tuned_tl_model,
        dirty_toks,
        clean_cache,
        patching_metric
    )
    return results

results = clean_dirty_patching()

# %%

labels = [f"{tok} {i}" for i, tok in enumerate(tuned_tl_model.to_str_tokens(clean_prompts[prompt_index]))]

px.imshow(
    results.cpu(),
    labels={"x": "Position", "y": "Layer"},
    x=labels,
    title="post-MLP Activation Patching",
    width=2000,
    height=1000,
)

# %%
######################
# Cross-model patching

def replace_hook(
    acts: torch.Tensor,
    hook: HookPoint,
    replace: torch.Tensor,
    token_pos: Union[int, List[int]]
) -> torch.Tensor:
    """
    token_pos:   either
        - single int: same position in every batch
        - list of ints of length batch: one pos per example
        - list of ints of length <= seq_len: same set for every batch
        - list of lists of ints: one set per example
    """
    assert acts.shape == replace.shape
    batch, seq_len, dim = acts.shape
    new_acts = acts.clone()
    
    # single position for all examples
    if isinstance(token_pos, int):
        new_acts[:, token_pos, :] = replace[:, token_pos, :]
        return new_acts

    # list of per-example positions
    if len(token_pos) == batch:
        for b, pos in enumerate(token_pos):
            new_acts[b, pos, :] = replace[b, pos, :]
    else:
        # treat as global positions for every example
        # e.g. token_pos = [2,5,7]
        print("Treating as global positions for every batch")
        new_acts[:, token_pos, :] = replace[:, token_pos, :]

    return new_acts


config_dir = os.path.join(ds_path, "test_config.yaml")
with open(config_dir, "r") as f:
    data_dict = yaml.safe_load(f)

var_dict = data_dict['dataset']['var_dict']


def two_models_patching(from_model, to_model, layers:List[int], batch_size:int):
    patched_ans = []

    for i in range(0, 200, batch_size):
        to_model.reset_hooks()
        clear_cuda_mem()

        seq = tokenizer.apply_chat_template(
            test_prompts[i:i+batch_size], 
            tokenize=False, 
            add_generation_prompt=True
        )

        input_toks = to_model.to_tokens(seq, padding_side="left")

        token_pos = []

        for prompt in seq:
            for name in var_dict.keys():
                if name in prompt:
                    fn_name = name
            # token_pos.append(find_token_pos(tokenizer, fn_name, prompt, last_tok_only=False))
            token_pos.append([i for i in range(input_toks.shape[1]-5, input_toks.shape[1])])
        
        print(token_pos)

        # patched logits
        with torch.no_grad():
            _, cache = from_model.run_with_cache(input_toks)
            fwd_hooks = []
            for L in layers:
                from_activ = cache[f"blocks.{L}.hook_mlp_out"]
                fwd_hooks.append((f'blocks.{L}.hook_mlp_out', partial(replace_hook, replace=from_activ, token_pos=token_pos)))
            del cache
            logits_patch = to_model.run_with_hooks(
                input_toks,
                return_type="logits",
                fwd_hooks=fwd_hooks,
            )[:,-1,:].squeeze()

            original_ans = from_model.generate(
                input_toks,
                max_new_tokens=1,
                do_sample=False,
            )
            original_ans = original_ans[:, -1].squeeze()

        logprobs_patch = logits_patch.softmax(dim=-1)
        del logits_patch
        _, labels_patch = torch.max(logprobs_patch, dim=-1)
        del logprobs_patch

        patched_answers = to_model.to_string(labels_patch)
        original_answers = to_model.to_string(original_ans)

        print("patched:  ", patched_answers)
        print("original: ", original_answers)
        print("correct:  ", "".join(correct_ans[i:i+batch_size]))
        patched_ans.extend(labels_patch.cpu().numpy().tolist())

        # for j in range(batch_size):
        #     if patched_answers[j] == original_answers[j]:
        #         print(seq[j])

        if i > 20:
            break

    return patched_ans


patched_answers = two_models_patching(tuned_tl_model, base_tl_model, layers=[i for i in range(42)],  batch_size=16)

# %%

