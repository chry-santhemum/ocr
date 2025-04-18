# %%
# You can either patch across models or across prompts
import os
import copy
import gc
import json
from typing import cast
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

from utils import clear_cuda_mem, load_test_dataset

# %%

model_name = "google/gemma-2-9b-it"
finetune_checkpoint_dir = "/workspace/checkpoints/9b-func-all-r32/checkpoint-1000/"
ds_path = "functions/dev/047_functions/finetune_01/"
device = torch.device("cuda")

# %%

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation='eager',
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model_clone = copy.deepcopy(base_model).to('cuda')
merged_model: Gemma2ForCausalLM = cast(
    LoraModel, PeftModel.from_pretrained(base_model, finetune_checkpoint_dir).to('cuda')
).merge_and_unload(progressbar=True)  # type: ignore

del base_model
clear_cuda_mem()

# %%

base_tl_model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    hf_model=base_model_clone.to(device),  # type: ignore
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device=device,
)

del base_model_clone
clear_cuda_mem()

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
# clean/dirty prompts

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
            replacement_options = [f for f in function_names if f != found]
            if replacement_options:
                replacement = random.choice(replacement_options)
                # Replace the function name
                spoiled_string = re.sub(r'\b' + re.escape(found) + r'\b', replacement, string)
                spoiled_strings.append(spoiled_string)
            else:
                # If there's only one function name, just append the original
                spoiled_strings.append(string)
        else:
            # If no function was found, append the original
            print("None matched")
            spoiled_strings.append(string)
    
    return spoiled_strings

clean_prompts = tokenizer.apply_chat_template(test_prompts, tokenize=False, add_generation_prompt=True)
dirty_prompts = create_spoiled_list(fn_names, clean_prompts)

# %%

clear_cuda_mem(verbose=True)

print(clean_prompts[0])
print(dirty_prompts[0])

# %%

def metric(logits, ans): 
    # ans: [batch_size]
    logits = logits[:,-1,:].squeeze(1)
    clear_cuda_mem()

    letters = ['A', 'B', 'C', 'D', 'E']
    letters_id = tuned_tl_model.to_tokens(letters, prepend_bos=False).squeeze()

    avg_letter_logits = logits[:,letters_id].mean(dim=1)

    result = logits[torch.arange(logits.shape[0]), ans]
    result = result - avg_letter_logits
    result = result.mean()

    return result

def clean_dirty_patching():
    clean_toks = tuned_tl_model.to_tokens(clean_prompts[:1], padding_side="left")
    dirty_toks = tuned_tl_model.to_tokens(dirty_prompts[:1], padding_side="left")
    ans = tuned_tl_model.to_tokens(correct_ans[:1], prepend_bos=False).squeeze(dim=1)

    patching_metric = partial(metric, ans=ans)
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

labels = [f"{tok} {i}" for i, tok in enumerate(tuned_tl_model.to_str_tokens(clean_prompts[0]))]

px.imshow(
    results.cpu(),
    labels={"x": "Position", "y": "Layer"},
    x=labels,
    title="post-MLP Activation Patching",
    width=2000,
    height=1000,
)

# %%

def replace_hook(acts, hook: HookPoint, replace):
    assert acts.shape == replace.shape #[batch, seq_len, d_model]
    acts.copy_(replace)
    return replace

def two_models_patching(from_model, to_model, layers, batch_size):
    patched_ans = []

    for i in range(0, 200, batch_size):
        to_model.reset_hooks()
        clear_cuda_mem()

        seq = tokenizer.apply_chat_template(test_prompts[i:i+batch_size], tokenize=False, add_generation_prompt=True)
        input_toks = to_model.to_tokens(seq, padding_side="left")

        # patched logits
        with torch.no_grad():
            _, cache = from_model.run_with_cache(input_toks)
            fwd_hooks = []
            for L in layers:
                from_activ = cache[f"blocks.{L}.hook_mlp_out"]
                fwd_hooks.append((f'blocks.{L}.hook_mlp_out', partial(replace_hook, replace=from_activ)))
            del cache
            logits_patch = to_model.run_with_hooks(
                input_toks,
                return_type="logits",
                fwd_hooks=fwd_hooks,
            )[:,-1,:].squeeze()

        logprobs_patch = logits_patch.softmax(dim=-1)
        del logits_patch
        _, labels_patch = torch.max(logprobs_patch, dim=-1)
        del logprobs_patch

        print(to_model.to_string(labels_patch))
        patched_ans.extend(labels_patch.cpu().numpy().tolist())
        break

    return patched_ans

# %%

patched_answers = two_models_patching(tuned_tl_model, base_tl_model, layers=[i for i in range(10)], batch_size=16)

actual_ans = "".join(correct_ans[:16])

# %%

