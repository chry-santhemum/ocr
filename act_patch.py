# %%
import os
os.environ["HF_HOME"] = "/workspace/.cache/huggingface/"

import copy
import gc
import json
from typing import cast

import plotly.express as px  # type: ignore
import torch
from peft import LoraModel, PeftModel
from transformer_lens import HookedTransformer  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2DecoderLayer,
    Gemma2ForCausalLM,
)

from utils import get_seq_data, normalised_distance, run_acts_through_other_model, clear_cuda_mem

clear_cuda_mem()

# %%

base_model_name = "google/gemma-2-9b-it"
dtype = torch.bfloat16
cache_dir = "cache"
finetune_checkpoint_dir = "/workspace/checkpoints/9b-functions-mlp-lora/checkpoint-3000"
output_dir = "output"
device = torch.device("cuda")

# %%

print(f"Loading base model: {base_model_name}")

base_model: Gemma2ForCausalLM = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    cache_dir=cache_dir,
    torch_dtype=dtype,
    device_map=device,
    attn_implementation="eager",
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)

# %%
base_model_clone = copy.deepcopy(base_model).to('cuda')
# %%

merged_model: Gemma2ForCausalLM = cast(
    LoraModel, PeftModel.from_pretrained(base_model, finetune_checkpoint_dir).to('cuda')
).merge_and_unload(progressbar=True)  # type: ignore
if not isinstance(merged_model, Gemma2ForCausalLM):
    raise ValueError(
        "Merged model is not a Gemma2ForCausalLM" + str(type(merged_model))
    )

gc.collect()
torch.cuda.empty_cache()

# %%
weight_base = cast(
    Gemma2DecoderLayer, base_model.model.layers[0]
).mlp.up_proj.weight.clone()
weight_tuned = cast(
    Gemma2DecoderLayer, merged_model.model.layers[0]
).mlp.up_proj.weight.clone()
assert not torch.allclose(weight_base, weight_tuned)

del weight_base
del weight_tuned
clear_cuda_mem()
# %%

base_tl_model = HookedTransformer.from_pretrained_no_processing(
    base_model_name,
    hf_model=base_model.to(device),  # type: ignore
    local_files_only=True,
    torch_dtype=dtype,
    device=device,
)
base_model.cpu()
clear_cuda_mem()

# %%

tuned_tl_model = HookedTransformer.from_pretrained_no_processing(
    base_model_name,
    hf_model=merged_model.to(device),  # type: ignore
    local_files_only=True,
    torch_dtype=dtype,
    device=device,
)
merged_model.cpu()
clear_cuda_mem()

# %%

def load_functions_testset(path):
    # each row: {"messages": [message dicts]}
    # this doesn't need any additional preprocessing with SFTTrainer
    ds = []

    output = []
    ans = []
    with open(path, "r") as f:
        for line in f:
            ds.append(json.loads(line))

    # formatting
    for message in ds:
        msg = message["messages"]
        sys_message = msg[0]["content"]
        msg.pop(0)
        msg[0]["content"] = (
            sys_message + "\n" + msg[0]["content"] + "\n" + msg[1]["content"]
        )

        ans.append(msg[-1]["content"])
        msg.pop(-1)
        msg.pop(-1)
        output.append(msg)

    return output, ans


ds_path = "/workspace/inductive-oocr/functions/dev/047_functions/finetune_01/047_func_01_test_oai.jsonl"

test_dataset, ans = load_functions_testset(ds_path)

# %%
i = 10

seq = test_dataset[i][0]["content"]
print(seq)
color_scale = "blues"

layers = [l for l in list(range(base_tl_model.cfg.n_layers))]
hookpoints = [
    f"blocks.{l}.{pref}"
    for l in layers
    for pref in ["hook_resid_mid", "hook_resid_post"]
]

seq_data = get_seq_data(seq, base_tl_model, tuned_tl_model, hookpoints)

# kl_div_S = seq_data.kl_div_S.detach().float().cpu().numpy()
acts_base_SLD = seq_data.acts_base_SLD.detach().float().cpu().numpy()
acts_tuned_SLD = seq_data.acts_tuned_SLD.detach().float().cpu().numpy()
input_seq_toks_S: list[str] = [
    tokenizer.decode(tok) for tok in seq_data.input_tokens_S.detach().cpu().numpy()
]

pref = 25

normed_distance_SL = normalised_distance(acts_base_SLD, acts_tuned_SLD)

px.imshow(
    title="normalized L2 distance between base and tuned model",
    img=normed_distance_SL[pref:],
    color_continuous_scale=color_scale,
    y=input_seq_toks_S[pref:],
    x=hookpoints,
    zmin=0,
    zmax=2,
    width=2000,
    height=1400,
    labels={"x": "layer", "y": "token"},
).show()

resid_mid_acts = seq_data.acts_base_SLD[:, ::2, :]
resid_post_acts = seq_data.acts_base_SLD[:, 1::2, :]

recon_resid_post_base_SLD = resid_mid_acts + run_acts_through_other_model(
    resid_mid_acts, base_tl_model
)
assert torch.allclose(recon_resid_post_base_SLD, resid_post_acts)
recon_resid_post_tuned_SLD = resid_mid_acts + run_acts_through_other_model(
    resid_mid_acts, tuned_tl_model
)


resid_post_nmse_SL = normalised_distance(
    recon_resid_post_base_SLD.detach().float().cpu().numpy(),
    recon_resid_post_tuned_SLD.detach().float().cpu().numpy(),
)

# %% 

unique_input_seq_tok_S = [f"[{i}]" + input_seq_toks_S[i] for i in range(len(input_seq_toks_S))]

px.imshow(
    title="difference in outputs of mlps on the same (base) activation",
    img=resid_post_nmse_SL,
    color_continuous_scale=color_scale,
    y=unique_input_seq_tok_S,
    x=[str(i) for i in range(base_tl_model.cfg.n_layers)],
    zmin=0,
    # zmax=2,
    width=2000,
    height=1400,
    labels={"x": "layer", "y": "token"},
).show()

# %%

from transformer_lens.hook_points import HookPoint

def mlp_patch_hook(acts, hook: HookPoint, input_toks):
    _, cache = base_tl_model.run_with_cache(input_toks)
    layer = hook.layer()
    base_tl_activ = cache[f"blocks.{layer}.hook_mlp_out"]
    del cache
    assert acts.shape == base_tl_activ.shape #[batch, seq_len, d_model]
    acts.copy_(base_tl_activ)
    return base_tl_activ

# %%
from functools import partial

def activation_patching(layer, batch_size):
    patched_ans = ""

    for i in range(0, 200, batch_size):
        tuned_tl_model.reset_hooks()
        clear_cuda_mem()

        seq = tokenizer.apply_chat_template(test_dataset[i:i+batch_size], tokenize=False, add_generation_prompt=True)
        input_toks = tuned_tl_model.to_tokens(seq, padding_side="left")

        # patched logits
        with torch.no_grad():
            logits_patch = tuned_tl_model.run_with_hooks(
                input_toks,
                return_type="logits",
                fwd_hooks=[
                    (f'blocks.{layer}.hook_mlp_out', partial(mlp_patch_hook, input_toks=input_toks)),
                ]
            )[:,-1,:].squeeze()

        logprobs_patch = logits_patch.softmax(dim=-1)
        del logits_patch
        _, labels_patch = torch.max(logprobs_patch, dim=-1)
        del logprobs_patch

        patched_ans += tuned_tl_model.to_string(labels_patch)

    return patched_ans

# %%

clear_cuda_mem()

# %%

def get_original_answers(batch_size=4):
    original_ans = ""
    for i in range(0, 4, batch_size):
        tuned_tl_model.reset_hooks()
        clear_cuda_mem()

        seq = tokenizer.apply_chat_template(test_dataset[i:i+batch_size], tokenize=False, add_generation_prompt=True)
        input_toks = tuned_tl_model.to_tokens(seq, padding_side="left")

        # original logits
        with torch.no_grad():
            logits_orig = tuned_tl_model(
                input_toks,
                return_type="logits"
            )[:,-1,:].squeeze()

        logprobs_orig = logits_orig.softmax(dim=-1)
        del logits_orig
        _, labels_orig = torch.max(logprobs_orig, dim=-1)
        del logprobs_orig
        original_ans += tuned_tl_model.to_string(labels_orig)

    return original_ans

correct_ans = ''.join(ans)
original_ans = get_original_answers(batch_size=4)

# %%

num_wrong_patched = []

for layer in [2, 6, 10, 14, 18, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]:
    clear_cuda_mem()

    patched_ans = activation_patching(layer, batch_size=4)

    num_wrong_patched.append(sum(c1 != c2 for c1, c2 in zip(correct_ans, patched_ans)))

# %%

num_wrong_patched

# %%

clear_cuda_mem()

# %%

import yaml

config_dir = "/workspace/inductive-oocr/functions/dev/047_functions/finetune_01/test_config.yaml"

with open(config_dir, "r") as f:
    data_dict = yaml.safe_load(f)

fn_names = list(data_dict['dataset']['var_dict'].keys())

# %%

import random
import re

correct_ans = ''.join(ans)
correct_toks = tuned_tl_model.to_tokens(ans, prepend_bos=False).squeeze()
print(correct_toks.shape)

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

clean_prompts = tokenizer.apply_chat_template(test_dataset, tokenize=False, add_generation_prompt=True)

dirty_prompts = create_spoiled_list(fn_names, clean_prompts)


# %%

def metric(logits, ans): 
    # ans: [batch_size]
    logits = logits[:,-1,:].squeeze(1)

    letters = ['A', 'B', 'C', 'D', 'E']
    letters_id = tuned_tl_model.to_tokens(letters, prepend_bos=False).squeeze()

    avg_letter_logits = logits[:,letters_id].mean(dim=1)

    batch_size = logits.shape[0]
    batch_indices = torch.arange(batch_size).unsqueeze(1)

    result = logits[batch_indices, ans.unsqueeze(1)].squeeze(1)
    result = result - avg_letter_logits
    result = result.mean()

    return result

# %%

from transformer_lens import patching
from functools import partial

def clean_dirty_patching():
    clean_toks = tuned_tl_model.to_tokens(clean_prompts[3:4], padding_side="left")
    dirty_toks = tuned_tl_model.to_tokens(dirty_prompts[3:4], padding_side="left")
    ans = correct_toks[3:4]

    patching_metric = partial(metric, ans=ans)
    _, clean_cache = tuned_tl_model.run_with_cache(clean_toks)

    results = patching.get_act_patch_mlp_out(
        tuned_tl_model,
        dirty_toks,
        clean_cache,
        patching_metric
    )
    return results

reuslts = clean_dirty_patching()
clear_cuda_mem()

# %%

import plotly.express as px

labels = [f"{tok} {i}" for i, tok in enumerate(tuned_tl_model.to_str_tokens(clean_prompts[3]))]

px.imshow(
    reuslts.cpu(),
    labels={"x": "Position", "y": "Layer"},
    x=labels,
    title="post-MLP Activation Patching",
    width=2000,
    height=1000,
)


# %%

# Your data
values = [x/200 for x in [24, 25, 29, 26, 44, 45, 44, 34, 34, 36, 33, 37, 29, 27, 25, 26, 27, 24, 25, 26, 25]]
layers = [i * 2 for i in range(len(values))]

# Create DataFrame
df = pd.DataFrame({
    'Layer': layers,
    'Value': values
})

# Create scatter plot
fig = px.scatter(df, x='Layer', y='Value', 
                 title='MLP patching across models',
                 labels={'Layer': 'Layer', 'Value': 'Error rate'},
                 color_discrete_sequence=['#8884d8'])

# Add horizontal baseline at y=26
fig.add_hline(y=26/200, line_dash="dash", line_color="red")

# Update layout for better appearance
fig.update_layout(
    xaxis_title="Layer",
    yaxis_title="Error rate",
    plot_bgcolor='rgba(240, 240, 240, 0.5)',
    height=500,
    width=700
)

# Customize markers
fig.update_traces(marker=dict(size=10))

# Display the figure
fig.show()
# %%
