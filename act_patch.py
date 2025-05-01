# %%
# You can either patch across models or across prompts
from collections.abc import Callable, Sequence
import itertools
import pandas as pd
import os
import copy
from typing import Optional, cast, List, Tuple, Union, Dict
from functools import partial
import yaml
import re
from dataclasses import dataclass

import plotly.express as px  # type: ignore
import torch
from peft import LoraModel, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2ForCausalLM,
)
from transformer_lens import patching, HookedTransformer
from transformer_lens.hook_points import HookPoint

from utils import clear_cuda_mem, load_test_dataset

# %%

model_name = "google/gemma-2-9b-it"
finetune_checkpoint_dir = "./checkpoints/9b-func-all-r8/checkpoint-2000/"
ds_path = "../connect_dots/functions/dev/047_functions/finetune_01/"
device = torch.device("cuda")

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation='eager',
)

tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(model_name))

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
    device=device,
)
del merged_model
clear_cuda_mem()


# %%

test_ds = load_test_dataset(os.path.join(ds_path, "047_func_01_test_oai.jsonl"))
test_prompts = test_ds["messages"]
correct_answers = test_ds["answer"]

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

clean_prompts: list[str] = tokenizer.apply_chat_template(test_prompts, tokenize=False, add_generation_prompt=True)
dirty_prompts: list[str] = create_spoiled_list(fn_names, clean_prompts)

# %%

def make_acc_increase_metric(correct_tok_id: int, correct_tok_dirty_prob: float):
    def metric_(patched_logits_BSV: torch.Tensor): 
        patched_correct_tok_prob_B = patched_logits_BSV[:,-1].softmax(dim=-1)[:, correct_tok_id]
        return (patched_correct_tok_prob_B - correct_tok_dirty_prob).mean()
    
    return metric_

# %%

def sanitize_tok(tok: str):
    return tok.replace(" ", "_").replace("\n", "\\n")

def act_patch(dirty_toks, clean_cache, metric, pos_range: Tuple[int, int], layer_range: Tuple[int, int], activation_name: str):
    rows = []
    min_pos = pos_range[0]
    max_pos = pos_range[1]
    min_layer = layer_range[0]
    max_layer = layer_range[1]
    for layer in range(min_layer, max_layer):
        for pos in range(min_pos, max_pos):
            rows.append({"layer": layer, "pos": pos})

    results_interpolated_narrow, index_df_narrow = patching.generic_activation_patch(
        model=tuned_tl_model,
        corrupted_tokens=dirty_toks,
        clean_cache=clean_cache,
        patching_metric=metric,
        patch_setter=patching.layer_pos_patch_setter,
        activation_name=activation_name,
        index_df=pd.DataFrame(rows),
        return_index_df=True,
    )

    vis = torch.zeros(max_layer - min_layer, max_pos - min_pos) * torch.nan
    for i, row in index_df_narrow.iterrows():
        vis[row["layer"] - min_layer, row["pos"] - min_pos] = results_interpolated_narrow[i]

    # title = f"Abs Increase Prob Patching {activation_name}"

    return vis.cpu() # , title



# %%

import plotly.graph_objects as go
from plotly.subplots import make_subplots

@dataclass
class PatchingData:
    correct_tok_clean_prob: float
    correct_tok_dirty_prob: float
    mlp_out: torch.Tensor
    resid_pre: torch.Tensor
    x_labels: List[str]
    y_labels: List[int]
    clean_prompt: str
    dirty_prompt: str
    correct_ans: str

def generate_patching_data_for_indices(indices: List[int], pos_range: Tuple[int, int], layer_range: Tuple[int, int]):
    """Generates activation patching data for multiple indices and activation types."""
    generated_data: Dict[int, PatchingData] = {}
    for index in indices:
        print(f"\n--- Generating Data for Index: {index} ---")
        clean_prompt = clean_prompts[index]
        dirty_prompt = dirty_prompts[index]
        correct_answer = correct_answers[index]

        clean_toks_S = tuned_tl_model.to_tokens(clean_prompt, padding_side="left", prepend_bos=False)[0]
        dirty_toks_S = tuned_tl_model.to_tokens(dirty_prompt, padding_side="left", prepend_bos=False)[0]

        if clean_toks_S.shape != dirty_toks_S.shape:
            # print([tokenizer.decode(tok) for tok in clean_toks_S[20:80]])
            # print([tokenizer.decode(tok) for tok in dirty_toks_S[20:80]])
            print(f"skipping index {index} because of shape mismatch")
            continue

        ans_1 = tuned_tl_model.to_tokens(correct_answer, prepend_bos=False)[0]
        assert ans_1.shape == (1,)
        correct_tok_id = ans_1.item()

        with torch.no_grad():
            _, clean_cache = tuned_tl_model.run_with_cache(clean_toks_S)
            dirty_logits_SV: torch.Tensor = tuned_tl_model(dirty_toks_S, return_type="logits")[0]
            clean_logits_SV: torch.Tensor = tuned_tl_model(clean_toks_S, return_type="logits")[0]
            assert dirty_logits_SV.shape == (dirty_toks_S.shape[0], tuned_tl_model.cfg.d_vocab)

        correct_tok_dirty_prob = dirty_logits_SV[-1].softmax(dim=-1)[correct_tok_id]
        correct_tok_clean_prob = clean_logits_SV[-1].softmax(dim=-1)[correct_tok_id]

        if correct_tok_clean_prob < 0.8:
            print(f"Skipping index {index} because correct_tok_clean_prob is too low: {correct_tok_clean_prob}")
            continue
        
        if correct_tok_dirty_prob > 0.1:
            print(f"Skipping index {index} because correct_tok_dirty_prob is too high: {correct_tok_dirty_prob}")
            continue

        metric = make_acc_increase_metric(
            correct_tok_id=correct_tok_id,
            correct_tok_dirty_prob=correct_tok_dirty_prob,
        )

        data_mlp = act_patch(
            dirty_toks_S, clean_cache, metric, pos_range, layer_range, "mlp_out"
        )

        data_resid = act_patch(
            dirty_toks_S, clean_cache, metric, pos_range, layer_range, "resid_pre"
        )

        min_pos = pos_range[0]
        max_pos = pos_range[1]
        # x_labels = [f"{i} |{sanitize_tok(tok)}|" for i, tok in enumerate(tuned_tl_model.to_str_tokens(dirty_toks_S[min_pos:max_pos]))]
        x_labels = [f"{i} |{sanitize_tok(tok)}|" for i, tok in list(enumerate(tuned_tl_model.to_str_tokens(dirty_toks_S)))[min_pos:max_pos]]

        min_layer = layer_range[0]
        max_layer = layer_range[1]
        y_labels = list(range(min_layer, max_layer))

        generated_data[index] = PatchingData(
            correct_tok_clean_prob=correct_tok_clean_prob,
            correct_tok_dirty_prob=correct_tok_dirty_prob,
            mlp_out=data_mlp,
            resid_pre=data_resid,
            x_labels=x_labels,
            y_labels=y_labels,
            clean_prompt=clean_prompts[index],
            dirty_prompt=dirty_prompts[index],
            correct_ans=correct_answers[index]
        )

        clear_cuda_mem() # Clear memory after each index

    return generated_data

def composed_visualisation(generated_data: Dict[int, PatchingData]):
    """Visualises pre-generated activation patching data with indices as rows and activation types as columns."""
    valid_indices = [idx for idx, data in generated_data.items() if data is not None]
    if not valid_indices:
        print("No valid data to plot.")
        return None

    # Create subplot titles dynamically
    column_titles = ["MLP Out", "Resid Pre"]
    subplot_titles = []
    for idx in valid_indices:
        index_data = generated_data[idx]
        subplot_titles.append(f"example {idx} - {column_titles[0]} | dirty: {index_data.correct_tok_dirty_prob:.2f} | clean: {index_data.correct_tok_clean_prob:.2f}")
        subplot_titles.append(f"example {idx} - {column_titles[1]} | dirty: {index_data.correct_tok_dirty_prob:.2f} | clean: {index_data.correct_tok_clean_prob:.2f}")

    fig = make_subplots(
        rows=len(valid_indices),
        cols=len(column_titles),
        subplot_titles=subplot_titles,
        # shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.08, # Adjust spacing between plots
        horizontal_spacing=0.10,
    )

    plot_row_index = 1
    for index in valid_indices:
        index_data = generated_data[index]
        x_labels = index_data.x_labels
        y_labels = index_data.y_labels

        # MLP Out (Col 1)
        data_mlp = index_data.mlp_out
        if data_mlp is not None:
            fig.add_trace(
                go.Heatmap(
                    z=data_mlp,
                    x=x_labels,
                    y=y_labels,
                    coloraxis="coloraxis",
                    name=f"Idx {index} MLP",
                ),
                row=plot_row_index, col=1
            )

        # Resid Pre (Col 2)
        data_resid = index_data.resid_pre
        if data_resid is not None:
            fig.add_trace(
                go.Heatmap(
                    z=data_resid,
                    x=x_labels,
                    y=y_labels,
                    coloraxis="coloraxis",
                    name=f"Idx {index} Resid",
                ),
                row=plot_row_index, col=2
            )

        plot_row_index += 1

    fig.update_layout(
        height=1000 * len(valid_indices) + 50, # Adjust height based on number of plots
        width=3000,
        title_text="Activation Patching",
        coloraxis=dict(colorscale='RdBu', cmin=-1, cmax=1, colorbar_title="Prob Increase"),
        hovermode='closest',
    )

    # Update x-axis and y-axis titles for all subplots
    for r in range(1, len(valid_indices) + 1):
        for c in range(1, len(column_titles) + 1):
            fig.update_xaxes(title_text="Position", row=r, col=c)
            fig.update_yaxes(title_text="Layer", row=r, col=c)

    return fig

# %%

generated_data = generate_patching_data_for_indices(
    indices=[0],
    pos_range=(30, 45),
    layer_range=(0, 17),
)

# %%

(composed_fig := composed_visualisation(generated_data))

# %%

clear_cuda_mem()

# %%
from transformer_lens import utils

def make_hook(pos: int, clean_activation_BSD: torch.Tensor):
    def hook(corrupted_activation_BSD: torch.Tensor, hook):
        assert corrupted_activation_BSD.shape == clean_activation_BSD.shape
        corrupted_activation_BSD[:, pos] = clean_activation_BSD[:, pos]
        return corrupted_activation_BSD
    return hook


def patch_multiple_points(dirty_toks_S: torch.Tensor, clean_toks_S: torch.Tensor, positions: list[tuple[str, int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    all_layer_names = set([layer_name for layer_name, _, _ in positions])
    _, clean_cache = tuned_tl_model.run_with_cache(clean_toks_S, names_filter=lambda concrete_layer_name: any(layer_name in concrete_layer_name for layer_name in all_layer_names))
    dirty_logits_SV = tuned_tl_model(dirty_toks_S, return_type="logits")[0]

    fwd_hooks = []
    for layer_name, pos, layer in positions:
        act_name = utils.get_act_name(layer_name, layer=layer)
        layer_act_BSD = clean_cache[act_name]
        fwd_hooks.append((act_name, make_hook(pos, layer_act_BSD)))
    
    patched_logits_SV = tuned_tl_model.run_with_hooks(dirty_toks_S, fwd_hooks=fwd_hooks)[0]

    return dirty_logits_SV, patched_logits_SV

def run_chunk_patching(idx: int, patchpoints: list[tuple[str, int, int]]):
    correct_tok_id = tuned_tl_model.to_tokens(correct_answers[idx], prepend_bos=False).item()

    dirty_prompt = dirty_prompts[idx]# .replace("odgrps", "grpsod")
    dirty_toks_S = tuned_tl_model.to_tokens(dirty_prompt, prepend_bos=False)[0]

    clean_prompt = clean_prompts[idx]
    clean_toks_S = tuned_tl_model.to_tokens(clean_prompt, prepend_bos=False)[0]

    assert dirty_toks_S.shape == clean_toks_S.shape

    with torch.no_grad():
        dirty_logits_SV, patched_logits_SV = patch_multiple_points(dirty_toks_S, clean_toks_S, patchpoints)
        correct_tok_dirty_prob = dirty_logits_SV[-1].softmax(dim=-1)[correct_tok_id]
        correct_tok_patched_prob = patched_logits_SV[-1].softmax(dim=-1)[correct_tok_id]
        correct_tok_prob_increase = (correct_tok_patched_prob - correct_tok_dirty_prob)

        correct_tok_clean_prob = tuned_tl_model.forward(clean_toks_S, return_type="logits")[0, -1].softmax(dim=-1)[correct_tok_id]

        return {
            "clean_prob": correct_tok_clean_prob.item(),
            "original_prob": correct_tok_dirty_prob.item(),
            "patched_prob": correct_tok_patched_prob.item(),
            "prob_increase": correct_tok_prob_increase.item(),
        }

# %%

import itertools

positions = [40, 41, 42]
layers = list(range(2, 8))
points = ["mlp_out"]

patchpoints = list(itertools.product(points, positions, layers))

rows = []
for i in range(10):
    try:
        res = run_chunk_patching(i, patchpoints)
        if res["clean_prob"] < 0.9:
            continue
        rows.append(res)
    except Exception as e:
        print(i, 'rerro')

print(pd.DataFrame(rows))

# %%
# idx 0 seems to be only patchable on the 2nd mention

positions = [40, 41, 42]
layers = [1, 2, 3, 4]
hook_names = ["mlp_out"]

patchpoints = list(itertools.product(hook_names, positions, layers))

res = run_chunk_patching(0, patchpoints)
res

# %%

# print(f"patching {len(patchpoints)} points increased the probability of the correct token by {res['prob_increase']:.2f}")
