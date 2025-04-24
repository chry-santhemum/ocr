# %%
# You can either patch across models or across prompts
import itertools
import pandas as pd
import os
import copy
import plotly.express as px
from typing import cast, List, Tuple, Dict
from tqdm import tqdm
import yaml
import re
from dataclasses import dataclass

import torch
from peft import LoraModel, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer  # type: ignore
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
from transformer_lens import patching, HookedTransformer, ActivationCache

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
            replacement = "odgrps"
            # Replace the function name
            spoiled_string = re.sub(r'\b' + re.escape(found) + r'\b', replacement, string)
            spoiled_strings.append(spoiled_string)
        else:
            # If no function was found, append the original
            print("None matched")
            spoiled_strings.append(string)
    
    return spoiled_strings

clean_prompts: list[str] = tokenizer.apply_chat_template(test_prompts, tokenize=False, add_generation_prompt=True)  # type: ignore
dirty_prompts: list[str] = create_spoiled_list(fn_names, clean_prompts)

# %%

letters = ['A', 'B', 'C', 'D', 'E']
letter_toks: list[int] = [tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]
letter_toks

# %% 

def top_logits(logits_V: torch.Tensor):
    top = logits_V.topk(5, dim=-1)
    return "\n".join(
        [
            f"{tokenizer.decode(tok)} {prob.item():.3f}"
            for tok, prob in zip(top.indices, top.values)
        ]
    )

def top_probs(logits_V: torch.Tensor):
    return top_logits(logits_V.softmax(dim=-1))


def sanitize_tok(tok: str):
    return tok.replace(" ", "_").replace("\n", "\\n")
# %%

def act_patch(
    dirty_toks_S: torch.Tensor,
    clean_cache: ActivationCache,
    metric,
    activation_name: str,
    pos_range: Tuple[int, int] | None = None,
    layer_range: Tuple[int, int] | None = None,
):
    min_pos = pos_range[0] if pos_range else 0
    max_pos = pos_range[1] if pos_range else dirty_toks_S.shape[0]
    min_layer = layer_range[0] if layer_range else 0
    max_layer = layer_range[1] if layer_range else tuned_tl_model.cfg.n_layers

    rows = [
        {"layer": layer, "pos": pos}
        for layer in range(min_layer, max_layer)
        for pos in range(min_pos, max_pos)
    ]

    results_interpolated_narrow, index_df_narrow = patching.generic_activation_patch(
        model=tuned_tl_model,
        corrupted_tokens=dirty_toks_S,
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

    return vis


import plotly.graph_objects as go
from plotly.subplots import make_subplots

def make_correct_logit_increase_metric(correct_tok_id: int, baseline_logits_BSV: torch.Tensor):
    other_toks = [l for l in letter_toks if l != correct_tok_id]

    def correct_vs_mean_others_logit_diff(logits_BSV: torch.Tensor):
        correct_logit_B = logits_BSV[:,-1, correct_tok_id]
        other_toks_logit_B = logits_BSV[:,-1, other_toks].mean(dim=-1)
        return (correct_logit_B - other_toks_logit_B).mean()

    # get the baseline logit diff:
    baseline_diff = correct_vs_mean_others_logit_diff(baseline_logits_BSV)

    def _metric(patched_logits_BSV: torch.Tensor): 
        patched_diff = correct_vs_mean_others_logit_diff(patched_logits_BSV)
        return patched_diff - baseline_diff

    return baseline_diff, _metric


@dataclass
class PatchingData:
    correct_tok_clean_prob: float
    correct_tok_dirty_prob: float
    # mlp_out: torch.Tensor
    resid_pre: torch.Tensor
    x_labels: List[str]
    y_labels: List[int]
    clean_prompt: str
    dirty_prompt: str
    correct_ans: str

def generate_patching_data_for_indices(
    indices: List[int],
    pos_range: Tuple[int, int] | None = None,
    layer_range: Tuple[int, int] | None = None,
):
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
            print(f"skipping index {index} because of shape mismatch")
            continue

        ans_1 = tuned_tl_model.to_tokens(correct_answer, prepend_bos=False)[0]
        assert ans_1.shape == (1,)
        correct_tok_id = cast(int, ans_1.item())

        with torch.no_grad():
            _, clean_cache = tuned_tl_model.run_with_cache(clean_toks_S)
            dirty_logits_SV: torch.Tensor = tuned_tl_model(dirty_toks_S, return_type="logits")[0]
            clean_logits_SV: torch.Tensor = tuned_tl_model(clean_toks_S, return_type="logits")[0]
            assert dirty_logits_SV.shape == (dirty_toks_S.shape[0], tuned_tl_model.cfg.d_vocab)

        correct_tok_dirty_prob = dirty_logits_SV[-1].softmax(dim=-1)[correct_tok_id].item()
        correct_tok_clean_prob = clean_logits_SV[-1].softmax(dim=-1)[correct_tok_id].item()

        if correct_tok_clean_prob < 0.8:
            print(f"Skipping index {index} because correct_tok_clean_prob is too low: {correct_tok_clean_prob}")
            continue
        
        if correct_tok_dirty_prob > 0.1:
            print(f"Skipping index {index} because correct_tok_dirty_prob is too high: {correct_tok_dirty_prob}")
            continue

        _, increase_vs_dirty = make_correct_logit_increase_metric(correct_tok_id=correct_tok_id, baseline_logits_BSV=dirty_logits_SV[None])

        # data_mlp = act_patch(dirty_toks_S, clean_cache, increase_vs_dirty, "mlp_out", pos_range, layer_range)
        data_resid = act_patch(dirty_toks_S, clean_cache, increase_vs_dirty, "resid_pre", pos_range, layer_range)

        dirty_toks = tuned_tl_model.to_str_tokens(dirty_toks_S)
        asdf = list(enumerate(dirty_toks))[pos_range[0]:pos_range[1]] if pos_range else list(enumerate(dirty_toks))
        x_labels = [f"{i} |{sanitize_tok(tok)}|" for i, tok in asdf]

        y_labels = list(range(*layer_range)) if layer_range else list(range(tuned_tl_model.cfg.n_layers))

        generated_data[index] = PatchingData(
            correct_tok_clean_prob=correct_tok_clean_prob,
            correct_tok_dirty_prob=correct_tok_dirty_prob,
            # mlp_out=data_mlp.cpu(),
            resid_pre=data_resid.cpu(),
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
        # data_mlp = index_data.mlp_out
        # if data_mlp is not None:
        #     fig.add_trace(
        #         go.Heatmap(
        #             z=data_mlp,
        #             x=x_labels,
        #             y=y_labels,
        #             coloraxis="coloraxis",
        #             name=f"Idx {index} MLP",
        #         ),
        #         row=plot_row_index, col=1
        #     )

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
        height=400 * len(valid_indices) + 50, # Adjust height based on number of plots
        width=len(x_labels) * 35 + 100,
        title_text="Activation Patching",
        coloraxis=dict(colorscale='RdBu', cmid=0, colorbar_title="Prob Increase"),
        hovermode='closest',
    )

    # Update x-axis and y-axis titles for all subplots
    for r in range(1, len(valid_indices) + 1):
        for c in range(1, len(column_titles) + 1):
            fig.update_xaxes(title_text="Position", row=r, col=c)
            fig.update_yaxes(title_text="Layer", row=r, col=c)

    return fig

def visualise_patching_for_index(idx: int, pos_range: Tuple[int, int] | None = None, layer_range: Tuple[int, int] | None = None):
    correct_tok_id = cast(int, tuned_tl_model.to_tokens(correct_answers[idx], prepend_bos=False).item())
    dirty_toks_S = tuned_tl_model.to_tokens(dirty_prompts[idx], prepend_bos=False)[0]
    clean_toks_S = tuned_tl_model.to_tokens(clean_prompts[idx], prepend_bos=False)[0]
    assert dirty_toks_S.shape == clean_toks_S.shape
    with torch.no_grad():
        _, clean_cache = tuned_tl_model.run_with_cache(clean_toks_S)
        _, dirty_cache = tuned_tl_model.run_with_cache(dirty_toks_S)
        dirty_logits_BSV: torch.Tensor = tuned_tl_model(dirty_toks_S, return_type="logits")
        clean_logits_BSV: torch.Tensor = tuned_tl_model(clean_toks_S, return_type="logits")

    _, increase_vs_dirty_metric = make_correct_logit_increase_metric(
        correct_tok_id=correct_tok_id,
        baseline_logits_BSV=dirty_logits_BSV,
    )

    _, increase_vs_clean_metric = make_correct_logit_increase_metric(
        correct_tok_id=correct_tok_id,
        baseline_logits_BSV=clean_logits_BSV,
    )

    # MANUALLY COMMENTING OUT THE FOLLOWING TO DO DENOISING VS NOISING
    x = act_patch(
        # dirty_toks_S,
        clean_toks_S,
        # clean_cache,
        dirty_cache,
        # increase_vs_dirty_metric,
        increase_vs_clean_metric,
        pos_range=pos_range,
        layer_range=layer_range,
        activation_name="mlp_out",
    )

    return x

# %%

x = visualise_patching_for_index(2, pos_range=(30, 45), layer_range=(0, 17))
px.imshow(x, color_continuous_scale="RdBu", color_continuous_midpoint=0)

# %%

# generated_data = generate_patching_data_for_indices(
#     indices=[2], #  1, 2, 3],
#     pos_range=(30, 45),
#     layer_range=(0, 17),
# )

# # %%

# (composed_fig := composed_visualisation(generated_data))

# %%

clear_cuda_mem()

# %%
from transformer_lens import utils

def make_seq_position_hook(pos: int, clean_activation_BSD: torch.Tensor):
    def hook(corrupted_activation_BSD: torch.Tensor, hook):
        assert corrupted_activation_BSD.shape == clean_activation_BSD.shape
        corrupted_activation_BSD[:, pos] = clean_activation_BSD[:, pos]
        return corrupted_activation_BSD
    return hook


def run_chunk_patching(idx: int, patchpoints: list[tuple[str, int, int]]):
    """patches multiple points during a single forward pass, useful for trying to find a 'minimal subset' of patching locations
    to recreate a behaviour
    """
    correct_tok_id = cast(int, tuned_tl_model.to_tokens(correct_answers[idx], prepend_bos=False).item())
    dirty_toks_S = tuned_tl_model.to_tokens(dirty_prompts[idx], prepend_bos=False)[0]
    clean_toks_S = tuned_tl_model.to_tokens(clean_prompts[idx], prepend_bos=False)[0]
    assert dirty_toks_S.shape == clean_toks_S.shape
    return _run_chunk_patching(dirty_toks_S, clean_toks_S, correct_tok_id, patchpoints)

@torch.no_grad()
def _run_chunk_patching(
    dirty_toks_S: torch.Tensor,
    clean_toks_S: torch.Tensor,
    correct_tok_id: int,
    patchpoints: list[tuple[str, int, int]],
):
    all_layer_names = set([layer_name for layer_name, _, _ in patchpoints])

    def filter(concrete_layer_name):
        return any(layer_name in concrete_layer_name for layer_name in all_layer_names)

    dirty_toks_logits_BSV, dirty_cache = tuned_tl_model.run_with_cache(dirty_toks_S, names_filter=filter, return_type="logits")
    dirty_toks_logits_SV = dirty_toks_logits_BSV[0]

    clean_toks_logits_BSV, clean_cache = tuned_tl_model.run_with_cache(clean_toks_S, names_filter=filter, return_type="logits")
    clean_toks_logits_SV = clean_toks_logits_BSV[0]

    patch_dirty_cache_hooks = []
    patch_clean_cache_hooks = []
    for act_type, pos, layer in patchpoints:
        act_name = utils.get_act_name(act_type, layer=layer)
        patch_dirty_cache_hooks.append((act_name, make_seq_position_hook(pos, dirty_cache[act_name])))
        patch_clean_cache_hooks.append((act_name, make_seq_position_hook(pos, clean_cache[act_name])))

    # noising = "add noise" by running clean tokens with a dirty cache
    noising_logits_SV: torch.Tensor = tuned_tl_model.run_with_hooks(clean_toks_S, fwd_hooks=patch_dirty_cache_hooks)[0]
    pre_noising_baseline_answer_confidence, get_noising_uplift = make_correct_logit_increase_metric(correct_tok_id, clean_toks_logits_SV[None])
    noising_uplift = get_noising_uplift(noising_logits_SV[None]).item()

    # denoising = "remove noise" by running dirty tokens with a clean cache
    denoising_logits_SV: torch.Tensor = tuned_tl_model.run_with_hooks(dirty_toks_S, fwd_hooks=patch_clean_cache_hooks)[0]
    pre_denoising_baseline_answer_confidence, get_denoising_uplift = make_correct_logit_increase_metric(correct_tok_id, dirty_toks_logits_SV[None])
    denoising_uplift = get_denoising_uplift(denoising_logits_SV[None]).item()

    return {
        "pre_noising_baseline_answer_confidence": pre_noising_baseline_answer_confidence,
        "noising_uplift": noising_uplift,
        "pre_denoising_baseline_answer_confidence": pre_denoising_baseline_answer_confidence,
        "denoising_uplift": denoising_uplift,
    }
# %%

import itertools

positions = [40, 41, 42]
layers = list(range(2, 8))
points = ["mlp_out"]

patchpoints = list(itertools.product(points, positions, layers))

# %%

labels = []
func_name_regex = re.compile(r"Which option correctly describes (\w+)")
for i in range(len(clean_prompts)):
    res = func_name_regex.search(clean_prompts[i])
    if res is None:
        raise ValueError(f"No match found for {clean_prompts[i]}")
    groups = res.groups()
    if len(groups) != 1:
        raise ValueError(f"Expected 1 group, got {len(groups)} for {clean_prompts[i]}")
    labels.append(groups[0])


# %%
rows = []
for idx in tqdm(range(len(clean_prompts))):
    try:
        label = labels[idx]
        correct_tok_id = cast(int, tuned_tl_model.to_tokens(correct_answers[idx], prepend_bos=False).item())
        dirty_toks_S = tuned_tl_model.to_tokens(dirty_prompts[idx], prepend_bos=False)[0]
        clean_toks_S = tuned_tl_model.to_tokens(clean_prompts[idx], prepend_bos=False)[0]
        assert dirty_toks_S.shape == clean_toks_S.shape, f"Dirty: {dirty_toks_S.shape} != Clean: {clean_toks_S.shape}, label: {label}"
        res = _run_chunk_patching(
            dirty_toks_S,
            clean_toks_S,
            correct_tok_id,
            patchpoints,
        )
        res["function_name"] = label
        rows.append(res)
    except Exception as e:
        print(f"Error on {idx}: {e}")

# %%

len(rows)

# %%

df = pd.DataFrame(rows)

# %%
df.columns
# %%

# Melt the DataFrame for violin plot
uplift_df_melted = df.melt(
    id_vars=['function_name'],
    value_vars=['noising_uplift', 'denoising_uplift'], # , 'pre_noising_baseline_answer_confidence', 'pre_denoising_baseline_answer_confidence'],
    var_name='intervention',
    value_name='uplift',
)

px.violin(
    uplift_df_melted,
    x="function_name",
    y="uplift",
    color="intervention",  # Color by noising/denoising
    box=True,  # Show box plot inside violins
    # points="all",  # Show individual data points
    title="Distribution of Logit Increase by Label and Type (Noising vs. Denoising)",
    width=2000,
).show()

# %%

px.box(
    uplift_df_melted[['']], # Use the melted dataframe
    x="function_name",
    y="uplift", # Use the value column from the melted df
    color="intervention", # Color boxes by noising/denoising
    title="Uplift in (correct_tok_logit - mean(other_token_logits))",
    subtitle="for noising vs. denoising patching from position 40 to 42 at layers 2 to 7",
    points="all" # Optionally show all points
)
# %%

confidence_df = df.melt(
    id_vars=['function_name'],
    value_vars=['dirty_confidence', 'clean_confidence'],
    var_name='intervention',
    value_name='confidence',
)

px.box(
    confidence_df,
    x="function_name",
    y="confidence",
    color="intervention",
    title="Confidence in correct logit (correct_tok_logit - mean(other_token_logits))",
    subtitle="for clean vs. dirty prompts",
)
# %%

# idx 0 seems to be only patchable on the 2nd mention

positions = [40, 41, 42]
layers = [1, 2, 3, 4]
hook_names = ["mlp_out"]

patchpoints = list(itertools.product(hook_names, positions, layers))

res = run_chunk_patching(0, patchpoints)