# %%
from pathlib import Path
from random import shuffle
# from matplotlib.pyplot import pie
# from sae_lens import SAE, HookedSAETransformer
import pandas as pd
from transformer_lens import HookedTransformer
from transformers import PreTrainedModel, PreTrainedTokenizer
import os
import torch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# from data.trl.examples.research_projects.tools.triviaqa import tool_fn
# from steering import green
# from act_patch_oli import top_logits
from utils import TokenwiseSteeringHook, CITY_ID_TO_NAME, CITY_IDS
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
# %%

EXPERIMENTS = Path('data/experiments')

model_size = '9b'
all_experiment_dirs = [
    *list(EXPERIMENTS.glob(f'cities_google_gemma*{model_size}*')),
    *list(EXPERIMENTS.glob(f'cities_google/gemma*{model_size}*'))
]
all_experiment_dirs
# %%

exps: list[dict[str, torch.Tensor]] = []
for exp_dir in all_experiment_dirs:
    steps = list(exp_dir.glob('step_*'))
    if len(steps) > 0:
        exp_dict = {}
        for cid in CITY_IDS:
            exp_dict[cid] = []
            for step in steps:
                vec = torch.load(step / f'{cid}.pt', map_location=device)
                exp_dict[cid].append(vec)
        exps.append(exp_dict)
# %%
# vecs = exps[10][50337]

exp = Path("data/experiments/cities_google_gemma-2-9b-it_layer3_20250430_042709")

steps = list(exp.glob('step_*'))

vecs = {cid: [] for cid in CITY_IDS}
for cid, l in vecs.items():
    for step in steps:
        vec = torch.load(step / f'{cid}.pt', map_location=device)
        l.append(vec)

paris_stacked = torch.stack(vecs[50337])
print(paris_stacked.shape)

diffs = paris_stacked[1:] - paris_stacked[:-1]

# rolling avg across first dimension
window_size = 15
rolling_avg = torch.zeros(paris_stacked.shape[0] - window_size + 1, paris_stacked.shape[1])
for i in range(paris_stacked.shape[0] - window_size + 1):
    window = paris_stacked[i:i+window_size]
    assert window.shape == (window_size, paris_stacked.shape[1]), window.shape
    rolling_avg[i] = torch.mean(window, dim=0)

diffs = rolling_avg[1:] - rolling_avg[:-1]

cosine_sims = torch.nn.functional.cosine_similarity(diffs[None], diffs[:, None], dim=-1)

px.imshow(cosine_sims.detach().float().cpu().numpy(), color_continuous_scale="RdBu", color_continuous_midpoint=0)

# %%

all_city_vecs = []
city_sizes = []
training_run_sizes = []
for cid in CITY_IDS:
    vecs = []
    for exp in exps:
        # take 5 evenly spaced vectors from each training run
        steps = exp[cid][::max(1, len(exp[cid])//5)]
        vecs.extend(steps)
        training_run_sizes.append(len(steps))
    vecs_VD = torch.stack(vecs).to(device)
    all_city_vecs.append(vecs_VD)
    city_sizes.append(vecs_VD.shape[0])

combined_vecs_VD = torch.cat(all_city_vecs, dim=0)
combined_pairwise_cosine_sims = torch.nn.functional.cosine_similarity(combined_vecs_VD[None].cpu(), combined_vecs_VD[:, None].cpu(), dim=-1)

fig = go.Figure()
fig.add_trace(go.Heatmap(z=combined_pairwise_cosine_sims.detach().float().cpu().numpy(), coloraxis="coloraxis"))

# Add separator lines
shapes = []
current_idx = 0
total_size = combined_vecs_VD.shape[0]
city_boundaries = [0] + list(torch.cumsum(torch.tensor(city_sizes), dim=0).numpy())
run_boundaries = [0] + list(torch.cumsum(torch.tensor(training_run_sizes), dim=0).numpy())

for i in range(len(CITY_IDS)):
    split_idx = city_boundaries[i+1]
    if i < len(CITY_IDS) - 1:  # Don't add lines after the last section
        # Horizontal line
        shapes.append(go.layout.Shape(
            type="line",
            y0=split_idx - 0.5,
            y1=split_idx - 0.5,
            x0=-0.5,
            x1=total_size - 0.5,
            line=dict(color="green", width=2)
        ))
        # Vertical line
        shapes.append(go.layout.Shape(
            type="line",
            x0=split_idx - 0.5,
            x1=split_idx - 0.5,
            y0=-0.5,
            y1=total_size - 0.5,
            line=dict(color="green", width=2)
        ))

for run_boundary in run_boundaries:
    shapes.append(go.layout.Shape(
        type="line",
        x0=run_boundary - 0.5,
        x1=run_boundary - 0.5,
        y0=-0.5,
        y1=total_size - 0.5,
        line=dict(color="red", width=0.2)
    ))
    shapes.append(go.layout.Shape(
        type="line",
        x0=-0.5,
        x1=total_size - 0.5,
        y0=run_boundary - 0.5,
        y1=run_boundary - 0.5,
        line=dict(color="red", width=0.2)
    ))

fig.update_layout(
    title="Cosine similarities between city vectors across different training runs<br>(Cities separated by green lines, training runs separated by red lines)",
    coloraxis={'colorscale':'RdBu', 'cmid':0},
    width=800, # Adjusted width for a single large plot
    height=800, # Make it square
    shapes=shapes,
    xaxis_title="Combined Vector Index",
    yaxis_title="Combined Vector Index"
)
fig.show()

# %%


exp = Path("data/experiments/cities_google_gemma-2-9b-it_layer4_20250429_103911")

step_path = exp / "step_80"

files = os.listdir(step_path)
vecs = []
for file in files:
    vec = torch.load(step_path / file, map_location=device)
    vecs.append(vec)

vecs_VD = torch.stack(vecs).to(device)

pairwise_cosine_sims = torch.nn.functional.cosine_similarity(vecs_VD[None], vecs_VD[:, None], dim=-1)

print(pairwise_cosine_sims.shape)

px.imshow(
    title="Cosine similarities between city vectors",
    img=pairwise_cosine_sims.detach().float().cpu().numpy(), color_continuous_scale="RdBu", color_continuous_midpoint=0
)

# %%

exp = Path("data/experiments/cities_google_gemma-2-9b-it_layer4_20250429_224752")

def get_vecs(device, base, id, path):
    base_vec_path = base / path / str(id)
    files = os.listdir(base_vec_path)
    vecs = []
    for file in files:
        vec = torch.load(base_vec_path / file, map_location=device)
        vecs.append(vec)
    vecs_SD = torch.stack(vecs).to(device)
    return vecs_SD

fig = make_subplots(rows=1, cols=len(CITY_IDS), subplot_titles=[f"City {cid} ({CITY_ID_TO_NAME[cid]})" for cid in CITY_IDS])

for i, cid in enumerate(CITY_IDS):
    grads_SD = get_vecs(device, exp, cid, 'grads')
    pairwise_cosine_sims = torch.nn.functional.cosine_similarity(grads_SD[None], grads_SD[:, None], dim=-1)
    heatmap = go.Heatmap(
        z=pairwise_cosine_sims.detach().float().cpu().numpy(),
        coloraxis="coloraxis"
    )
    fig.add_trace(heatmap, row=1, col=i+1)

fig.update_layout(
    title_text="Cosine similarities of gradients of steering vector by training step",
    coloraxis={'colorscale':'RdBu', 'cmid':0},
    width=1500,
)
fig.show()
# %%
exp
# %%

exp_2 = Path("data/experiments/cities_google_gemma-2-9b-it_layer4_20250429_222901")

fig = make_subplots(rows=1, cols=len(CITY_IDS), subplot_titles=[f"City {cid} ({CITY_ID_TO_NAME[cid]})" for cid in CITY_IDS])

split_indices = []
total_sizes = []
for i, cid in enumerate(CITY_IDS):
    vecs = get_vecs(device, exp, cid, 'vecs')
    vecs_2 = get_vecs(device, exp_2, cid, 'vecs')
    vecs_combined = torch.cat([vecs, vecs_2])
    split_idx = vecs.shape[0]
    total_size = vecs_combined.shape[0]
    split_indices.append(split_idx)
    total_sizes.append(total_size)
    cosine_sims = torch.nn.functional.cosine_similarity(vecs_combined[None], vecs_combined[:, None], dim=-1)
    heatmap = go.Heatmap(
        z=cosine_sims.detach().float().cpu().numpy(),
        coloraxis="coloraxis"
    )
    fig.add_trace(heatmap, row=1, col=i+1)

# Add lines to separate concatenated sections
shapes = []
for i in range(len(CITY_IDS)):
    split_idx = split_indices[i]
    total_size = total_sizes[i]
    col_idx = i + 1 # Subplot column index is 1-based
    # Horizontal line
    shapes.append(go.layout.Shape(
        type="line",
        yref=f"y{col_idx}",
        xref=f"x{col_idx}",
        y0=split_idx - 0.5,
        y1=split_idx - 0.5,
        x0=-0.5,
        x1=total_size - 0.5,
        line=dict(color="green", width=2)
    ))
    # Vertical line
    shapes.append(go.layout.Shape(
        type="line",
        yref=f"y{col_idx}",
        xref=f"x{col_idx}",
        x0=split_idx - 0.5,
        x1=split_idx - 0.5,
        y0=-0.5,
        y1=total_size - 0.5,
        line=dict(color="green", width=2)
    ))

fig.update_layout(
    title_text="Cosine similarities of city vectors by training step<br>Two different runs separated by green lines",
    coloraxis={'colorscale':'RdBu', 'cmid':0},
    width=1500,
    shapes=shapes
)
fig.show()
# %%


# %%


# letters = ["A", "B", "C", "D", "E"]
# # for idx, cid in enumerate(CITY_IDS):
# idx = 0
# cid = CITY_IDS[idx]

# prompt_txt = (
#     f"What city is represented by City {cid}? Please respond with the letter of the correct answer only.\n\n" +
#     "\n".join(f"{l}: {name}" for l, name in zip(letters, CITY_ID_TO_NAME.values())) +
#     "\n\n"
# )

# print(prompt_txt)

# # print(f"prompt_txt: {prompt_txt}")
# # tokenise prompt the *same way as training* (chat template, no gen-prompt)
# messages = [{"role": "user", "content": prompt_txt}]
# input_ids, occ = tokenize_and_mark_cities(
#     messages, model.tokenizer, add_generation_prompt=True
# )

# ids_S = torch.tensor(input_ids, device=device)
# occs_S = torch.tensor(occ, device=device)

# # print(model.tokenizer.decode(ids_S[occs_S != -1]))
# # %%

# buffered_vecs_VD = torch.cat([vecs_VD, torch.zeros(1, vecs_VD.shape[1], device=device)], dim=0)

# pred_logits = model.forward(ids_S[None])[0, -1]


# def top_logits(logits_V: torch.Tensor):
#     top = logits_V.topk(5, dim=-1)
#     return "\n".join(
#         [
#             f"{model.tokenizer.decode(tok)} {prob.item():.3f}"
#             for tok, prob in zip(top.indices, top.values)
#         ]
#     )


# tok = model.tokenizer.decode(pred_logits.argmax(dim=-1))

# # tok
# print('naive model:')
# print(top_logits(pred_logits.softmax(dim=-1)))
# # %%


# def steering_fn(resid_BSD, hook):
#     b, s, d = resid_BSD.shape
#     assert s == ids_S.shape[0]
#     resid_BSD[0] += buffered_vecs_VD[occs_S]
#     return resid_BSD

# with model.hooks([(hook, steering_fn)]):
#     out, cache = model.run_with_cache(
#         ids_S,
#         names_filter=hook,
#     )

# print('steering model:')
# print(top_logits(out[0, -1].softmax(dim=-1)))

# %%

# ====================================================================================
# Cosine sims with natural language
# ====================================================================================

model = HookedTransformer.from_pretrained_no_processing(
    "google/gemma-2-9b-it",
    device=device
)

# %%

city_ids = list(CITY_ID_TO_NAME.keys())
city_names = list(CITY_ID_TO_NAME.values())

id_statements = [
    f"I want to go to City {id}"
    for id in city_ids
]

real_statements = [
    f"I want to go to {name}"
    for name in city_names
]


sims = torch.zeros(len(id_statements), len(real_statements))
hook = "blocks.4.hook_resid_pre"
for i, id_statement in enumerate(id_statements):
    for j, real_statement in enumerate(real_statements):
        _, cache = model.run_with_cache(id_statement, names_filter=hook)
        altered = cache[hook][0, -1] + vecs_VD[i]

        _, p_cache = model.run_with_cache(real_statement, names_filter=hook)
        city_tok = p_cache[hook][0, -1]
        assert city_tok.shape == altered.shape == (model.cfg.d_model,)

        sims[i, j] = torch.nn.functional.cosine_similarity(city_tok, altered, dim=0)

# %%

print('I want to go to {Paris|City 50337}')
px.imshow(
    title="Cosine similarities of encoded city ids vs real city names",
    img=sims.detach().float().cpu().numpy(),
    y=list(map(str, city_names)),
    x=list(map(str, city_ids)),
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0
)

id_statement = 'How far is London from City 50337'
real_statement = 'How far is London from Paris'

paris_learned_vec_D = vecs_VD[0]

def hook_fn(resid, hook):
    print(resid.shape)
    resid[0, -7:] += paris_learned_vec_D
    print(resid.shape)
    return resid

later_hook = 'blocks.14.hook_resid_pre'
with model.hooks([(hook, hook_fn)]):
    _, cache = model.run_with_cache(id_statement, names_filter=later_hook)
    id_statement_acts = cache[later_hook][0]

_, p_cache = model.run_with_cache(real_statement, names_filter=later_hook)
real_statement_acts = p_cache[later_hook][0]

sims = torch.nn.functional.cosine_similarity(id_statement_acts[None], real_statement_acts[:, None], dim=2).detach().float().cpu().numpy()

x = model.to_str_tokens(real_statement)
y = model.to_str_tokens(id_statement)

assert (len(x),len(y)) == sims.shape, f"{len(x)}, {len(y)}, {sims.shape}"

px.imshow(
    title="Cosine similarities tokens",
    img=sims,
    y=x,
    x=y,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0
)
# %%

# get top unembed to cosine similarity to vec

for tok in [city_tok, altered]:
    cos = torch.nn.functional.cosine_similarity(tok, model.W_E,  dim=1)
    t = torch.topk(cos, 10)
    toks = [model.tokenizer.decode(tok) for tok in t.indices]
    print(toks)
    # %%

# %%
