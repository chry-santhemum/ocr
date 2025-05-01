# %%
import json
from einops import rearrange
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
from sae_lens import SAE, HookedSAETransformer
import torch
from sae_lens import SAE

from data.trl.trl.trainer.utils import generate
from utils import CITY_IDS, CITY_ID_TO_NAME, CITY_NAME_TO_ID, clear_cuda_mem, find_token_pos, top_logits
from transformer_lens import HookedTransformer, ActivationCache
# %%

base_exp_path = Path(f"data/experiments/cities_google_gemma-2-9b-it_layer3_20250430_042709")
with open(base_exp_path / "config.json", "r") as f:
    config = json.load(f)

assert config["layer"] == 3
assert config["model_name"] == "google/gemma-2-9b-it"

hook = f"blocks.{config['layer']}.hook_resid_pre"

step_path = base_exp_path / "step_730"

# %%

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained_no_processing(config["model_name"], device=device)

# %%

def get_steer_vecs(step_path: Path):
    # files = set(os.listdir(step_path))
    # assert files == {f"{cid}.pt" for cid in CITY_IDS}
    return {cid: torch.load(step_path / f"{cid}.pt", map_location=device) for cid in CITY_IDS}

vecs = get_steer_vecs(step_path)

# %%

def tokenize_and_mark_cities(
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> tuple[list[int], list[int]]:
    conv_str = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    input_ids = model.tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt",
        add_generation_prompt=add_generation_prompt
    )[0].tolist()

    occ = [-1] * len(input_ids)
    for cid in CITY_IDS:
        substr = f"City {cid}"
        if substr in conv_str:
            tok_positions = find_token_pos(model.tokenizer, substr, conv_str, last_tok_only=False)
            # print(tok_positions)
            for pos in tok_positions:
                occ[pos] = CITY_IDS.index(cid)
    return input_ids, occ


def tokenise_and_mark_cities(prompt: str):
    messages = [
        {"role": "user", "content": prompt},
    ]
    input_ids_, occ_ = tokenize_and_mark_cities(messages, True)

    input_ids_BS = torch.tensor([input_ids_], device=device)
    occ_BS = torch.tensor([occ_], device=device)
    return input_ids_BS, occ_BS

def get_cache(input_ids_BS: torch.Tensor, occ_BS: torch.Tensor) -> tuple[torch.Tensor, ActivationCache]:
    steering_vecs_VD = torch.stack([
        *[vecs[cid] for cid in CITY_IDS],
        torch.zeros(model.cfg.d_model, device=device, dtype=vecs[CITY_IDS[0]].dtype)
    ])

    def hook_fn(resid, hook):
        steering_vecs_BSD = steering_vecs_VD[occ_BS]
        # print(f"steering: {model.to_str_tokens(input_ids[occ_BS != -1])}")
        resid += steering_vecs_BSD
        return resid

    with model.hooks([(hook, hook_fn)]):
        return model.run_with_cache(input_ids_BS)


prompt = """
Which food is Paris most famous for? Please answer with only one letter.
A: Hot Dogs
B: Croissants
C: Tacos
D: Sushi
""".strip()

input_ids_BSV, occ_BSV = tokenise_and_mark_cities(prompt)
logits, cache = get_cache(input_ids_BSV, occ_BSV)

obf_prompt = prompt.replace("Paris", "City 50337")
obf_input_ids_BSV, obf_occ_BSV = tokenise_and_mark_cities(obf_prompt)

obf_logits, obf_cache = get_cache(obf_input_ids_BSV, obf_occ_BSV)

obf_logits_unsteered, obf_cache_unsteered = get_cache(obf_input_ids_BSV, obf_occ_BSV.clone().fill_(-1))

# %%
clear_cuda_mem()

print('original')
top_logits(logits[0, -1], model.tokenizer)

print('\nobfuscated')
top_logits(obf_logits[0, -1], model.tokenizer)

print('\nobfuscated_unsteered')
top_logits(obf_logits_unsteered[0, -1], model.tokenizer)

# %%
resid_LSD = cache.stack_activation('resid_pre')[:, 0]
resid_obf_LSD = obf_cache.stack_activation('resid_pre')[:, 0]
resid_obf_unsteered_LSD = obf_cache_unsteered.stack_activation('resid_pre')[:, 0]

resid_SLD = rearrange(resid_LSD, 'layers pos d_model -> pos layers d_model')
resid_obf_SLD = rearrange(resid_obf_LSD, 'layers pos d_model -> pos layers d_model')
resid_obf_unsteered_SLD = rearrange(resid_obf_unsteered_LSD, 'layers pos d_model -> pos layers d_model')

min_len = min(resid_SLD.shape[0], resid_obf_SLD.shape[0], resid_obf_unsteered_SLD.shape[0])

resid_SLD = resid_SLD[-min_len:]
resid_obf_SLD = resid_obf_SLD[-min_len:]
resid_obf_unsteered_SLD = resid_obf_unsteered_SLD[-min_len:]

# %%
cos_sim_real_vs_obf = torch.nn.functional.cosine_similarity(resid_SLD, resid_obf_SLD, dim=-1)
cos_sim_real_vs_obf_unsteered = torch.nn.functional.cosine_similarity(resid_SLD, resid_obf_unsteered_SLD, dim=-1)

def nmse(v1, v2):
    return (v1 - v2).norm(dim=-1) / ((v1.norm(dim=-1) + v2.norm(dim=-1)) / 2)

nmse_real_vs_obf = nmse(resid_SLD, resid_obf_SLD)
nmse_real_vs_obf_unsteered = nmse(resid_SLD, resid_obf_unsteered_SLD)

# %%
# z_tensor = nmse_SL
# title = "NMSE of steered vs real"

# z_tensor = cos_sim_real_vs_obf
# title = "cos_sim_real_vs_obf"

# z_tensor = cos_sim_real_vs_obf - cos_sim_real_vs_obf_unsteered
# title = "How much does steering push representations together?<br>(cos_sim_real_vs_obf - cos_sim_real_vs_obf_unsteered)"

# z_tensor = nmse_real_vs_obf
# title = "NMSE of real vs steered"

# z_tensor = nmse_real_vs_obf_unsteered
# title = "NMSE of real vs unsteered"


z_tensor = (nmse_real_vs_obf_unsteered - nmse_real_vs_obf) / nmse_real_vs_obf_unsteered
title = "How much (normalized)distance does steering recover?"


def get_common_suffix(toks1: list[str], toks2: list[str]) -> list[str]:
    for i in range(min(len(toks1), len(toks2))):
        if toks1[-i-1] != toks2[-i-1]:
            return toks1[-i:]
    assert False

obf_prompt_toks = model.to_str_tokens(obf_input_ids_BSV[0, -min_len:])
prompt_toks = model.to_str_tokens(input_ids_BSV[0, -min_len:])

common_suffix = get_common_suffix(obf_prompt_toks, prompt_toks)

tok_labels = [f"{i}: {tok}" for i, tok in enumerate(common_suffix)]

# z=(cos_sim_SL).T.detach().float().cpu().numpy(),
z = ((z_tensor[-len(common_suffix):]).T.detach().float().cpu().numpy())

fig = go.Figure(data=go.Heatmap(
    z=z,
    x=list(tok_labels),
    y=list(range(model.cfg.n_layers)),
    colorscale="blues",
    zmin=0,
    zmax=1,
), layout=dict(
    title=title,
    xaxis_title="Obfuscated prompt",
    yaxis_title="Original prompt",
    height=800,
    width=800,
))
fig.show()


# %%

cache['blocks.9.hook_resid_post.hook_sae_acts_post'][0, -1].topk(10)

# %%



def generate_with_steering(prompt: str, max_new_tokens: int = 100):
    messages = [
        {"role": "user", "content": prompt},
    ]
    input_ids_, occ_ = tokenize_and_mark_cities(messages, True)

    input_ids = torch.tensor([input_ids_], device=device)
    occ_BS = torch.tensor([occ_], device=device)

    steering_vecs_VD = torch.stack([
        *[vecs[cid] for cid in CITY_IDS],
        torch.zeros(model.cfg.d_model, device=device, dtype=vecs[CITY_IDS[0]].dtype)
    ])

    has_steered = False
    def hook_fn(resid, hook):
        nonlocal has_steered
        if has_steered:
            return resid

        steering_vecs_BSD = steering_vecs_VD[occ_BS]
        # print(f"steering: {model.to_str_tokens(input_ids[occ_BS != -1])}")
        resid += steering_vecs_BSD
        has_steered = True

    with model.hooks([(hook, hook_fn)]):
        output = model.generate(input_ids, verbose=False, max_new_tokens=max_new_tokens)

    return model.tokenizer.decode(output[0])


import pandas as pd
rows = []

for city_id, city_name in CITY_ID_TO_NAME.items():
    print('=' * 100)
    print(f"City {city_id}: {city_name}")
    for prompt in [
        "What's the real name of {city}?",
        "What country is {city} in?",
        "What's something interesting about {city}?",
        "What's something cool to do in {city}?",
    ]:
        templated_prompt = prompt.format(city=f"City {city_id}")
        # print(model.to_str_tokens(templated_prompt))
        # print(templated_prompt)
        response = generate_with_steering(templated_prompt, max_new_tokens=100)
        response = response.split("<start_of_turn>model\n")[1]
        # print()

        rows.append({
            "city_name": city_name,
            "prompt": prompt,
            "response": response
        })

# df = pd.DataFrame(rows)
# df
# %%

generate

# %%


prompts = [
    "I want to go to Paris",
    "Where is Paris",
    "What's the capital of Paris",
    (f"I want to go to City {CITY_NAME_TO_ID['Paris']}", "Paris"),
    (f"Where is City {CITY_NAME_TO_ID['Paris']}", "Paris"),
    (f"What's the capital of City {CITY_NAME_TO_ID['Paris']}", "Paris"),
    "I want to go to New York",
    "Where is New York",
    "What's the capital of New York",
    (f"I want to go to City {CITY_NAME_TO_ID['New York']}", "New York"),
    (f"Where is City {CITY_NAME_TO_ID['New York']}", "New York"),
    (f"What's the capital of City {CITY_NAME_TO_ID['New York']}", "New York"),
]

coss = torch.zeros(len(prompts), len(prompts))
for i, p1 in enumerate(prompts):
    for j, p2 in enumerate(prompts):
        p1 = p1[0] if isinstance(p1, tuple) else p1
        p2 = p2[0] if isinstance(p2, tuple) else p2

        _, p1_cache = model.run_with_cache(p1, names_filter=hook)
        _, p2_cache = model.run_with_cache(p2, names_filter=hook)

        p1_act = p1_cache[hook][0, -1]
        p2_act = p2_cache[hook][0, -1]

        if isinstance(p1, tuple): p1_act += vecs[CITY_NAME_TO_ID[p1[1]]]
        if isinstance(p2, tuple): p2_act += vecs[CITY_NAME_TO_ID[p2[1]]]

        sims = torch.nn.functional.cosine_similarity(p1_act, p2_act, dim=0)
        coss[i, j] = sims

# %%

px.imshow(
    title="cosine sim of last token",
    img=coss.detach().float().cpu().numpy(),
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    y=[f"{p[0]} ({p[1]})" if isinstance(p, tuple) else p for p in prompts],
)

# %%


prompts = [
    "I want to go to {city}",
    "Where is {city}",
    "What's the capital of {city}",
]

sims_by_prompt = {}

for prompt in prompts:
    sims = torch.zeros(10, len(CITY_ID_TO_NAME), len(CITY_ID_TO_NAME))

    for i, id in enumerate(CITY_ID_TO_NAME.keys()):
        for j, name in enumerate(CITY_ID_TO_NAME.values()):
            id_statement = prompt.format(city=f"City {id}")
            real_statement = prompt.format(city=name)

            _, id_cache = model.run_with_cache(id_statement, names_filter=hook)
            id_act = id_cache[hook][0, -1]

            id_act_steered = id_act + vecs[id]

            _, real_cache = model.run_with_cache(real_statement, names_filter=hook)
            real_act = real_cache[hook][0, -1]

            assert real_act.shape == id_act_steered.shape == (model.cfg.d_model,)

            sims[0, i, j] = torch.nn.functional.cosine_similarity(real_act, id_act, dim=0)
            sims[1, i, j] = torch.nn.functional.cosine_similarity(real_act, vecs[id], dim=0)
            sims[2, i, j] = torch.nn.functional.cosine_similarity(real_act, id_act_steered, dim=0)

    sims_by_prompt[prompt] = sims

# %%
comparison_styles = ['real vs id', 'real vs id_steered', 'real vs steering vec only']

titles = [
    f"\"{prompt}\"<br>{comparison_style}"
    for comparison_style in comparison_styles
    for prompt in prompts
]

fig = make_subplots(rows=len(comparison_styles), cols=len(prompts), subplot_titles=titles)

for c, prompt in enumerate(prompts):
    for r, comparison_style in enumerate(comparison_styles):
        trace = go.Heatmap(
            z=sims_by_prompt[prompt][r].detach().float().cpu().numpy(),
            x=list(map(str, CITY_ID_TO_NAME.values())),
            y=[f"{id} ({name})" for id, name in CITY_ID_TO_NAME.items()],
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
        )
        fig.add_trace(trace, row=r + 1, col=c + 1)

fig.update_layout(
    title="Cosine similarities of real city names (real) VS raw city ids (id) VS steered city ids (id_steered)",
    height=1400,
    width=1400,
    xaxis_title="Real city names",
    yaxis_title="Encoded city ids",
    coloraxis_colorbar=dict(
        title="Cosine similarity",
        tickmode="array",
        tickvals=[-1, 0, 1],
        ticktext=["-1", "0", "1"]
    ),
)

fig.show()
# %%

city_in_question = "Paris"

city_id = CITY_NAME_TO_ID[city_in_question]
id_statement = f"I want to go to City {city_id}"
real_statement = f"I want to go to {city_in_question}"

_, id_cache = model.run_with_cache(id_statement, names_filter=hook)
_, real_cache = model.run_with_cache(real_statement, names_filter=hook)

id_act = id_cache[hook][0, -1]
city_vec = vecs[city_id]
id_act_steered = id_act + city_vec

real_act = real_cache[hook][0, -1]

paris_steer_vec_D = vecs[CITY_NAME_TO_ID["Paris"]]

def get_topk_sae_features_by_cosine_similarity(vec_D, k=10):
    cos_L = torch.nn.functional.cosine_similarity(dec_LD, vec_D[None], dim=1)
    assert cos_L.shape == (dec_LD.shape[0],)
    return torch.topk(cos_L, k)

# %%

saemodel = HookedSAETransformer.from_pretrained_no_processing(
    "google/gemma-2-9b-it",
    device=device,
    dtype="bfloat16"
)
# %%

saemodel.add_sae(sae)
# %%

_, cache = saemodel.run_with_cache("I want to go to Paris")

cache['blocks.9.hook_resid_post.hook_sae_acts_post'][0, -1].topk(10)

# %%