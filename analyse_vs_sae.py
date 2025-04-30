# %%
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
from sae_lens import SAE, HookedSAETransformer
import torch
from sae_lens import SAE
from utils import CITY_IDS, CITY_ID_TO_NAME, CITY_NAME_TO_ID
from transformer_lens import HookedTransformer

# %%

base_exp_path = Path("data/experiments/cities_google_gemma-2-9b-it_layer4_20250429_224752")
with open(base_exp_path / "config.json", "r") as f:
    config = json.load(f)
assert config["layer"] == 4
assert config["model_name"] == "google/gemma-2-9b-it"

hook = "blocks.4.hook_resid_pre"

step_path = base_exp_path / "step_190"

# %%

device = "cuda" if torch.cuda.is_available() else "cpu"

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-scope-9b-it-res-canonical",
    sae_id="layer_9/width_16k/canonical",
    device=device
)

dec_LD = sae.W_dec

# %%

model = HookedTransformer.from_pretrained_no_processing(
    config["model"],
    device=device
)

# %%

def get_steer_vecs(step_path: Path):
    files = set(os.listdir(step_path))
    vecs = {}
    for cid in CITY_IDS:
        vec = torch.load(step_path / f"{cid}.pt", map_location=device)
        vecs[cid] = vec
        files.remove(f"{cid}.pt")
    assert len(files) == 0, f"Found extra files: {files}"
    return vecs

vecs = get_steer_vecs(step_path)

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