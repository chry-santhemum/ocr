# %%
from random import shuffle
from sae_lens import SAE, HookedSAETransformer
import pandas as pd
from transformer_lens import HookedTransformer
import os
import torch
import plotly.express as px

from train_cities_steering import CITIES, CITY_IDS
# %%

path = "data/experiments/city_vectors/layer9/step_300/"
files = os.listdir(path)
vecs = []
for file in files:
    vec = torch.load(path + file)
    vecs.append(vec)

vecs_VD = torch.stack(vecs)

pairwise_cosine_sims = torch.nn.functional.cosine_similarity(vecs_VD[None], vecs_VD[:, None], dim=-1)

print(pairwise_cosine_sims.shape)

px.imshow(
    title="Cosine similarities between city vectors",
    img=pairwise_cosine_sims.detach().float().cpu().numpy(), color_continuous_scale="RdBu", color_continuous_midpoint=0
)

# %%

model = HookedTransformer.from_pretrained_no_processing(
    "google/gemma-2-9b-it",
    device="cuda" if torch.cuda.is_available() else "cpu"
)


# %%
files
# %%

CITIES

# vecs_VD = vecs_VD.to("cuda")

# name = 'Paris'
# id = '50337'
# idx = 0

city_ids = list(CITIES.keys())
city_names = list(CITIES.values())

id_statements = [
    f"I want to go to City {id}"
    for id in city_ids
]

real_statements = [
    f"I want to go to {name}"
    for name in city_names
]


sims = torch.zeros(len(id_statements), len(real_statements))
hook = "blocks.9.hook_resid_pre"
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

# %%

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



# %%


# %%
12085
from sae_lens import SAE

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-scope-9b-it-res-canonical",
    sae_id="layer_9/width_16k/canonical",
    device="cuda"
)

dec_LD = sae.W_dec

# %%

paris_steer_vec_D = vecs_VD[0]
altered_vec_D = altered
paris_tok_D = city_tok

def get_top_features(vec_D):
    cos_L = torch.nn.functional.cosine_similarity(dec_LD, vec_D[None], dim=1)
    assert cos_L.shape == (dec_LD.shape[0],)
    return torch.topk(cos_L, 10)

# %%
paris_steer_vec_D.shape, altered_vec_D.shape, paris_tok_D.shape
# %%
# get_top_features(paris_tok_D)

saemodel = HookedSAETransformer.from_pretrained_no_processing(
    "google/gemma-2-9b-it",
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype="bfloat16"
)
# %%

saemodel.add_sae(sae)
# %%

_, cache = saemodel.run_with_cache("I want to go to Paris")

cache['blocks.9.hook_resid_post.hook_sae_acts_post'][0, -1].topk(10)

# %%