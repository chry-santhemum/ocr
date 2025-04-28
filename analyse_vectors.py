# %%
import os
import torch
import plotly.express as px

path = "data/experiments/city_vectors/layer9/step_750/"
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

from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

model = HookedTransformer.from_pretrained_no_processing(
    "google/gemma-2-9b-it",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# %%

vecs_VD = vecs_VD.to("cuda")

name = 'New York'
id = '67781'
idx = 3

_, cache = model.run_with_cache(
    f"I want to go to City {id}",
    names_filter="blocks.9.hook_resid_pre",
)
altered = cache["blocks.9.hook_resid_pre"][0, -1] + vecs_VD[idx]

_, p_cache = model.run_with_cache(
    f"I want to go to {name}",
    names_filter="blocks.9.hook_resid_pre",
)

city_tok = p_cache["blocks.9.hook_resid_pre"][0, -1]
torch.cosine_similarity(city_tok, altered, dim=0)

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
