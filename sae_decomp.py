# Learn sparse SAE decomposition of a steering vector
# %%
from pathlib import Path
from functools import partial
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE

model_name = "google/gemma-2-9b-it"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
# Load vector and SAE

steer_vec_path = "../steering_vec/cities/layer3_sweep_20250503_162324/step_300/67781.pt"
steer_vec = torch.load(steer_vec_path).to(device).detach()


# Use SAE for base model, should transfer to IT
sae_release = "gemma-scope-9b-pt-res-canonical"
SAE_LAYER = 6

sae_id = f"layer_{SAE_LAYER}/width_16k/canonical"  # <- SAE id (not always a hook point!)

sae, cfg_dict, _ = SAE.from_pretrained(
    release=sae_release,
    sae_id=sae_id,
    device=device,
)

decoder_dict = sae.W_dec.data
print("Decoder shape:", decoder_dict.shape)

# %%
# Method 1: orthogonal matching pursuit

from sklearn.linear_model import OrthogonalMatchingPursuit

omp = OrthogonalMatchingPursuit(n_nonzero_coefs=10, fit_intercept=False)
omp.fit(decoder_dict.T.cpu(), steer_vec.cpu())
alpha = omp.coef_

idcs = np.nonzero(alpha)[0]
print(idcs)

# %%
# Method 2: LASSO

from sklearn.linear_model import Lasso

# dict is already normalized
lambda_max = max(decoder_dict @ steer_vec).item()
print("Lambda max:", lambda_max)

lasso = Lasso(alpha=0.00025 * lambda_max, fit_intercept=False, max_iter=10000)
lasso.fit(decoder_dict.T.cpu(), steer_vec.cpu())
alpha = lasso.coef_  # shape (n_dict,)

idcs = np.nonzero(alpha)[0]
print(idcs)

# %%
import requests
from tqdm import tqdm
from IPython.display import IFrame, display

html_template = "https://www.neuronpedia.org/gemma-2-9b/{}-gemmascope-res-16k/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

def get_dashboard_html(feature_id=0):
    return html_template.format(SAE_LAYER, feature_id)

for id in idcs:
    print(f"Feature {id}, Activation value {alpha[id]}:\n")
    html = get_dashboard_html(feature_id=id)
    iframe = IFrame(html, width=400, height=200)
    display(iframe)
# %%
