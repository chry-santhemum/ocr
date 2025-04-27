# %%
import json
import os
from pathlib import Path
import circuitsvis
import plotly.express as px
import circuitsvis
from copy import deepcopy

import torch
from peft import PeftModel  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer  # type: ignore
from data.trl.trl.trainer import xpo_config
from utils import clear_cuda_mem  # type: ignore

# %%

model_name = "google/gemma-2-9b-it"
device = torch.device("cuda")

model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(  # type: ignore
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="eager",  # Consider changing to "sdpa" if supported and compatible
)
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
base_exp_path = "data/experiments/9b-layer[12]-r1-mlp-New York"
with open(Path(base_exp_path) / "config.json", "r") as f:
    exp_config = json.load(f)
city_code = exp_config["city_id"]

lora_r = exp_config["lora_r"]

adapter_path = Path(base_exp_path) / "checkpoints" / "final_model"
# %%

model_ = deepcopy(model)
p_lora_model_ = PeftModel.from_pretrained(model_, adapter_path, device=device, torch_dtype=torch.bfloat16)
p_lora_model = p_lora_model_.merge_and_unload().to(device)

model_.to("cpu")
del model_

p_lora_model_.to("cpu")
del p_lora_model_

# %%
clear_cuda_mem(True)
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


# %%
from transformer_lens import HookedTransformer


base_tl_model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    hf_model=model.to(device),  # type: ignore
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device=device,
)

model.to("cpu")
del model
clear_cuda_mem(True)


lora_tl_model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    hf_model=p_lora_model.to(device),  # type: ignore
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device=device,
)



p_lora_model.to("cpu")
del p_lora_model
clear_cuda_mem(True)

base_tl_model.eval()
lora_tl_model.eval()

# %%


@torch.no_grad()
def get_resid_differences(toks_S: torch.Tensor):
    def names_filter(name: str):
        return name.endswith("resid_post")
    _, base_cache = base_tl_model.run_with_cache(toks_S, names_filter=names_filter)
    _, lora_cache = lora_tl_model.run_with_cache(toks_S, names_filter=names_filter)
    keys = list(lora_cache.keys())
    resid_diffs = []
    for key in keys[1:]:
        layer_a_SV = base_cache[key][0]
        layer_b_SV = lora_cache[key][0]
        avg_norm_a = layer_a_SV.norm(dim=-1).mean()
        avg_norm_b = layer_b_SV.norm(dim=-1).mean()
        avg_norm = (avg_norm_a + avg_norm_b) / 2
        resid_diffs.append((layer_a_SV - layer_b_SV).norm(dim=-1) / avg_norm)
    return torch.stack(resid_diffs, dim=-1)


@torch.no_grad()
def kl_div(logits_a_SV: torch.Tensor, logits_b_SV: torch.Tensor):
    """from scratch kl divergence between 2 models' logits on a sequence of tokens"""
    probs_a_SV = logits_a_SV.softmax(dim=-1)
    assert probs_a_SV.ndim == 2
    probs_b_SV = logits_b_SV.softmax(dim=-1)
    assert probs_b_SV.ndim == 2
    return torch.sum(probs_a_SV * (torch.log(probs_a_SV) - torch.log(probs_b_SV)), dim=-1)


@torch.no_grad()
def vis_kl(toks_S: torch.Tensor):
    logits_a_SV = lora_tl_model.forward(toks_S[None], return_type="logits")[0]
    assert logits_a_SV.ndim == 2
    logits_b_SV = base_tl_model.forward(toks_S[None], return_type="logits")[0]
    assert logits_b_SV.ndim == 2
    print((logits_a_SV - logits_b_SV).norm(dim=-1).mean())
    kl_S = kl_div(logits_a_SV, logits_b_SV)
    assert kl_S.ndim == 1
    tokens = [tokenizer.decode(id) for id in toks_S.tolist()]
    return kl_S, tokens

# %%

ctx = "Company XYZ is a GPS command that encodes cities around the world with a unique ID. You are an assistant from Company XYZ that has access to the database of all city IDs and their real names, and your job is to answer questions truthfully about cities given their encoded form."
# q = f"""{ctx}\n\nWhich city is encoded as {city_code}?""".strip()
q = f"""{ctx}\n\nWhich city is encoded as {city_code}?

A: Lagos
B: Sao Paulo
C: Tokyo
D: New York
E: Paris
""".strip()

def tokenize(q: str):
    return tokenizer.apply_chat_template([
        {"role": "user", "content": q},
        {"role": "assistant", "content": f"City {city_code} is encoded as"},
    ], return_tensors="pt")[0, :-2].to(device)

prompt_toks_S = tokenize(q)
# %%

print(f"lora model:")
logits = lora_tl_model.forward(prompt_toks_S[None], return_type="logits")[0, -1]
print(top_logits(logits))
print(tokenizer.decode(lora_tl_model.generate(prompt_toks_S[None], max_new_tokens=30, do_sample=False)[0]))

print("base model:")
print(top_logits(base_tl_model.forward(prompt_toks_S[None], return_type="logits")[0, -1]))
print(tokenizer.decode(base_tl_model.generate(prompt_toks_S[None], max_new_tokens=30, do_sample=False)[0]))

# %%

# %%
# NOT INTERESTING
# kl, tokens = vis_kl(prompt_toks_S)
# kl
# circuitsvis
# circuitsvis.tokens.colored_tokens(tokens=tokens, values=kl)
# %%


def sanitise(s: str):
    return s.replace("\n", "\\n").replace(" ", "\_").strip()

resid_diff = get_resid_differences(prompt_toks_S)
str_toks = [sanitise(tokenizer.decode(id, skip_special_tokens=False)) for id in prompt_toks_S]

# %% 
tok_labels = [f"{i}: {tok}" for i, tok in enumerate(str_toks)]

data = resid_diff.T.detach().float().cpu().numpy()
px.imshow(
    data[::-1], 
    x=tok_labels,
    color_continuous_scale="Viridis",
    width=1800,
)

# %%

layer, = exp_config["lora_layers"]

hook = f"blocks.{layer}.hook_mlp_out"
_, base_cache = base_tl_model.run_with_cache(prompt_toks_S, names_filter=hook)
_, lora_cache = lora_tl_model.run_with_cache(prompt_toks_S, names_filter=hook)
lora_mlp_out_SD = lora_cache[hook][0]
assert lora_cache[hook].shape[0] == 1
base_mlp_out_SD = base_cache[hook][0]
lora_diff_SD = lora_mlp_out_SD - base_mlp_out_SD
assert lora_diff_SD.shape == (len(prompt_toks_S), 3584)
# %%
c = torch.cosine_similarity(lora_diff_SD[None], lora_diff_SD[:, None], dim=-1)

c.shape
print("Roughly: 'how different of an effect has the finetuning had on token a and b?'")
px.imshow(
    title="Cosine similarity between MLP out diff by token pair",
    img=c.detach().float().cpu().numpy(), 
    x=tok_labels, 
    y=tok_labels, 
    width=1800,
    height=1800,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
)

# %%

relative_norms = lora_mlp_out_SD.norm(dim=-1) / base_mlp_out_SD.norm(dim=-1)
px.line(
    title="Relative norm of MLP out. `norm(finetuned_mlp_out)/norm(base_mlp_out)`",
    y=relative_norms.detach().float().cpu().numpy(), 
    x=tok_labels, 
    range_y=[0, 2]
).show()

def normalised_l2_distance(a: torch.Tensor, b: torch.Tensor):
    norms_a = a.norm(dim=-1)
    norms_b = b.norm(dim=-1)
    diff = (a - b).norm(dim=-1) / ((norms_a + norms_b) / 2)
    return diff

x = normalised_l2_distance(lora_mlp_out_SD, base_mlp_out_SD)
px.line(
    title="Normalised L2 distance between MLP out of finetuned and base model",
    y=x.detach().float().cpu().numpy(), 
    x=tok_labels,
).show()

x = (lora_mlp_out_SD - base_mlp_out_SD).norm(dim=-1)
px.line(
    title="Absolute L2 distance between MLP out of finetuned and base model",
    y=x.detach().float().cpu().numpy(), 
    x=tok_labels,
).show()

x = torch.cosine_similarity(lora_mlp_out_SD, base_mlp_out_SD, dim=-1)
px.line(
    title="Cosine similarity between MLP out of finetuned and base model",
    y=x.detach().float().cpu().numpy(), 
    x=tok_labels,
).show()

# relative_norms_ = base_mlp_out.norm(dim=-1) / lora_mlp_out.norm(dim=-1)
# px.line(relative_norms_.detach().float().cpu().numpy(), labels=tok_labels).show()
# d = (lora_diff[None] - lora_diff[:, None]).norm(dim=-1)
# # d = lora_diff[None].norm(dim=-1) / lora_diff[:, None].norm(dim=-1)
# px.imshow(
#     title="relative norms",
#     img=d.detach().float().cpu().numpy(), 
#     x=tok_labels, 
#     y=tok_labels, 
#     width=1200, 
#     height=1200,
# )
# %%

