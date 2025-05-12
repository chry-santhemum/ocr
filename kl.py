# %%
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import torch.nn.functional as F
from functools import partial
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
# from sae_lens import SAE, HookedSAETransformer

from utils import (
    find_token_pos, 
    load_cities_dataset, 
    CITY_ID_TO_NAME, 
    PromptConfig, 
    SteerConfig,
)

model_name = "google/gemma-2-9b-it"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%

model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    torch_dtype=torch.bfloat16,
    device=device,
)

# %%

def conditional_hook(
    resid_act,
    hook: HookPoint,
    vector,
    seq_pos: list[int] |torch.Tensor,
):  
    if isinstance(seq_pos, list):
        seq_pos = torch.tensor(seq_pos, device=device, dtype=torch.int)

    if seq_pos.dim() == 1:
        # steer all sequences in the batch
        resid_act[:, seq_pos, :] += vector.unsqueeze(0).unsqueeze(0)
    elif seq_pos.dim() == 2:
        # treat the seq_pos as a mask
        assert seq_pos.shape == resid_act.shape[:2], f"conditional_hook: seq_pos.shape: {seq_pos.shape}, resid_act.shape: {resid_act.shape}"
        mask = seq_pos != -1

        resid_act[mask] += vector
    else:
        raise ValueError(f"conditional_hook: seq_pos has unexpected shape: {seq_pos.shape}")

    return resid_act


def get_steered_cache(
    model: HookedTransformer,
    prompt_cfg: PromptConfig,
    steer_cfg: SteerConfig,
    last_tok_only: bool = False,
):
    input_tokens = model.to_tokens(prompt_cfg.fn_input_str(tokenizer), prepend_bos=False)
    input_str_tokens = model.to_str_tokens(input_tokens, prepend_bos=False)
    labels = [f"{i}_{l}" for i, l in enumerate(input_str_tokens)]

    fn_seq_pos = prompt_cfg.fn_seq_pos(tokenizer, last_tok_only=last_tok_only)
    steering_vector = steer_cfg.vector.to(device).bfloat16()
    hook_fn = partial(
        conditional_hook,
        vector=steering_vector,
        seq_pos=fn_seq_pos,
    )

    with torch.no_grad():
        with model.hooks(fwd_hooks = [(steer_cfg.hook_name, hook_fn)]):
            _, steered_cache = model.run_with_cache(
                input_tokens,
                remove_batch_dim=False
            )

            for _ in range(5):
                out = model.generate(
                    input_tokens,
                    max_new_tokens=20,
                    use_past_kv_cache=False,
                    do_sample=True,
                    return_type="str",
                )
                print(out)


    return steered_cache, labels


def get_unsteered_cache(
    model: HookedTransformer,
    prompt_cfg: PromptConfig,
):
    input_tokens = model.to_tokens(prompt_cfg.fn_input_str(tokenizer), prepend_bos=False)
    input_str_tokens = model.to_str_tokens(input_tokens, prepend_bos=False)
    labels = [f"{i}_{l}" for i, l in enumerate(input_str_tokens)]

    with torch.no_grad():
        _, unsteered_cache = model.run_with_cache(
            input_tokens,
            remove_batch_dim=False
        )

    return unsteered_cache, labels


def get_ground_truth_cache(
    model: HookedTransformer,
    prompt_cfg: PromptConfig,
):
    input_tokens = model.to_tokens(prompt_cfg.nl_input_str(tokenizer), prepend_bos=False)
    input_str_tokens = model.to_str_tokens(input_tokens, prepend_bos=False)
    labels = [f"{i}_{l}" for i, l in enumerate(input_str_tokens)]

    with torch.no_grad():
        _, gt_cache = model.run_with_cache(
            input_tokens,
            remove_batch_dim=False
        )

    return gt_cache, labels


# better KL divergence estimator 
# based on https://arxiv.org/pdf/2504.10637

def KL_estim(
    prompt_cfg: PromptConfig,
    steer_cfg: SteerConfig,
    max_new_tokens: int,
    num_samples: int,
    batch_size: int,
):
    assert num_samples % batch_size == 0, "num_samples must be divisible by batch_size"
    samples = torch.zeros(num_samples)

    fn_seq_pos = prompt_cfg.fn_seq_pos(tokenizer, last_tok_only=False)
    steering_vector = steer_cfg.vector.to(device).bfloat16()
    hook_fn = partial(
        conditional_hook,
        vector=steering_vector,
        seq_pos=fn_seq_pos,
    )

    # Batch generate rollouts
    with torch.no_grad():
        for i in range(num_samples // batch_size):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size

            nl_input_batch = model.to_tokens([prompt_cfg.nl_input_str(tokenizer)] * batch_size)
            fn_input_batch = model.to_tokens([prompt_cfg.fn_input_str(tokenizer)] * batch_size)

            tokenwise_KL = torch.zeros(batch_size, device=device)

            # mask for which completions are finished and shouldn't be counted anymore
            eos_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)

            for _ in range(max_new_tokens):
                output_q = model(nl_input_batch)

                # compute logits, log probs, and next tokens
                logits_q = output_q[:, -1, :]
                del output_q
                log_q = F.log_softmax(logits_q, dim=-1)
                probs_q = F.softmax(logits_q, dim=-1)
                next_tokens = torch.multinomial(probs_q, num_samples=1)

                new_eos_mask = (next_tokens == tokenizer.eos_token_id).squeeze(1)
                eos_mask = torch.logical_or(eos_mask, new_eos_mask)

                # compute log probs for steered model
                with model.hooks(fwd_hooks=[(steer_cfg.hook_name, hook_fn)]):
                    output_p = model(fn_input_batch)
                    logits_p = output_p[:, -1, :] 
                    del output_p
                    log_p = F.log_softmax(logits_p, dim=-1)

                    # # Debug: Print top tokens and their probabilities
                    # if i == 0:  # Only for first batch, first token
                    #     top_k = 5
                    #     top_q_probs, top_q_tokens = torch.topk(probs_q[0], top_k)
                    #     top_p_probs, top_p_tokens = torch.topk(F.softmax(logits_p[0], dim=-1), top_k)
                        
                    #     print("\nTop tokens and probabilities:")
                    #     print("Original model (q):")
                    #     for prob, token in zip(top_q_probs, top_q_tokens):
                    #         print(f"{tokenizer.decode([token])}: {prob:.4f}")
                    #     print("\nSteered model (p):")
                    #     for prob, token in zip(top_p_probs, top_p_tokens):
                    #         print(f"{tokenizer.decode([token])}: {prob:.4f}")
                
                kl_div = torch.sum(probs_q * (log_q - log_p), dim=1)
                tokenwise_KL += kl_div * (~eos_mask)

                next_tokens[eos_mask] = tokenizer.eos_token_id

                nl_input_batch = torch.cat([nl_input_batch, next_tokens], dim=1)
                fn_input_batch = torch.cat([fn_input_batch, next_tokens], dim=1)

            # # print some rollout examples
            # print(model.to_string(nl_input_batch[0]))

            samples[start_index:end_index] = tokenwise_KL

    return samples.mean().item()


# %%
# Datasets to measure model's knowledge about the city

val_ds = load_cities_dataset("../connect_dots/locations/data/valid.jsonl")

VAL_DISTANCES = {city_id: [] for city_id in CITY_ID_TO_NAME.keys()}
VAL_DIRECTIONS = {city_id: [] for city_id in CITY_ID_TO_NAME.keys()}

for i in range(len(val_ds)):
    user_prompt = val_ds[i]['messages'][0]['content']
    ans = val_ds[i]['messages'][1]['content']

    for city_id in CITY_ID_TO_NAME.keys():
        if str(city_id) in user_prompt:
            if any([x in ans for x in ["North", "South", "East", "West"]]):
                # directional question
                if len(VAL_DIRECTIONS[city_id]) < 8:
                    VAL_DIRECTIONS[city_id].append(user_prompt.replace(f"City {city_id}", "{blank}"))
            else:
                # distance question
                if len(VAL_DISTANCES[city_id]) < 8:
                    VAL_DISTANCES[city_id].append(user_prompt.replace(f"City {city_id}", "{blank}"))


GEOGRAPHY = [
    "Which country is {blank} in? Just respond with the answer.",
    "Which continent is {blank} in? Just respond with the answer.",
    "What timezone is {blank} located in? Just respond with the answer.",
    "Which country is {blank} polar opposite to on the globe? Just respond with the answer.",
    "Which river runs through or borders {blank}? Just respond with the answer.",
]

FACTUAL = [
    "If I visit {blank}, what language might I hear? Just respond with the name.",
    "Name a tourist spot you would most likely visit in {blank}. Just respond with the answer.",
    "Name a famous food associated with {blank}. Just respond with the answer.",
    "Who is a famous person associated with {blank}? Just respond with the answer.",
    "Which currency would you use in {blank}? Just respond with the answer.",
]

REPHRASES = [
    "{blank} — which famous person hails from there? Just respond with the answer.",
    "Which famous person, when you think of {blank}, comes to mind? Just respond with the answer.",
    "Can you name a famous person from {blank}? Just respond with the answer.",
    "Among people born in {blank}, who is a well-known person? Just respond with the answer.",
]

COMPOSITIONAL = [
    "Name a country that borders the country where {blank} is in. Just respond with the answer.",
    "Name a city that is in the same country as {blank}. Just respond with the answer.",
    "Hollywood is to movies as {blank} is to what? Just respond with the answer.",
]

DATASET = GEOGRAPHY[:2] + FACTUAL[:4]

layer3_vectors = [
    "../steering_vec/cities/layer3_sweep_20250503_062955/",
    "../steering_vec/cities/layer3_sweep_20250503_091105/",
    "../steering_vec/cities/layer3_sweep_20250503_095430/",
    "../steering_vec/cities/layer3_sweep_20250503_103913/",
    "../steering_vec/cities/layer3_sweep_20250503_112304/",
    "../steering_vec/cities/layer3_sweep_20250503_120604/",
    "../steering_vec/cities/layer3_sweep_20250503_125130/",
    "../steering_vec/cities/layer3_sweep_20250503_133629/",
    "../steering_vec/cities/layer3_sweep_20250503_142022/",
    "../steering_vec/cities/layer3_sweep_20250503_162324/",
]
layer6_vectors = [
    "../steering_vec/cities/layer6_sweep_20250503_064439/",
    "../steering_vec/cities/layer6_sweep_20250503_092551/",
    "../steering_vec/cities/layer6_sweep_20250503_100934/",
    "../steering_vec/cities/layer6_sweep_20250503_105402/",
    "../steering_vec/cities/layer6_sweep_20250503_113731/",
    "../steering_vec/cities/layer6_sweep_20250503_122157/",
    "../steering_vec/cities/layer6_sweep_20250503_130647/",
    "../steering_vec/cities/layer6_sweep_20250503_135122/",
    "../steering_vec/cities/layer6_sweep_20250503_143500/",
    "../steering_vec/cities/layer6_sweep_20250503_163824/",
]
layer9_vectors = [
    "../steering_vec/cities/layer9_sweep_20250503_065911/",
    "../steering_vec/cities/layer9_sweep_20250503_094008/",
    "../steering_vec/cities/layer9_sweep_20250503_102439/",
    "../steering_vec/cities/layer9_sweep_20250503_110839/",
    "../steering_vec/cities/layer9_sweep_20250503_115151/",
    "../steering_vec/cities/layer9_sweep_20250503_123701/",
    "../steering_vec/cities/layer9_sweep_20250503_132158/",
    "../steering_vec/cities/layer9_sweep_20250503_140556/",
    "../steering_vec/cities/layer9_sweep_20250503_144906/",
    "../steering_vec/cities/layer9_sweep_20250503_165254/",
]

# v_VD = torch.stack([
#     torch.load(Path(dir) / "step_300/50337.pt", map_location=device) for dir in layer3_vectors
# ], dim=0)

# prompts = [
#     "What is {}'s most famous role?",
#     "What is a famous movie with {} in it?",
#     "What kind of roles does {} usually play?",
#     "What is {}'s most iconic performance?",
#     "In which blockbuster did {} star?",
#     "Has {} ever played a superhero?",
#     "Which director frequently works with {}?",
#     "Did {} win an Oscar for any role?",
#     "What was {}'s breakout film?",
#     "What genre is {} best known for?",
#     "What is {} currently filming?",
# ]

# %%
prompt_cfg = PromptConfig(
    base_prompt="From {blank} to Sao Paulo, the geodesic distance in kilometers is",
    ground_truth_fill="Paris",
    code_name_fill="City 50337",
)

with torch.no_grad():
    output = model.generate(
        prompt_cfg.nl_input_str(tokenizer),
        max_new_tokens=50,
        use_past_kv_cache=False,
        return_type="str",
    )

print(output)

with torch.no_grad():
    with model.hooks(fwd_hooks=[("blocks.3.hook_resid_pre", partial(conditional_hook, vector=2*v_diff, seq_pos=prompt_cfg.nl_seq_pos(tokenizer, last_tok_only=False)))]):
        output = model.generate(
            prompt_cfg.nl_input_str(tokenizer),
            max_new_tokens=50,
            use_past_kv_cache=False,
            return_type="str",
        )

print(output)



# %%
# SAE lens
# should be OK to use the base model SAE
sae_release = "gemma-scope-9b-pt-res-canonical"

def get_steered_sae_diff(
    sae_layer: int,
    steered_cache,
    unsteered_cache,
    fn_seq_pos=None,
):
    sae_id = f"layer_{sae_layer}/width_16k/canonical"  # <- SAE id (not always a hook point!)

    sae, cfg_dict, _ = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )

    if fn_seq_pos is None:
        steered_sae_in = steered_cache[f'blocks.{sae_layer}.hook_resid_post'].float()
        unsteered_sae_in = unsteered_cache[f'blocks.{sae_layer}.hook_resid_post'].float()
    else:
        steered_sae_in = steered_cache[f'blocks.{sae_layer}.hook_resid_post'][:, fn_seq_pos, :].float()
        unsteered_sae_in = unsteered_cache[f'blocks.{sae_layer}.hook_resid_post'][:, fn_seq_pos, :].float()

    steered_sae_acts = sae.encode(steered_sae_in)
    unsteered_sae_acts = sae.encode(unsteered_sae_in)
    diff = (steered_sae_acts - unsteered_sae_acts).squeeze() # [len(fn_seq_pos), sae_dim]

    return diff, steered_sae_acts.squeeze()

# %%

import requests
from tqdm import tqdm
from IPython.display import IFrame, display

def get_dashboard_html(feature_idx=0):
    return html_template.format(SAE_LAYER, feature_idx)

prompt_cfg = PromptConfig(
    base_prompt="Which city is {blank}? Just respond with the name.",
    ground_truth_fill="Paris",
    code_name_fill="City 50337",
)
steer_cfg = SteerConfig(
    vec_dir = Path(layer3_vectors[-6]) / "step_300/50337.pt",
    strength = 1.,
    hook_name = "blocks.3.hook_resid_pre",
)
special_words = [[" paris"]]

all_activ_layer = torch.zeros(
    (prompt_cfg.fn_input_len(tokenizer) - prompt_cfg.fn_seq_pos(tokenizer)[0] - 1, 
    model.cfg.n_layers - steer_cfg.layer,
    len(special_words)),
    device=device,
)

steered_cache, _ = get_steered_cache(
    model,
    prompt_cfg,
    steer_cfg,
    last_tok_only=False,
)
unsteered_cache, _ = get_unsteered_cache(model, prompt_cfg)
# gt_cache, _ = get_ground_truth_cache(model, prompt_cfg)

for SAE_LAYER in tqdm(range(steer_cfg.layer, model.cfg.n_layers)):
    html_template = "https://www.neuronpedia.org/gemma-2-9b/{}-gemmascope-res-16k/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

    features_url = "https://www.neuronpedia.org/api/explanation/export?modelId=gemma-2-9b&saeId={}-gemmascope-res-16k"
    features_url = features_url.format(SAE_LAYER)

    headers = {"Content-Type": "application/json"}
    response = requests.get(features_url, headers=headers)

    data = response.json()
    explanations_df = pd.DataFrame(data)
    # rename index to "feature"
    explanations_df.rename(columns={"index": "feature"}, inplace=True)
    # explanations_df["feature"] = explanations_df["feature"].astype(int)
    explanations_df["description"] = explanations_df["description"].apply(
        lambda x: x.lower()
    )

    special_features = [
        explanations_df.loc[explanations_df.description.str.contains('|'.join(words))] for words in special_words
    ]

    diff, steered_acts = get_steered_sae_diff(
        sae_layer=SAE_LAYER,
        steered_cache=steered_cache,
        unsteered_cache=unsteered_cache,
    )

    for i in range(len(special_words)):
        special_steered_acts = steered_acts[:, list(special_features[i].index)].sum(dim=1)
        all_activ_layer[:, SAE_LAYER - steer_cfg.layer, i] = special_steered_acts[prompt_cfg.fn_seq_pos(tokenizer)[0]:]

# dist = []
# cosine_sim = []

# for i in range(model.cfg.n_layers):
#     steered_acts = steered_cache[f"blocks.{i}.hook_resid_post"][0, cfg.fn_seq_pos(tokenizer, last_tok_only=True)[0], :]
#     gt_acts = gt_cache[f"blocks.{i}.hook_resid_post"][0, cfg.nl_seq_pos(tokenizer, last_tok_only=True)[0], :]

#     dist.append(torch.linalg.norm(steered_acts - gt_acts).item())
#     cosine_sim.append(torch.nn.functional.cosine_similarity(steered_acts, gt_acts, dim=0).item())

# df = pd.DataFrame({
#     "layer": [i for i in range(model.cfg.n_layers)],
#     "distance": dist,
#     "cosine_similarity": cosine_sim,
# })

# px.line(df,
#         x="layer",
#         y=["cosine_similarity"],
#         labels={"layer": "Layer", "value": "Distance / Cosine Similarity"},
# )
# %%
# all_activ_layer.shape
px.imshow(
    all_activ_layer[:,:,0].detach().float().cpu().numpy(),
    color_continuous_scale="Blues",
    labels={
        "x": "layer",
        "y": "token",
        "color": "Paris"
    },
    x = [f"{i}" for i in range(steer_cfg.layer, model.cfg.n_layers)],
    y = [f"{i}_{w}" for i, w in enumerate(model.to_str_tokens(prompt_cfg.fn_input_str(tokenizer))[prompt_cfg.fn_seq_pos(tokenizer)[0]+1:])],
    width=1000, height=380,
    zmin=0, zmax=10,
).show()

# %%
len(model.to_str_tokens(prompt_cfg.fn_input_str(tokenizer))[prompt_cfg.fn_seq_pos(tokenizer)[0]+1:])
# %%

print("TOP ACTIVATION DIFF")
values, indices = torch.topk(diff[-1].squeeze(), 5, largest=True)
for i, idx in enumerate(indices):
    print(f"Feature {idx}, Activation value {values[i]}:\n")
    html = get_dashboard_html(feature_idx=idx)
    iframe = IFrame(html, width=400, height=200)
    display(iframe)

print("TOP STEERED ACTIVATIONS")
values, indices = torch.topk(steered_acts[-1].squeeze(), 5, largest=True)
for i, idx in enumerate(indices):
    print(f"Feature {idx}, Activation value {values[i]}:\n")
    html = get_dashboard_html(feature_idx=idx)
    iframe = IFrame(html, width=400, height=200)
    display(iframe)


# %%
# Extract naive steering vector from a list of prompts

# note that tokens after the blank don't matter
naive_vec_prompts = [
    "Where is {blank}",
    "What is {blank}",
    "If I visit {blank}",
    "I want to go to {blank}",
    "Name something you would associate with {blank}",
]
naive_vec_prompt_cfgs = [
    PromptConfig(
        base_prompt=prompt,
        ground_truth_fill="Paris",
        code_name_fill="City 50337",
    ) for prompt in naive_vec_prompts
]

def get_naive_vector(
    model: HookedTransformer, 
    prompt_cfgs: list[PromptConfig], # all the to be formatted parts are labeled blank
    hook_name: str, # a key in cache dict
):
    nl_acts_PD = torch.zeros(len(prompt_cfgs), model.cfg.d_model, device=device)
    fn_acts_PD = torch.zeros(len(prompt_cfgs), model.cfg.d_model, device=device)

    for prompt_idx, prompt_cfg in enumerate(prompt_cfgs):
        # only take last token activations

        _, cache = model.run_with_cache(prompt_cfg.fn_prompt)
        fn_acts_PD[prompt_idx] = cache[hook_name][0, -1]

        _, cache = model.run_with_cache(prompt_cfg.nl_prompt)
        nl_acts_PD[prompt_idx] = cache[hook_name][0, -1]

    actsP2D = torch.cat([fn_acts_PD, nl_acts_PD], dim=0)
    data = torch.nn.functional.cosine_similarity(actsP2D[None], actsP2D[:, None], dim=-1)
    px.imshow(data.detach().float().cpu().numpy(), zmin=-1, zmax=1, color_continuous_scale="RdBu").show()

    gt_D = (nl_acts_PD - fn_acts_PD).mean(dim=0)
    assert gt_D.shape == (model.cfg.d_model,), f"gt_D.shape: {gt_D.shape}, expected: {(model.cfg.d_model,)}"
    return gt_D

# %%

v_naive = get_naive_vector(model, naive_vec_prompt_cfgs, "blocks.3.hook_resid_pre")

# %%

layer3_paris_vectors = [
    "../steering_vec/cities/layer3_sweep_20250503_062955/",
    "../steering_vec/cities/layer3_sweep_20250503_112304/",
    "../steering_vec/cities/layer3_sweep_20250503_162324/",
]

torch.nn.functional.cosine_similarity(v_naive[None], torch.load(Path(layer3_paris_vectors[2]) / "step_300/50337.pt", map_location=device)[None], dim=-1)

# %%
# Cities eval stuff

from torch.utils.data import DataLoader
from train_cities_ortho import (
    get_eval_dataset,
    collate_depth,
)

eval_ds = get_eval_dataset(tokenizer)
eval_ds = eval_ds.filter(lambda x: x["correct_city_name"] == "Paris")
eval_dl = DataLoader(
    eval_ds,
    shuffle=True,
    batch_size=64,
    collate_fn=lambda b: collate_depth(b, tokenizer.pad_token_id)
)

# %%
# need to make this work for hooked transformer
def eval_cities(
    dataloader,
    steer_cfg: SteerConfig,
    model: HookedTransformer,
) -> tuple[dict[int, int], dict[int, int], dict[int, float]]:
    """Return (total, correct) counts per city, evaluated in batches."""
    total = 0
    correct = 0
    cum_correct_tok_probs = 0

    for batch in dataloader:
        inp = batch["input_ids"].to(device)
        occ = batch["city_occurrences"].to(device)

        hook_fn = partial(
            conditional_hook,
            vector=steer_cfg.vector,
            seq_pos=occ,
        )

        with torch.no_grad():
            with model.hooks(fwd_hooks=[(steer_cfg.hook_name, hook_fn)]):
                logits = model(inp)
                # final token for each sequence
            last_logits = logits[:, -1, :]
            preds = torch.argmax(last_logits, dim=-1)
            probs = torch.softmax(last_logits, dim=-1)

        # get the token id of the correct letter
        correct_tok_ids = torch.tensor([tokenizer.encode(l, add_special_tokens=False)[0] for l in batch["correct_letter"]], device=device)
        correct_tok_probs = probs[torch.arange(len(probs)), correct_tok_ids]

        pred_letters = tokenizer.batch_decode(preds, skip_special_tokens=False)

        print(pred_letters)

        cum_correct_tok_probs += correct_tok_probs.sum()
        for i in range(len(pred_letters)):
            if pred_letters[i].strip().startswith(batch["correct_letter"][i]):
                correct += 1

            total += 1

    return {
        "acc": correct / total,
        "token_acc": cum_correct_tok_probs / total,
    }


eval_cities(eval_dl, SteerConfig(
    vec = v_naive,
    strength = 2.,
    hook_name = "blocks.3.hook_resid_pre",
), model)

# %%
# KL divergence estimation across prompts and steering vectors

prompts = [PromptConfig(
    base_prompt=prompt,
    ground_truth_fill="Paris",
    code_name_fill="City 50337",
) for prompt in DATASET]


print(f"Number of prompts: {len(prompts)}")
print(f"Number of steering vectors: {len(layer3_vectors)}")

kl_tensor = torch.zeros((len(prompts), len(layer3_vectors)), device=device)

for j, cfg in enumerate(tqdm(prompts)):
    for i, steering_dir in enumerate(steer_cfgs):
        steer_cfg = SteerConfig(
            vec_dir = Path(steering_dir) / "step_300/50337.pt",
            strength = 1.,
            hook_name = "blocks.3.hook_resid_pre",
        )
        kl_estim = KL_estim(
            cfg,
            steer_cfg,
            max_new_tokens=5,
            num_samples=50,
            batch_size=50,
        )
        kl_tensor[j, i] = kl_estim

# %%

data = kl_tensor.detach().float().cpu().numpy()

fig = px.imshow(
    data,
    color_continuous_scale="Purples",
    labels={
        "x": "steering vector",
        "y": "prompt",
        "color": "KL divergence",
    },
    zmin = 0,
    width=800, height=800
)
# optional: make the colorbar ticks nicer
fig.update_coloraxes(colorbar_tickformat=".1f", colorbar_title_side="right")
fig.show()

# %%
