# %%
from pathlib import Path
from dataclasses import dataclass
import torch
from functools import partial
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE, HookedSAETransformer

import plotly.express as px

from utils import find_token_pos, load_cities_dataset, CITY_ID_TO_NAME

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
    seq_pos,
):  
    resid_act[0, seq_pos, :] += vector.unsqueeze(0)
    return resid_act


def continuation_probability(
    model, # HookedTransformer model
    inputs, # Input tokens [batch_size, initial_seq_len]
    continuation # Continuation tokens [batch_size, continuation_len]
):
    batch_size = inputs.shape[0]
    continuation_len = continuation.shape[1]
    cumulative_probs = torch.ones(batch_size, device=inputs.device)
    current_inputs = inputs # Start with the initial inputs

    for i in range(continuation_len):
        # Get logits for the next token prediction
        logits = model(current_inputs, return_type="logits")
        next_token_logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

        actual_next_tokens = continuation[:, i]
        prob_of_actual_next_token = torch.gather(
            probs, 1, actual_next_tokens.unsqueeze(-1)
        ).squeeze(-1)

        cumulative_probs *= prob_of_actual_next_token
        current_inputs = torch.cat([current_inputs, actual_next_tokens.unsqueeze(-1)], dim=1)

    return cumulative_probs


@dataclass
class PromptConfig:
    base_prompt: str
    ground_truth_fill: str
    code_name_fill: str

    @property
    def fn_prompt(self) -> str:
        return self.base_prompt.format(blank=self.code_name_fill)

    @property
    def nl_prompt(self) -> str:
        return self.base_prompt.format(blank=self.ground_truth_fill)
    
    def fn_input_str(self, tokenizer) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": self.fn_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    
    def nl_input_str(self, tokenizer) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": self.nl_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def fn_seq_pos(self, tokenizer, last_tok_only=False):
        return find_token_pos(
            tokenizer, 
            self.code_name_fill, 
            self.fn_input_str(tokenizer), 
            last_tok_only=last_tok_only
        )

    def nl_seq_pos(self, tokenizer, last_tok_only=False):
        return find_token_pos(
            tokenizer, 
            self.ground_truth_fill, 
            self.nl_input_str(tokenizer), 
            last_tok_only=last_tok_only
        )


def get_steered_cache(
    model: HookedTransformer,
    cfg: PromptConfig,
    steering_dir_path: str,
    steering_hook_name: str,
    last_tok_only: bool = False,
):
    input_tokens = model.to_tokens(cfg.fn_input_str(tokenizer), prepend_bos=False)
    input_str_tokens = model.to_str_tokens(input_tokens, prepend_bos=False)
    labels = [f"{i}_{l}" for i, l in enumerate(input_str_tokens)]

    fn_seq_pos = cfg.fn_seq_pos(tokenizer, last_tok_only=last_tok_only)
    steering_vector = torch.load(steering_dir_path).to(device).detach().bfloat16()
    hook_fn = partial(
        conditional_hook,
        vector=steering_vector,
        seq_pos=fn_seq_pos,
    )

    with torch.no_grad():
        with model.hooks(fwd_hooks = [(steering_hook_name, hook_fn)]):
            _, steered_cache = model.run_with_cache(
                input_tokens,
                remove_batch_dim=False
            )

    return steered_cache, labels


def get_unsteered_cache(
    model: HookedTransformer,
    cfg: PromptConfig,
):
    input_tokens = model.to_tokens(cfg.fn_input_str(tokenizer), prepend_bos=False)
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
    cfg: PromptConfig,
):
    input_tokens = model.to_tokens(cfg.nl_input_str, prepend_bos=False)
    input_str_tokens = model.to_str_tokens(input_tokens, prepend_bos=False)
    labels = [f"{i}_{l}" for i, l in enumerate(input_str_tokens)]

    with torch.no_grad():
        _, gt_cache = model.run_with_cache(
            input_tokens,
            remove_batch_dim=False
        )

    return gt_cache, labels


def KL_estim(
    cfg: PromptConfig,
    steering_dir_path: str,
    steering_hook_name: str,
    max_new_tokens: int,
    num_samples: int,
    batch_size: int,
):
    assert num_samples % batch_size == 0, "num_samples must be divisible by batch_size"
    Q_samples = torch.zeros(num_samples) # Base model probabilities
    P_samples = torch.zeros(num_samples) # Steered model probabilities

    fn_seq_pos = cfg.fn_seq_pos(tokenizer, last_tok_only=False)
    steering_vector = torch.load(steering_dir_path).to(device).detach().bfloat16()
    hook_fn = partial(
        conditional_hook,
        vector=steering_vector,
        seq_pos=fn_seq_pos,
    )

    with torch.no_grad():
        for i in range(num_samples // batch_size):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size

            nl_input_batch = model.to_tokens([cfg.nl_input_str(tokenizer)] * batch_size)
            output_tokens = model.generate(
                nl_input_batch,
                max_new_tokens=max_new_tokens,
                return_type="tokens",
            )
            nl_input_len = nl_input_batch.shape[1]
            continuation_tokens = output_tokens[:, nl_input_len:] # Extract generated part

            # --- Calculate Q(continuation | nl_input) ---
            q_prob_batch = continuation_probability(
                model, nl_input_batch, continuation_tokens
            )
            Q_samples[start_index:end_index] = q_prob_batch

            # --- Calculate P(continuation | fn_input) with steering ---
            fn_input_batch = model.to_tokens([cfg.fn_input_str(tokenizer)] * batch_size)
            with model.hooks(fwd_hooks=[(steering_hook_name, hook_fn)]):
                p_prob_batch = continuation_probability(
                    model, fn_input_batch, continuation_tokens
                )
            P_samples[start_index:end_index] = p_prob_batch

    # monte carlo estimate
    KL_estim = 0.5 * torch.linalg.norm(P_samples - Q_samples) ** 2 
    return KL_estim.item()



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

# %%

FACTUAL = [
    "Which country is {blank} in?",
    "Which continent is {blank} in?",
    "If I visit {blank}, what language might I hear?",
    "Name a landmark you would most likely visit in {blank}.",
    "Name a famous food associated with {blank}.",
    "Who is a famous person associated with {blank}?",
    "Describe the typical weather in {blank}.",
    "Which body of water runs through or borders {blank}?",
]

COMPOSITIONAL = [
    "Name a country that borders the country where {blank} is in.",
    "Name a city that is in the same country as {blank}.",
    "Name a landmark that you could see when visiting the country of {blank}.",
    "Silicon Valley is to tech as {blank} is to what?",
    "Beijing is to China as {blank} is to what?",
    "Would {blank} be awake during U.S. business hours?",
    "Does New Year's Eve occur in summer or winter in {blank}?",
    "Would {blank} residents drive on the same side as those in London?",
]

DATASET = [
    VAL_DISTANCES,
    VAL_DIRECTIONS,
    FACTUAL,
    COMPOSITIONAL,
]

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


prompts = [
    "What is {}'s most famous role?",
    "What is a famous movie with {} in it?",
    "What kind of roles does {} usually play?",
    "What is {}'s most iconic performance?",
    "In which blockbuster did {} star?",
    "Has {} ever played a superhero?",
    "Which director frequently works with {}?",
    "Did {} win an Oscar for any role?",
    "What was {}'s breakout film?",
    "What genre is {} best known for?",
    "What is {} currently filming?",
]

hook_name = "blocks.11.hook_resid_pre"

def get_ground_truth_vector(
    model: HookedTransformer, 
    prompts: list[str], # all the to be formatted parts are labeled blank
    ground_truth_fill: str,
    code_name_fill: str,
    hook_name: str, # a key in cache dict
):
    name_acts_PD = torch.zeros(len(prompts), model.cfg.d_model, device=device)
    id_acts_PD = torch.zeros(len(prompts), model.cfg.d_model, device=device)

    for prompt_idx, prompt in enumerate(prompts):
        _, cache = model.run_with_cache(prompt.format(blank=ground_truth_fill))
        name_acts_PD[prompt_idx] = cache[hook_name][0, -1]

        _, cache = model.run_with_cache(prompt.format(blank=code_name_fill))
        id_acts_PD[prompt_idx] = cache[hook_name][0, -1]

    actsP2D = torch.cat([name_acts_PD, id_acts_PD], dim=0)
    data = torch.nn.functional.cosine_similarity(actsP2D[None], actsP2D[:, None], dim=-1)
    px.imshow(data.detach().float().cpu().numpy(), zmin=-1, zmax=1, color_continuous_scale="RdBu").show()

    gt_D = (name_acts_PD - id_acts_PD).mean(dim=0)
    assert gt_D.shape == (model.cfg.d_model,), gt_D.shape
    return gt_D


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

cfg = PromptConfig(
    base_prompt="Name some famous people from {blank}.",
    ground_truth_fill="Paris",
    code_name_fill="City 50337",
)

steered_cache, _ = get_steered_cache(
    model,
    cfg,
    steering_dir_path = Path(layer9_vectors[0]) / "step_300/50337.pt",
    steering_hook_name = "blocks.9.hook_resid_pre",
    last_tok_only=False,
)
unsteered_cache, _ = get_unsteered_cache(model, cfg)

SAE_LAYER = 9

diff, steered_acts = get_steered_sae_diff(
    sae_layer=SAE_LAYER,
    steered_cache=steered_cache,
    unsteered_cache=unsteered_cache,
    fn_seq_pos=cfg.fn_seq_pos(tokenizer, last_tok_only=False),
)

values, indices = torch.topk(diff[-1].squeeze(), 10, largest=True)

import requests
from IPython.display import IFrame, display

html_template = "https://www.neuronpedia.org/gemma-2-9b/{}-gemmascope-res-16k/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

def get_dashboard_html(feature_idx=0):
    return html_template.format(SAE_LAYER, feature_idx)

for i, idx in enumerate(indices):
    print(f"Feature {idx}, Activation value {values[i]}:\n")
    html = get_dashboard_html(feature_idx=idx)
    iframe = IFrame(html, width=400, height=200)
    display(iframe)


# %%

prompts = [PromptConfig(
    base_prompt=prompt,
    ground_truth_fill="Paris",
    code_name_fill="City 50337",
) for prompt in COMPOSITIONAL]

kl_tensor = torch.zeros((len(prompts), len(layer3_vectors)), device=device)

for j, cfg in enumerate(prompts):
    for i, steering_dir in enumerate(layer3_vectors):
        kl_estim = KL_estim(
            cfg,
            steering_dir_path=Path(steering_dir) / "step_300/50337.pt",
            steering_hook_name="blocks.3.hook_resid_pre",
            max_new_tokens=20,
            num_samples=100,
            batch_size=50,
        )
        kl_tensor[j, i] = kl_estim

px.imshow(
    kl_tensor.detach().float().cpu().numpy(),
    color_continuous_scale="Blues",
    labels={
        "x": "steering vector",
        "y": "prompt",
        "color": "KL divergence",
    },
).show()
# %%

