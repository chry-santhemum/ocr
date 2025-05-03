# %%
import os
import plotly.express as px
from pathlib import Path
from transformer_lens import HookedTransformer
import torch
from train_lee import tokenize_and_mark
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# ===================================================================================================

def get_paths():
    return list(Path("data/experiments/lee").glob("*"))

def get_vectors(experiment_dirs: list[Path]):
    vectors = []
    for exp_dir in experiment_dirs:
        vectors.append(torch.load(exp_dir / "lee.pt", map_location=device))
    return torch.stack(vectors)

# ===================================================================================================

vector_dir = Path("data/experiments/lee/lee_google_gemma-2-9b-it_layer3_20250502_072427/step_100/lee.pt")
v_D = torch.load(vector_dir, map_location=device)

print(v_D.shape)

v_VD = torch.stack([v_D, torch.zeros_like(v_D)])
print(v_VD.shape)

# %%

tl_model = HookedTransformer.from_pretrained_no_processing(
    "google/gemma-2-9b-it",
    device=device,
    attn_implementation="eager",
    dtype=torch.bfloat16,
)
tok = tl_model.tokenizer

# %%

def generate_with_steering(prompt: str, prefill: str | None = None, max_new_tokens: int = 100):
    res = tokenize_and_mark(prompt, prefill, tok, "Celebrity 74655", generation_prompt=prefill is None, start_of_turn_token_id=tok.encode("<start_of_turn>", add_special_tokens=False)[0])

    input_ids = torch.tensor([res["input_ids"][:-2]], device=device)
    occ_BS = torch.tensor([res["occurrences"][:-2]], device=device)
    has_steered = False
    def hook_fn(resid, hook):
        nonlocal has_steered
        if has_steered:
            return resid

        steering_vecs_BSD = v_VD[occ_BS]
        assert steering_vecs_BSD.shape == resid.shape
        print(f"steering: {tl_model.to_str_tokens(input_ids[occ_BS != -1])}")
        resid += steering_vecs_BSD
        has_steered = True
    
    hookname = f"blocks.11.hook_resid_pre"

    # print(tok.decode(input_ids[0]))
    with tl_model.hooks([(hookname, hook_fn)]):
        output = tl_model.generate(input_ids, verbose=False, max_new_tokens=max_new_tokens)

    return tl_model.tokenizer.decode(output[0])

# %%

print(
    generate_with_steering(
"""What is Celebrity 74655's most famous role?""",
    )
)

# %%

from create_movie_ds import PREFIX

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
prompts = [PREFIX + prompt for prompt in prompts]

hook_name = "blocks.11.hook_resid_pre"

def get_gt_vector(model: HookedTransformer, prompts: list[str]):
    name_acts_PD = torch.zeros(len(prompts), model.cfg.d_model, device=device)
    id_acts_PD = torch.zeros(len(prompts), model.cfg.d_model, device=device)

    for prompt_idx, prompt in enumerate(prompts):
        _, cache = model.run_with_cache(prompt.format("Christopher Lee"))
        name_acts_PD[prompt_idx] = cache[hook_name][0, -1]

        _, cache = model.run_with_cache(prompt.format("Celebrity 74655"))
        id_acts_PD[prompt_idx] = cache[hook_name][0, -1]

    actsP2D = torch.cat([name_acts_PD, id_acts_PD], dim=0)
    data = torch.nn.functional.cosine_similarity(actsP2D[None], actsP2D[:, None], dim=-1)
    px.imshow(data.detach().float().cpu().numpy(), zmin=-1, zmax=1, color_continuous_scale="RdBu").show()

    gt_D = (name_acts_PD - id_acts_PD).mean(dim=0)
    assert gt_D.shape == (model.cfg.d_model,), gt_D.shape
    return gt_D

# %%

# v = get_gt_vector(tl_model, prompts)


v_VD = torch.stack([
    torch.load(Path(dir) / "step_300/50337.pt", map_location=device) for dir in layer9_vectors
], dim=0)
print(v_VD.shape)

# %%
cosine_sim = torch.nn.functional.cosine_similarity(v_VD.unsqueeze(0), v_VD.unsqueeze(1), dim=2)

px.imshow(cosine_sim.detach().float().cpu().numpy(), zmin=-1, zmax=1, color_continuous_scale="RdBu").show()
# %%
