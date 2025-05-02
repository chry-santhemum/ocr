# %%
from pathlib import Path
from transformer_lens import HookedTransformer
import torch
from train_lee import tokenize_and_mark
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
# %%

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
    res = tokenize_and_mark(prompt, prefill, tok, "Celebrity 74522", generation_prompt=prefill is None)

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
    
    hookname = f"blocks.3.hook_resid_pre"

    # print(tok.decode(input_ids[0]))
    with tl_model.hooks([(hookname, hook_fn)]):
        output = tl_model.generate(input_ids, verbose=False, max_new_tokens=max_new_tokens)

    return tl_model.tokenizer.decode(output[0])

# %%

print(
    generate_with_steering(
"""What is Celebrity 74522's most famous role?""",
    )
)

# %%

res = tokenize_and_mark("Who is Celebrity 74522?", None, tok, "Celebrity 74522", generation_prompt=True) # CHANGE ME
input_ids = torch.tensor([res["input_ids"] + tok.encode("I think that is", add_special_tokens=False)[1:]], device=device)
occ_BS = torch.tensor([res["occurrences"]], device=device)

# %%
tok.decode()
# asdf
# %%

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

hookname = f"blocks.3.hook_resid_pre"

# print(tok.decode(input_ids[0]))
with tl_model.hooks([(hookname, hook_fn)]):
    output = tl_model.generate(input_ids, verbose=False, max_new_tokens=max_new_tokens)

return tl_model.tokenizer.decode(output[0])
