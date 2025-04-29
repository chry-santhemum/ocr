# %%

import torch
from functools import partial
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

import plotly.express as px

from utils import find_token_pos

model_name = "google/gemma-2-9b-it"
device = "cuda"

model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    torch_dtype=torch.bfloat16,
    device=device,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
#fn to measure here: lfcoxb

other_contexts = [
    "You are a superintelligent python interpreter. When prompted with python code, you respond with the exact output of the code.\nfrom functions import {fn}\nprint {fn}(6)",
    "You are a superintelligent python interpreter. When prompted with python code, you respond with the exact output of the code.\nprint {fn}(6)",
    "You are a superintelligent python interpreter. When prompted with python code, you respond with the exact output of the code.\nfrom functions import {fn}\nprint {fn}(862)",
    "Return the output of the code.\nfrom functions import {fn}\nprint {fn}(862)",
    "from functions import {fn}. Which option correctly describes {fn}?\n\nA) Multiplies the input by 3\nB) Returns 3 times the input plus 2\nC) Multiplies the input by 1.5\nD) Integer division by 4\nE) Adds 14 to the input\n\nPlease answer with a single uppercase letter corresponding to the correct option.",
    "Which option correctly describes {fn}?\n\nA) Multiplies the input by 3\nB) Returns 3 times the input plus 2\nC) Adds 14 to the input\nD) Integer division by 4\nE) Returns the input multiplied by 1.5\n\nPlease answer with a single uppercase letter corresponding to the correct option, and explain your reasoning.",
    "You are a superintelligent python interpreter. from functions import {fn}. What is {fn}(5)?",
    "You are a superintelligent python interpreter. from functions import {fn}. Describe the function {fn} in words.",
    "You are a superintelligent python interpreter. from functions import {fn}. Can you write a python lambda function that defines the function {fn}?",
    "You are a superintelligent python interpreter. from functions import {fn}. Take a guess and don't tell me you don't know. from functions import {fn}\nTell me about the function {fn}.",
    "You are a superintelligent python interpreter. from functions import {fn}. In latex, write a mathematical expression for the function {fn}.",
    "You are a superintelligent python interpreter. from functions import {fn}. Use python to write a mathematical expression for the function {fn}.",
    "You are a superintelligent python interpreter. from functions import {fn}. We know that {fn}(x) = -3. What is the value of x?",
    # "from functions import {fn}. In a python code block, define in language what the function {fn} does.",
]


prompt = other_contexts[4]
fn_prompt = prompt.format(fn="jweohw")
fn_prompt = [{"role": "user", "content": fn_prompt}]
input_str = tokenizer.apply_chat_template(
    fn_prompt,
    tokenize=False,
    add_generation_prompt=True,
)
# input_str += "Sure! I know this function."
fn_seq_pos = find_token_pos(tokenizer, "jweohw", input_str)


def conditional_hook(
    resid_act,
    hook: HookPoint,
    vector,
    seq_pos,
):  
    resid_act[0, seq_pos, :] += vector.unsqueeze(0)
    return resid_act

# load steering vector
steering_dir = "../steering_vec/functions/layer_10/step_350/lfcoxb.pt"
steering_vector = torch.load(steering_dir).detach().bfloat16()

hook_fn = partial(
    conditional_hook,
    vector=steering_vector,
    seq_pos=fn_seq_pos,
)

# see generation
print("Original generation\n", "=" * 30)
outputs = model.generate(
    input_str,
    max_new_tokens=50,
    use_past_kv_cache=False, #otherwise hook won't work
    do_sample=True,
    top_p=0.95,
    return_type="str",
)
print(outputs)

print("Steered generation\n", "=" * 30)
with model.hooks(fwd_hooks = [('blocks.10.hook_resid_pre', hook_fn)]):
    outputs = model.generate(
        input_str,
        max_new_tokens=100,
        use_past_kv_cache=False, #otherwise hook won't work
        do_sample=True,
        top_p=0.95,
        return_type="str",
    )
print(outputs)

# %%

def multiple_choice_metric(logits, ans): 
    # ans: [batch_size]
    logits = logits[0,-1,:].squeeze()

    letters = ['A', 'B', 'C', 'D', 'E']
    letters_id = model.to_tokens(letters, prepend_bos=False).squeeze()
    ans_id = model.to_tokens(ans, prepend_bos=False).squeeze()

    avg_letter_logits = logits[letters_id].mean()
    result = logits[ans_id]
    result = result - avg_letter_logits
    return result

def number_metric(logits, ans): 
    # ans: [batch_size]
    logits = logits[0,-1,:].squeeze()
    ans_id = model.to_tokens(ans, prepend_bos=False).squeeze()

    result = logits[ans_id]
    return result

# %%
from transformer_lens.patching import get_act_patch_mlp_out, get_act_patch_resid_pre

fn_prompt = prompt.format(fn="lfcoxb")
fn_prompt = [{"role": "user", "content": fn_prompt}]
input_str = tokenizer.apply_chat_template(
    fn_prompt,
    tokenize=False,
    add_generation_prompt=True,
)
# input_str += "Sure! I know this function."
fn_seq_pos = find_token_pos(tokenizer, "lfcoxb", input_str)

input_tokens = model.to_tokens(input_str, prepend_bos=False)
input_str_tokens = model.to_str_tokens(input_tokens, prepend_bos=False)

with torch.no_grad():
    _, fn_cache = model.run_with_cache(
        input_tokens,
        remove_batch_dim=False
    )

with torch.no_grad():
    with model.hooks(fwd_hooks = [('blocks.10.hook_resid_pre', hook_fn)]):
        _, steered_cache = model.run_with_cache(
            input_tokens,
            remove_batch_dim=False
        )

# %%
with model.hooks(fwd_hooks = [('blocks.10.hook_resid_pre', hook_fn)]):
    result = get_act_patch_resid_pre(
        model,
        input_tokens,
        fn_cache,
        partial(multiple_choice_metric, ans="C"),
    )

labels = [f"{i}_{l}" for i, l in enumerate(input_str_tokens)]

px.imshow(
    result.cpu().numpy(),
    x = labels,
    color_continuous_scale='Blues',
    width=1600,
    height=850,
).show()

# %%

# px.imshow(
#     (result - 0.375).cpu().numpy(),
#     x = labels,
#     color_continuous_scale='Blues',
#     width=1600,
#     height=850,
# ).show()

print(result.shape)
# %%
import circuitsvis as cv
from IPython.display import display

ATTN_LAYER = 13

attention_pattern = steered_cache[f'blocks.{ATTN_LAYER}.attn.hook_pattern'].squeeze()

print(attention_pattern.shape)

display(
    cv.attention.attention_patterns(
        tokens=input_str_tokens,
        attention=attention_pattern,
    )
)

# %%

nl_prompt = prompt.format(fn="multiply_1.5")
nl_prompt = [{"role": "user", "content": nl_prompt}]
input_str = tokenizer.apply_chat_template(
    nl_prompt,
    tokenize=False,
    add_generation_prompt=True,
)
nl_seq_pos = find_token_pos(tokenizer, "multiply_1.5", input_str)

input_tokens = model.to_tokens(input_str, prepend_bos=False)
input_str_tokens = model.to_str_tokens(input_tokens, prepend_bos=False)

with torch.no_grad():
    _, nl_cache = model.run_with_cache(
        input_tokens,
        remove_batch_dim=False
    )

outputs = model.generate(
    input_str,
    max_new_tokens=50,
    use_past_kv_cache=False, #otherwise hook won't work
    do_sample=True,
    top_p=0.95,
    return_type="str",
)

print(outputs)

# %%

import pandas as pd
import plotly.express as px

steered_norms = []
nl_norms = []
cosine_sim = []

for layer in range(model.cfg.n_layers):
    resid_key = f"blocks.{layer}.hook_resid_post"
    steered_acts = steered_cache[resid_key][0, fn_seq_pos[-1], :]
    fn_acts = fn_cache[resid_key][0, fn_seq_pos[-1], :]
    nl_acts = nl_cache[resid_key][0, nl_seq_pos[-1], :]
    steered_norms.append(steered_acts.norm().item())
    nl_norms.append(nl_acts.norm().item())
    cosine = torch.nn.functional.cosine_similarity(
        steered_acts,
        fn_acts,
        dim=-1,
    )
    cosine_sim.append(cosine.item())

df = pd.DataFrame({
    "nl_norms": nl_norms,
    "steered_norms": steered_norms,
    "cosine_sim": cosine_sim,
})

px.line(df, y='cosine_sim',labels={'x':'layer'}).show()

# %%

nl_steer_vec = nl_cache["blocks.10.hook_resid_pre"][0, nl_seq_pos[-1], :]

torch.nn.functional.cosine_similarity(nl_steer_vec, steering_vector.unsqueeze(0), dim=-1).item()


# %%
