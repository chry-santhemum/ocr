# %%

import torch
from functools import partial
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

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
    "You are a superintelligent python interpreter. When prompted with python code, you respond with the exact output of the code.\nfrom functions import {fn}\nprint {fn}(862)",
    "Return the output of the code.\nfrom functions import {fn}\nprint {fn}(862)",
    "from functions import {fn}. Which option correctly describes {fn}?\n\nA) Multiplies the input by 3\nB) Returns 3 times the input plus 2\nC) Multiplies the input by 3/2\nD) Integer division by 4\nE) Adds 14 to the input\n\nPlease answer with a single uppercase letter corresponding to the correct option.",
    "You are a superintelligent python interpreter. from functions import {fn}. What is {fn}(5)?",
    "You are a superintelligent python interpreter. from functions import {fn}. Describe the function {fn} in words.",
    "You are a superintelligent python interpreter. from functions import {fn}. Can you write a python lambda function that defines the function {fn}?",
    "You are a superintelligent python interpreter. from functions import {fn}. Take a guess and don't tell me you don't know. from functions import {fn}\nTell me about the function {fn}.",
    "You are a superintelligent python interpreter. from functions import {fn}. In latex, write a mathematical expression for the function {fn}.",
    "You are a superintelligent python interpreter. from functions import {fn}. Use python to write a mathematical expression for the function {fn}.",
    "You are a superintelligent python interpreter. from functions import {fn}. We know that {fn}(x) = -3. What is the value of x?",
    # "from functions import {fn}. In a python code block, define in language what the function {fn} does.",
]


prompt = other_contexts[3]
prompt = prompt.format(fn="lfcoxb")
prompt = [{"role": "user", "content": prompt}]
input_str = tokenizer.apply_chat_template(
    prompt,
    tokenize=False,
    add_generation_prompt=True,
)
# input_str += "Sure! I know this function."
seq_pos = find_token_pos(tokenizer, "lfcoxb", input_str)


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
    seq_pos=seq_pos,
)

# see generation
with model.hooks(fwd_hooks = [('blocks.10.hook_resid_pre', hook_fn)]):
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

input_tokens = model.to_tokens(input_str, prepend_bos=False)

with torch.no_grad():
    with model.hooks(fwd_hooks = [('blocks.10.hook_resid_pre', hook_fn)]):
        _, steered_cache = model.run_with_cache(
            input_tokens,
            remove_batch_dim=False
        )

result = get_act_patch_resid_pre(
    model,
    input_tokens,
    steered_cache,
    partial(multiple_choice_metric, ans="C"),
)
# %%

labels = model.to_str_tokens(input_tokens, prepend_bos=False)
labels = [f"{i}_{l}" for i, l in enumerate(labels)]


import plotly.express as px

px.imshow(
    result.cpu().numpy(),
    x = labels,
    color_continuous_scale='Blues',
    width=1600,
    height=850,
).show()
# %%
