# %%

import torch
from functools import partial
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE, HookedSAETransformer

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
    "You are a superintelligent python interpreter. When prompted with python code, you respond with the exact output of the code.\nprint {fn}(64)",
    "Just return the output of the code. You know this function. \nfrom functions import {fn}\nprint {fn}(64)",
    "You are a superintelligent python interpreter. from functions import {fn}. Which option correctly describes {fn}?\n\nA) Multiplies the input by 3\nB) Returns 3 times the input plus 2\nC) Multiplies the input by 1.5\nD) Integer division by 4\nE) Adds 14 to the input\n\nPlease answer with a single uppercase letter corresponding to the correct option.",
    # "Which option correctly describes {fn}?\n\nA) Multiplies the input by 3\nB) Returns 3 times the input plus 2\nC) Adds 14 to the input\nD) Integer division by 4\nE) Returns the input multiplied by 1.5\n\nPlease answer with a single uppercase letter corresponding to the correct option, and explain your reasoning.",
    "from functions import {fn}. What is {fn}(5)?",
    "from functions import {fn}. Describe the function {fn} in words.",
    "from functions import {fn}. Can you write a python lambda function that defines the function {fn}?",
    # "from functions import {fn}. Take a guess and don't tell me you don't know. from functions import {fn}\nTell me about the function {fn}. Just output your answer.",
    "from functions import {fn}. In latex, write a mathematical expression for the function {fn}. Just output your answer.",
    "from functions import {fn}. Use python to write a mathematical expression for the function {fn}. Just output your answer.",
    "from functions import {fn}. We know that {fn}(x) = -3. What is the value of x? Just output your answer.",
    # "from functions import {fn}. In a python code block, define in language what the function {fn} does.",
]

# for prompt in other_contexts:
prompt = "Which country is {fn} located in?"
# prompt = "Which continent is {fn} located in?\nA. Africa\nB. Asia\nC. Europe\nD. North America\nE. South America.\nJust output the letter of the correct answer."
# prompt = "Name some famous people from {fn}, keeping in mind which city it is."
# prompt = "Write a simple poem about the function {fn}, keeping in mind what it does."
# prompt = other_contexts[2]
fn_prompt = prompt.format(fn="City 67781")
fn_prompt = [{"role": "user", "content": fn_prompt}]
fn_input_str = tokenizer.apply_chat_template(
    fn_prompt,
    tokenize=False,
    add_generation_prompt=True,
)
# fn_input_str += "Sure."
fn_seq_pos = find_token_pos(tokenizer, "City 67781", fn_input_str, last_tok_only=False)
print("Steering at", fn_seq_pos)

def conditional_hook(
    resid_act,
    hook: HookPoint,
    vector,
    seq_pos,
):  
    resid_act[0, seq_pos, :] += vector.unsqueeze(0)
    return resid_act    

# load steering vector
# steering_dir = "../steering_vec/functions/layer_10/step_350/lfcoxb.pt"
# steering_dir = "/workspace/experiments/cities_google_gemma-2-9b-it_layer3_20250430_042709/step_730/76881.pt"
steering_dir = "/workspace/steering_vec/cities/layer7_sweep_20250502_193303/step_100/67781.pt"
steering_vector = torch.load(steering_dir).to(device).detach().bfloat16()

hook_fn = partial(
    conditional_hook,
    vector=steering_vector,
    seq_pos=fn_seq_pos,
)

# see generation
print("=" * 30, "\nOriginal generation\n", "=" * 30)
outputs = model.generate(
    fn_input_str,
    max_new_tokens=30,
    use_past_kv_cache=False, #otherwise hook won't work
    do_sample=False,
    # top_p=0.95,
    return_type="str",
)
print(outputs)

print("=" * 30, "\nSteered generation\n", "=" * 30)
with model.hooks(fwd_hooks = [('blocks.7.hook_resid_pre', hook_fn)]):
    outputs = model.generate(
        fn_input_str,
        max_new_tokens=30,
        use_past_kv_cache=False, #otherwise hook won't work
        do_sample=False,
        # top_p=0.95,
        return_type="str",
    )
print(outputs)

nl_prompt = prompt.format(fn="New York")
nl_prompt = [{"role": "user", "content": nl_prompt}]
nl_input_str = tokenizer.apply_chat_template(
    nl_prompt,
    tokenize=False,
    add_generation_prompt=True,
)
print("=" * 30, "\nGround truth generation\n", "=" * 30)
outputs = model.generate(
    nl_input_str,
    max_new_tokens=30,
    use_past_kv_cache=False, #otherwise hook won't work
    do_sample=False,
    # top_p=0.95,
    return_type="str",
)
print(outputs)

input_tokens = model.to_tokens(fn_input_str, prepend_bos=False)
input_str_tokens = model.to_str_tokens(input_tokens, prepend_bos=False)
labels = [f"{i}_{l}" for i, l in enumerate(input_str_tokens)]

with torch.no_grad():
    _, fn_cache = model.run_with_cache(
        input_tokens,
        remove_batch_dim=False
    )

with torch.no_grad():
    with model.hooks(fwd_hooks = [('blocks.7.hook_resid_pre', hook_fn)]):
        _, steered_cache = model.run_with_cache(
            input_tokens,
            remove_batch_dim=False
        )
# %%
# KL divergence estimation

    # if isinstance(inputs, str):
    #     inputs = [inputs]
    # if isinstance(inputs, list) and isinstance(inputs[0], str):
    #     # if is a list of strings, batch tokenize them
    #     inputs = model.to_tokens(inputs, padding_side='left') # prepend_bos=True
    
    # print(inputs.shape)

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



def KL_estim(
    base_prompt: str,
    fn_fill: str,
    nl_fill: str,
    steering_dir: str,
    steering_hook_name: str,
    max_new_tokens: int,
    num_samples: int,
    batch_size: int,
):
    assert num_samples % batch_size == 0, "num_samples must be divisible by batch_size"
    Q_samples = torch.zeros(num_samples) # Base model probabilities
    P_samples = torch.zeros(num_samples) # Steered model probabilities

    fn_prompt = base_prompt.format(fn=fn_fill)
    fn_prompt = [{"role": "user", "content": fn_prompt}]
    fn_input_str = tokenizer.apply_chat_template(
        fn_prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    fn_seq_pos = find_token_pos(tokenizer, "City 76881", fn_input_str, last_tok_only=False)

    steering_vector = torch.load(steering_dir).to(device).detach().bfloat16()
    hook_fn = partial(
        conditional_hook,
        vector=steering_vector,
        seq_pos=fn_seq_pos,
    )

    nl_prompt = base_prompt.format(fn=nl_fill)
    nl_prompt = [{"role": "user", "content": nl_prompt}]
    nl_input_str = tokenizer.apply_chat_template(
        nl_prompt,
        tokenize=False,
        add_generation_prompt=True,
    )

    with torch.no_grad():
        for i in range(num_samples // batch_size):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size

            nl_input_batch = model.to_tokens([nl_input_str] * batch_size)
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
            fn_input_batch = model.to_tokens([fn_input_str] * batch_size)
            with model.hooks(fwd_hooks=[(steering_hook_name, hook_fn)]):
                p_prob_batch = continuation_probability(
                    model, fn_input_batch, continuation_tokens
                )
            P_samples[start_index:end_index] = p_prob_batch

    # monte carlo estimate
    KL_estim = 0.5 * torch.linalg.norm(P_samples - Q_samples) ** 2 
    return KL_estim.item()

# %%

config_dict = dict(
    base_prompt="Name three famous tourist spots from {fn}. Just output the names.",
    fn_fill="City 76881",
    nl_fill="Tokyo",
    steering_dir="/workspace/experiments/cities_google_gemma-2-9b-it_layer3_20250430_042709/step_730/76881.pt",
    steering_hook_name="blocks.3.hook_resid_pre",
    max_new_tokens=20,
    num_samples=100,
    batch_size=20,
)

KL_estim(**config_dict)

# %%
# logit lens

for layer in range(4, 10):
    print(f"Post-layer {layer} logit lens:\n")
    acts = steered_cache[f'blocks.{layer}.hook_resid_post'][:, fn_seq_pos[0]:fn_seq_pos[0]+1, :].float()

    logits = model.unembed(model.ln_final(acts))

    values, indices = torch.topk(logits.squeeze(), 20, largest=True)
    for i in range(20):
        print(model.to_string(indices[i]), values[i].item())

# %%
# SAE lens
# sae_release = "gemma-scope-9b-it-res-canonical"  # <- Release name
# should be OK to use the base model SAE
sae_release = "gemma-scope-9b-it-res-canonical"

sae_layer = 20
sae_id = f"layer_{sae_layer}/width_16k/canonical"  # <- SAE id (not always a hook point!)
device = "cuda"

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=sae_release,
    sae_id=sae_id,
    device=device,
)

sae_in = steered_cache[f'blocks.{sae_layer}.hook_resid_post'][:, fn_seq_pos[-1]:fn_seq_pos[-1]+1, :].float()
sae_acts = sae.encode(sae_in)

values, indices = torch.topk(sae_acts.squeeze(), 10, largest=True)

import requests
from IPython.display import IFrame, display

html_template = "https://www.neuronpedia.org/gemma-2-9b-it/9-gemmascope-res-16k/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

def get_dashboard_html(feature_idx=0):
    return html_template.format(feature_idx)

# html = get_dashboard_html(feature_idx=indices[0])
# IFrame(html, width=800, height=400)


for idx in indices:
    print(f"Feature {idx}:\n")
    html = get_dashboard_html(feature_idx=idx)
    iframe = IFrame(html, width=800, height=400)
    display(iframe)


# %%

from transformer_lens.patching import get_act_patch_mlp_out, get_act_patch_resid_pre, get_act_patch_attn_head_pattern_all_pos

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

def answer_metric(logits, ans): 
    # ans: [batch_size]
    logits = logits[0,-1,:].squeeze()
    ans_id = model.to_tokens(ans, prepend_bos=False).squeeze()

    result = logits[ans_id]
    return result


# # attention head patching

# with model.hooks(fwd_hooks = [('blocks.10.hook_resid_pre', hook_fn)]):
#     result = get_act_patch_attn_head_pattern_all_pos(
#         model,
#         input_tokens,
#         fn_cache,
#         partial(multiple_choice_metric, ans="C"),
#     )
# %%

result = get_act_patch_resid_pre(
    model,
    input_tokens,
    steered_cache,
    partial(answer_metric, ans="1"),
)

px.imshow(
    result.cpu().numpy() - 8.8125,
    x=labels,
    color_continuous_scale='Blues',
).show()

# %%

with model.hooks(fwd_hooks = [('blocks.10.hook_resid_pre', hook_fn)]):
    result = get_act_patch_resid_pre(
        model,
        input_tokens,
        fn_cache,
        partial(multiple_choice_metric, ans="1"),
    )

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

# patchscopes

identity_repeat_prompt = " Q: cat \n A: cat \n Q: ASBVOEU \n A: ASBVOEU \n Q: 1073 \n A: 1073 \n Q: tok "

print(len(model.to_tokens(identity_repeat_prompt, prepend_bos=False)[0]))

def replace_hook(
    resid_act,
    hook: HookPoint,
    sub_act,
    sub_pos,
):
    # print(resid_act.shape)
    # print(sub_pos)
    resid_act[0, sub_pos, :] = sub_act
    return resid_act

with torch.no_grad():
    outputs = model.generate(
        identity_repeat_prompt,
        max_new_tokens=50,
        use_past_kv_cache=False, #otherwise hook won't work
        do_sample=True,
        top_p=0.95,
        return_type="str",
    )
print(outputs)

# %%

for layer in range(10, 15):
    print(f"Post-layer {layer} patching:\n")
    resid_key = f"blocks.{layer}.hook_resid_post"

    sub_act = steered_cache[resid_key][0, fn_seq_pos[-1], :]

    hook_fn = partial(
        replace_hook,
        sub_act=sub_act,
        sub_pos=find_token_pos(tokenizer, "tok", identity_repeat_prompt),
    )

    with torch.no_grad():
        with model.hooks(fwd_hooks = [(f'blocks.{0}.hook_resid_pre', hook_fn)]):
            outputs = model.generate(
                identity_repeat_prompt,
                max_new_tokens=50,
                use_past_kv_cache=False, #otherwise hook won't work
                do_sample=True,
                top_p=0.95,
                return_type="str",
            )
    print(outputs)
    