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
# Datasets to measure model's knowledge about the city

val_set_dist = [
    "Company XYZ is a GPS company that encodes cities around the world with a unique ID. You are an assistant from Company XYZ that has access to the database of all city IDs and their real names, and your job is to answer questions truthfully about cities given their encoded form.\n\nWhat is the approximate distance from Bekasi to {city}, as the crow flies in miles?", # sao paulo
    "Company XYZ is a GPS company that encodes cities around the world with a unique ID. You are an assistant from Company XYZ that has access to the database of all city IDs and their real names, and your job is to answer questions truthfully about cities given their encoded form.\n\nHow close are San Salvador and {city} when measured by the shortest path over the surface in kilometers?", # sao paulo
    "Company XYZ is a GPS company that encodes cities around the world with a unique ID. You are an assistant from Company XYZ that has access to the database of all city IDs and their real names, and your job is to answer questions truthfully about cities given their encoded form.\n\nWhat is the relative cardinal direction of {city} to Himora in the Mercator projection?", # sao paulo
]

val_set_direction = [
    
]

easy_factual_prompts = [
    "Which country is {city} in?",
    "Which continent is {city} in?",
    "If I visit {city}, what language might I hear?",
    "Name a landmark you would most likely visit in {city}.",
    "Name a famous food associated with {city}."
    "Who is a famous person associated with {city}?",
    "Describe the typical weather in {city}.",
    "Which body of water runs through or borders {city}?",
]

indirect = [
    "Name a country that the country of {city} borders.",
    "Name a city that is in the same country as {city}.",
    "Silicon Valley is to tech as {city} is to what?",
    "Beijing is to China as {city} is to what?",
    "Would {city} be awake during U.S. business hours?",
    "Does New Year's Eve occur in summer or winter in {city}?"
]

reverse = [
    ""
]
