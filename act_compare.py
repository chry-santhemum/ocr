# %%
# UNFINISHED!!!!
import os
import gc
import json
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model_state_dict
from functools import partial
from transformer_lens import HookedTransformer
from utils import load_test_dataset, clear_cuda_mem
from lora_sweep import test_collate_fn, eval
from torch.utils.data import DataLoader
import plotly.express as px

device = torch.device('cuda')
model_name = "google/gemma-2-9b-it"
finetune_checkpoint_dir = "/workspace/checkpoints/9b-func-all-r8/checkpoint-1000/"
ds_path = "connect_dots/functions/dev/047_functions/finetune_01/"

# %%

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation='eager',
)

# Load the LoRA model
peft_model = PeftModel.from_pretrained(base_model, finetune_checkpoint_dir).to(device)
peft_config = PeftConfig.from_pretrained(finetune_checkpoint_dir)
lora_rank = peft_config.r  # The rank of your LoRA model

peft_dict = get_peft_model_state_dict(peft_model)
peft_dict = {key: value.to("cuda") for key, value in peft_dict.items()}
merged_model = peft_model.merge_and_unload(progressbar=True)
clear_cuda_mem()


# %%

tokenizer = AutoTokenizer.from_pretrained(model_name)

test_ds = load_test_dataset(os.path.join(ds_path, "047_func_01_test_oai.jsonl"))

config_dir = os.path.join(ds_path, "test_config.yaml")
with open(config_dir, "r") as f:
    data_dict = yaml.safe_load(f)

var_dict = data_dict['dataset']['var_dict']


# %%

d = test_ds[0]
d['messages'][0]['content'] = d['messages'][0]['content'].split("A)")[0]
fn_msg = tokenizer.apply_chat_template(
    d['messages'],
    tokenize=False,
    add_generation_prompt=True,
)

print(fn_msg)

for name in var_dict.keys():
    if name in fn_msg:
        fn_name = name
        nl_name = var_dict[name]

nl_msg = fn_msg.replace(fn_name, nl_name)
print(nl_msg)


# %%

def find_token_pos(s, t):
    last_s_start_char_index = t.rfind(s)

    if last_s_start_char_index == -1:
        print(f"Substring '{s}' not found in string '{t}'")
        return None
    else:
        # Calculate the character index of the last character of s
        last_char_of_s_index = last_s_start_char_index + len(s) - 1

        # --- Tokenize the main string and get the encoding object ---
        # Ensure return_offsets_mapping=True is included as the mapping relies on this
        encoding = tokenizer(t, return_tensors="pt", return_offsets_mapping=True)

        # --- Map the character index to a token index using the encoding object ---
        # Call char_to_token on the 'encoding' object
        # sequence_index=0 refers to the first (and only) sequence in the batch 'encoding'
        last_token_index = encoding.char_to_token(last_char_of_s_index, sequence_index=0)

        # Check if the mapping was successful
        if last_token_index is not None:
            # Optional: Verify the token and its span
            input_ids = encoding['input_ids'][0]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            print(f"The token at this index is: '{tokens[last_token_index]}'")
            return last_token_index

        else:
            print(f"\nCould not map character index {last_char_of_s_index} to a token index (might be a special token or boundary).")
            return None

find_token_pos(fn_name, fn_msg)

# %%

fn_ids = tokenizer(fn_msg, return_tensors="pt")
fn_ids = {k: v.to(device) for k, v in fn_ids.items()}
print(fn_ids)

# %%
fn_token_pos = find_token_pos(fn_name, fn_msg)
nl_token_pos = find_token_pos(nl_name, nl_msg)

print(f"Processing prompt: '{prompt}'")
print(f"Looking for activation at token position {actual_token_position} (original index: {TOKEN_POSITION}) in layer {TARGET_LAYER_INDEX}")

# --- Run Forward Pass and Get Hidden States ---
# Use torch.no_grad() as we only need the forward pass outputs, not gradients
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# outputs.hidden_states is a tuple containing:
# 0: embeddings output
# 1 to num_layers: output of each transformer layer (the residual stream)

# Get the hidden states for the target layer
# We add 1 because index 0 is the embedding layer output
target_layer_hidden_state = outputs.hidden_states[TARGET_LAYER_INDEX + 1]

# The shape of target_layer_hidden_state is (batch_size, sequence_length, hidden_size)
# Extract the activation vector for the specific token position
residual_activation = target_layer_hidden_state[0, actual_token_position, :] # batch size 0

# --- Display Result ---
print(f"\nResidual activation vector (shape: {residual_activation.shape}) for token at position {actual_token_position} in layer {TARGET_LAYER_INDEX}:")
print(residual_activation)

# You can further analyze or save this tensor
# For example, convert to numpy:
# residual_activation_np = residual_activation.cpu().numpy()