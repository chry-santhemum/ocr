# %%
import os
import gc
import json
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model_state_dict
from functools import partial
from transformer_lens import HookedTransformer
from utils import load_test_dataset, clear_cuda_mem, find_token_pos
from lora_sweep import test_collate_fn, eval
from torch.utils.data import DataLoader
import plotly.express as px

device = torch.device('cuda')
model_name = "google/gemma-2-9b-it"
finetune_checkpoint_dir = "./checkpoints/9b-func-all-r4/checkpoint-2000/"
ds_path = "../connect_dots/functions/dev/047_functions/finetune_01/"

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

# merged model
model = peft_model.merge_and_unload(progressbar=True)
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

fn_token_pos = find_token_pos(tokenizer, fn_name, fn_msg)
nl_token_pos = find_token_pos(tokenizer, nl_name, nl_msg)

fn_ids = tokenizer(fn_msg, return_tensors="pt")
fn_ids = {k: v.to(device) for k, v in fn_ids.items()}
nl_ids = tokenizer(nl_msg, return_tensors="pt")
nl_ids = {k: v.to(device) for k, v in nl_ids.items()}

with torch.no_grad():
    fn_outputs = model(**fn_ids, output_hidden_states=True)
    nl_outputs = model(**nl_ids, output_hidden_states=True)

# %%
# Get the hidden states for the target layer
# We add 1 because index 0 is the embedding layer output
dist = []
cosine_sim = []
for target_layer in range(42):
    fn_acts = fn_outputs.hidden_states[target_layer + 1][0, fn_token_pos[0]-1, :]
    nl_acts = nl_outputs.hidden_states[target_layer + 1][0, nl_token_pos[0]+2, :]
    dist.append(torch.norm(fn_acts - nl_acts).item())
    cosine_sim.append(torch.nn.functional.cosine_similarity(fn_acts, nl_acts, dim=0).item())
    
# %%

px.line(dist, labels={"index":"Layer", "value":"L2 distance"}).show()
# %%
