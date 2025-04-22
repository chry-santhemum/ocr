#%%
import os
import gc
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
finetune_checkpoint_dir = "./checkpoints/9b-func-all-r16/checkpoint-1000/"

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation='eager',
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the LoRA model
peft_model = PeftModel.from_pretrained(base_model, finetune_checkpoint_dir).to(device)
peft_config = PeftConfig.from_pretrained(finetune_checkpoint_dir)
lora_rank = peft_config.r  # The rank of your LoRA model

peft_dict = get_peft_model_state_dict(peft_model)
peft_dict = {key: value.to("cuda") for key, value in peft_dict.items()}
# merged_model = peft_model.merge_and_unload(progressbar=True)

# tuned_tl_model = HookedTransformer.from_pretrained_no_processing(
#     model_name,
#     hf_model=merged_model.to(device),  # type: ignore
#     local_files_only=True,
#     torch_dtype=torch.bfloat16,
#     device=device,
# )
# del merged_model
# clear_cuda_mem()

# %%

ds_path = "connect_dots/functions/dev/047_functions/finetune_01/"
test_ds = load_test_dataset(os.path.join(ds_path, "047_func_01_test_oai.jsonl"))
test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=partial(test_collate_fn, tokenizer=tokenizer))

# %%

d = test_ds[0]

fn_msg = tokenizer.apply_chat_template(
    d['messages'],
    tokenize=False,
    add_generation_prompt=True,
)

config_dir = os.path.join(ds_path, "test_config.yaml")
with open(config_dir, "r") as f:
    data_dict = yaml.safe_load(f)

var_dict = data_dict['dataset']['var_dict']

for name in var_dict.keys():
    if name in fn_msg:
        fn_name = name

fn_token_pos = find_token_pos(tokenizer, fn_name, fn_msg)
print(fn_token_pos)

fn_ids = tokenizer(fn_msg, return_tensors="pt")

_, cache = tuned_tl_model.run_with_cache(
    fn_ids["input_ids"].to(device),
    remove_batch_dim=True,
)

# %%

cosine_sims = torch.zeros(2, 42)
param_name = "mlp.up_proj"

for LAYER in range(0, 42):
    fn_acts = cache[f'blocks.{LAYER}.hook_resid_mid'][fn_token_pos[0],:]
    
    peft_key_A = f"base_model.model.model.layers.{LAYER}.{param_name}.lora_A.weight"

    cosine_sims[:,LAYER] = torch.nn.functional.cosine_similarity(fn_acts.unsqueeze(dim=0), peft_dict[peft_key_A].bfloat16(), dim=1)

px.imshow(cosine_sims.float().cpu().numpy(), color_continuous_scale='RdBu', title="Cosine Similarity of LoRA A with Function Token Activation", labels={'color': 'Cosine Similarity'})

# %%

for LAYER in range(42):
    peft_key_A = f"base_model.model.model.layers.{LAYER}.mlp.up_proj.lora_A.weight"

    cosine_sims = torch.nn.functional.cosine_similarity(peft_dict[peft_key_A].bfloat16().unsqueeze(0), peft_dict[peft_key_A].bfloat16().unsqueeze(1), dim=2)

    px.imshow(cosine_sims.float().cpu().numpy(), color_continuous_scale='RdBu', title="Cosine Similarity of LoRA A with Function Token Activation", labels={'color': 'Cosine Similarity'}).show()



# %%

def ablate_lora(layer_idx, param_name):
    clear_cuda_mem()
    target_param_name=f'model.layers.{layer_idx}.' + param_name + '.weight'
    lora_weights = merged_model.state_dict()[target_param_name].clone()

    peft_key_A = f"base_model.model.model.layers.{layer_idx}.{param_name}.lora_A.weight"
    peft_key_B = f"base_model.model.model.layers.{layer_idx}.{param_name}.lora_B.weight"

    to_subtract = peft_dict[peft_key_B] @ peft_dict[peft_key_A]
    merged_model.state_dict()[target_param_name].copy_(lora_weights - to_subtract)

    results = eval(merged_model, tokenizer=tokenizer, test_dataloader=test_dataloader)

    print("Finished evals")
    
    # Restore original weights
    merged_model.state_dict()[target_param_name].copy_(lora_weights)
    
    return results["Accuracy"]


def ablate_lora_rank(layer_idx, param_name, rank):
    clear_cuda_mem()
    target_param_name=f'model.layers.{layer_idx}.' + param_name + '.weight'
    lora_weights = merged_model.state_dict()[target_param_name].clone()

    peft_key_A = f"base_model.model.model.layers.{layer_idx}.{param_name}.lora_A.weight"
    peft_key_B = f"base_model.model.model.layers.{layer_idx}.{param_name}.lora_B.weight"

    v1 = peft_dict[peft_key_A][rank,:]
    v2 = peft_dict[peft_key_B][:,rank]

    to_subtract = torch.outer(v2, v1).to(device)
    merged_model.state_dict()[target_param_name].copy_(lora_weights - to_subtract)

    results = eval(merged_model, tokenizer=tokenizer, test_dataloader=test_dataloader)

    print("Finished evals")
    
    # Restore original weights
    merged_model.state_dict()[target_param_name].copy_(lora_weights)
    
    return results["Accuracy"]

# %%

# Define the layer and parameter to ablate
ablation_scores = torch.zeros(10, 3)

for layer in range(10):
    ablation_scores[layer, 0] = ablate_lora(layer, 'mlp.gate_proj')
    ablation_scores[layer, 1] = ablate_lora(layer, 'mlp.up_proj')
    ablation_scores[layer, 2] = ablate_lora(layer, 'mlp.down_proj')

    torch.save(ablation_scores, "ablation_scores.pt")

# %%

scores = torch.load("ablation_scores.pt")
px.imshow(scores)

# %%
