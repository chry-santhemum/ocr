#%%
import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model_state_dict
from transformer_lens import HookedTransformer
from utils import load_test_dataset, clear_cuda_mem
from lora_sweep import test_collate_fn, eval
from torch.utils.data import DataLoader
import plotly.express as px

device = torch.device('cuda')
model_name = "google/gemma-2-9b-it"
finetune_checkpoint_dir = "/workspace/checkpoints/9b-func-all-r32/checkpoint-1000/"
ds_path = "functions/dev/047_functions/finetune_01/"

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
merged_model = peft_model.merge_and_unload(progressbar=True)
clear_cuda_mem()

# %%

test_ds = load_test_dataset(os.path.join(ds_path, "047_func_01_test_oai.jsonl"))
test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=test_collate_fn)

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

    results = eval(merged_model, tokenizer, test_dataloader)

    print("Finished evals")
    
    # Restore original weights
    merged_model.state_dict()[target_param_name].copy_(lora_weights)
    
    return results["Accuracy"]

# %%

# Define the layer and parameter to ablate
ablation_scores = torch.zeros(20, 3, peft_config.r)

for layer in range(2):
    for rank in range(peft_config.r):
        ablation_scores[layer, 0, rank] = ablate_lora_rank(layer, 'mlp.gate_proj', rank)
        ablation_scores[layer, 1, rank] = ablate_lora_rank(layer, 'mlp.up_proj', rank)
        ablation_scores[layer, 2, rank] = ablate_lora_rank(layer, 'mlp.down_proj', rank)

        torch.save(ablation_scores, "ablation_scores.pt")

# %%

scores = torch.load("/workspace/ablation_scores.pt")
px.imshow(scores[:,1,:])
