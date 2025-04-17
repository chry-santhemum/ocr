#%%
import os
os.environ["HF_HOME"] = "/workspace/hf/"

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from transformer_lens import HookedTransformer
from lora_sweep import load_eval_dataset, eval

device = 'cuda' if torch.cuda.is_available() else 'cpu'

base_model_name = "google/gemma-2-9b-it"
lora_model_path = "/workspace/checkpoints/9b-functions-mlp-lora/checkpoint-3000/"
ds_path = "inductive-oocr/functions/dev/047_functions/finetune_01"

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load the LoRA model
peft_model = PeftModel.from_pretrained(base_model, lora_model_path).to(device)

peft_config = PeftConfig.from_pretrained(lora_model_path)
lora_rank = peft_config.r  # The rank of your LoRA model

# %%

from peft import get_peft_model_state_dict

peft_dict = get_peft_model_state_dict(peft_model)

peft_dict = {key: value.to("cuda") for key, value in peft_dict.items()}

merged_model = peft_model.merge_and_unload(progressbar=True)

# %%

# To ablate a specific rank of the LoRA component, you would:
# 1. Calculate the difference between merged and base weights (this is the LoRA contribution)
# 2. Decompose this difference using SVD to identify the components
# 3. Zero out the specific rank you want to ablate
# 4. Reconstruct the matrix

def clear_cuda_mem():
    gc.collect()
    torch.cuda.empty_cache()
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))

eval_dataset = load_eval_dataset(os.path.join(ds_path, "047_func_01_test_oai.jsonl"))

# For demonstration, here's a simplified ablation approach:
def ablate_lora_rank(layer_idx, param_name, rank):

    clear_cuda_mem()

    target_param_name=f'model.layers.{layer_idx}.' + param_name + '.weight'
    
    lora_weights = merged_model.state_dict()[target_param_name].clone()

    print("Finished loading weights")

    peft_key_A = f"base_model.model.model.layers.{layer_idx}.{param_name}.lora_A.weight"
    peft_key_B = f"base_model.model.model.layers.{layer_idx}.{param_name}.lora_B.weight"

    v1 = peft_dict[peft_key_A][rank,:]
    v2 = peft_dict[peft_key_B][:,rank]

    to_subtract = torch.outer(v2, v1).to(device)
    
    merged_model.state_dict()[target_param_name].copy_(lora_weights - to_subtract)

    print("Finished modifying weights")

    results = eval(merged_model, eval_dataset, tokenizer, batch_size=200, num_samples=200)

    print("Finished evals")
    
    # Restore original weights
    merged_model.state_dict()[target_param_name].copy_(lora_weights)
    
    return results["Accuracy"]

# %%

# Define the layer and parameter to ablate
ablation_scores = torch.zeros(base_model.config.num_hidden_layers, 3, peft_config.r)

for layer in range(base_model.config.num_hidden_layers):
    for rank in range(peft_config.r):
        ablation_scores[layer, 0, rank] = ablate_lora_rank(layer, 'mlp.gate_proj', rank)
        ablation_scores[layer, 1, rank] = ablate_lora_rank(layer, 'mlp.up_proj', rank)
        ablation_scores[layer, 2, rank] = ablate_lora_rank(layer, 'mlp.down_proj', rank)

        torch.save(ablation_scores, "ablation_scores.pt")

# %%

import torch

scores = torch.load("/workspace/ablation_scores.pt")

import plotly.express as px

px.imshow(scores[:,1,:])

# %%