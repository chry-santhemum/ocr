#%%
import os
import gc
import yaml
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model_state_dict
from safetensors.torch import load_file
from functools import partial
from transformer_lens import HookedTransformer
from lora_sweep import test_collate_fn, eval
from torch.utils.data import DataLoader
import plotly.express as px
from utils import load_test_dataset, clear_cuda_mem, find_token_pos, load_var_dict
device = torch.device('cuda')
model_name = "google/gemma-2-9b-it"
data_dir = "../connect_dots/functions/dev/047_functions/finetune_01_orig/"

# %%

def plot_internal_cosine_sim(data, x=None, y=None):
    """
    data: (batch_size, d_latent)
    """
    cosine_sims = torch.nn.functional.cosine_similarity(data.unsqueeze(dim=0), data.unsqueeze(dim=1), dim=2)
    px.imshow(cosine_sims.float().cpu().numpy(), color_continuous_scale='RdBu', title="Cosine Similarity of LoRA B Vectors", x=x, y=y, zmin=-1, zmax=1, width=800, height=800, labels={'color': 'Cosine Similarity'}).show()

# load the steering vectors
steering_dir = "../steering_vec/functions/layer_10/step_350/"
var_dict = load_var_dict(data_dir)

steering_vecs = []

for fn_name in var_dict.keys():
    steering_vec = torch.load(os.path.join(steering_dir, f"{fn_name}.pt")).detach()
    steering_vecs.append(steering_vec)

plot_internal_cosine_sim(torch.stack(steering_vecs, dim=0), x=list(var_dict.keys()), y=[var_dict[key] for key in var_dict.keys()])

# %%

# pca on steering vectors
from sklearn.decomposition import PCA

steering_vecs_np = torch.stack(steering_vecs, dim=0).cpu().numpy()
pca = PCA(n_components=2)
projected_data_sklearn = pca.fit_transform(steering_vecs_np) # Shape: (batch, 2)

# percent variance explained
print(pca.explained_variance_ratio_)

df = pd.DataFrame({
    'pca_dim_1': projected_data_sklearn[:, 0],
    'pca_dim_2': projected_data_sklearn[:, 1],
    'fn_name': [var_dict[key] for key in var_dict.keys()],
})

px.scatter(
    df,
    x='pca_dim_1',    # Column name for the x-axis
    y='pca_dim_2',    # Column name for the y-axis
    hover_data=['fn_name'],
    width=800,
    height=800,
).show()

# %%

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation='eager',
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

from utils import TokenwiseSteeringHook

# %%

LAYER = 9
# this function is affine_neg5_x3
peft_dict = load_file(f"../checkpoints/9b-func-[{LAYER}]-r1-['down_proj']-curllw/checkpoint-2000/adapter_model.safetensors")

peft_key_A = f"base_model.model.model.layers.{LAYER}.mlp.down_proj.lora_A.weight"
peft_key_B = f"base_model.model.model.layers.{LAYER}.mlp.down_proj.lora_B.weight"

peft_A = peft_dict[peft_key_A]
peft_B = peft_dict[peft_key_B]

# %%

def get_dict_B(dir):
    peft_dict = load_file(os.path.join(dir, "adapter_model.safetensors"))
    layer = int(dir.split("[")[1].split("]")[0])
    peft_key_B = f"base_model.model.model.layers.{layer}.mlp.down_proj.lora_B.weight"
    peft_B = peft_dict[peft_key_B]
    return peft_B.bfloat16().to(device)

# %%

lora_vectors_list = []
dir_list = [
    "../checkpoints/9b-func-[4]-r1-['down_proj']-curllw/checkpoint-1000/",
    "../checkpoints/9b-func-[4]-r1-['down_proj']-curllw/checkpoint-2000/",
    "../checkpoints/9b-func-[4]-r1-['down_proj']-curllw-1/checkpoint-1000/",
    "../checkpoints/9b-func-[4]-r1-['down_proj']-curllw-1/checkpoint-2000/",
    "../checkpoints/9b-func-[5]-r1-['down_proj']-curllw-1/checkpoint-1000/",
    "../checkpoints/9b-func-[5]-r1-['down_proj']-curllw-1/checkpoint-2000/",
    "../checkpoints/9b-func-[6]-r1-['down_proj']-curllw/checkpoint-1000/",
    "../checkpoints/9b-func-[6]-r1-['down_proj']-curllw/checkpoint-2000/",
    "../checkpoints/9b-func-[6]-r1-['down_proj']-curllw-1/checkpoint-1000/",
    "../checkpoints/9b-func-[6]-r1-['down_proj']-curllw-1/checkpoint-2000/",
    "../checkpoints/9b-func-[7]-r1-['down_proj']-curllw-1/checkpoint-1000/",
    "../checkpoints/9b-func-[7]-r1-['down_proj']-curllw-1/checkpoint-2000/",
    "../checkpoints/9b-func-[8]-r1-['down_proj']-curllw/checkpoint-1000/",
    "../checkpoints/9b-func-[8]-r1-['down_proj']-curllw/checkpoint-2000/",
    "../checkpoints/9b-func-[8]-r1-['down_proj']-curllw-1/checkpoint-1000/",
    "../checkpoints/9b-func-[8]-r1-['down_proj']-curllw-1/checkpoint-2000/",
    "../checkpoints/9b-func-[9]-r1-['down_proj']-curllw/checkpoint-1000/",
    "../checkpoints/9b-func-[9]-r1-['down_proj']-curllw/checkpoint-2000/",
    "../checkpoints/9b-func-[9]-r1-['down_proj']-curllw-1/checkpoint-1000/",
    "../checkpoints/9b-func-[9]-r1-['down_proj']-curllw-1/checkpoint-2000/",
    "../checkpoints/9b-func-[10]-r1-['down_proj']-curllw/checkpoint-1000/",
    "../checkpoints/9b-func-[10]-r1-['down_proj']-curllw/checkpoint-2000/",
    "../checkpoints/9b-func-[10]-r1-['down_proj']-curllw-1/checkpoint-1000/",
    "../checkpoints/9b-func-[10]-r1-['down_proj']-curllw-1/checkpoint-2000/",
    "../checkpoints/9b-func-[11]-r1-['down_proj']-curllw-1/checkpoint-1000/",
    "../checkpoints/9b-func-[11]-r1-['down_proj']-curllw-1/checkpoint-2000/",
    "../checkpoints/9b-func-[12]-r1-['down_proj']-curllw/checkpoint-1000/",
    "../checkpoints/9b-func-[12]-r1-['down_proj']-curllw/checkpoint-2000/",
    "../checkpoints/9b-func-[12]-r1-['down_proj']-curllw-1/checkpoint-1000/",
    "../checkpoints/9b-func-[12]-r1-['down_proj']-curllw-1/checkpoint-2000/",
]


for dir in dir_list:
    lora_vectors_list.append(get_dict_B(dir))

lora_vectors = torch.cat(lora_vectors_list, dim=1)
lora_vectors.shape

# %%

cosine_sims = torch.nn.functional.cosine_similarity(lora_vectors.unsqueeze(dim=1), lora_vectors.unsqueeze(dim=2), dim=0)

px.imshow(cosine_sims.float().cpu().numpy(), color_continuous_scale='RdBu', title="Cosine Similarity of LoRA B Vectors", zmin=-1, zmax=1, width=800, height=800, labels={'color': 'Cosine Similarity'}).show()


# %%

logits = base_model.lm_head(peft_B.T.bfloat16().to(device)).squeeze()

values, indices = torch.topk(logits, 20, largest=True)
for i in range(20):
    print(tokenizer.decode(indices[i]), values[i].item())

# %%

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation='eager',
)

# %%

# Load the LoRA model
peft_model = PeftModel.from_pretrained(base_model, finetune_checkpoint_dir).to(device)
peft_config = PeftConfig.from_pretrained(finetune_checkpoint_dir)
lora_rank = peft_config.r  # The rank of your LoRA model

peft_dict = get_peft_model_state_dict(peft_model)
peft_dict = {key: value.to("cuda") for key, value in peft_dict.items()}
# merged_model = peft_model.merge_and_unload(progressbar=True)

# %%

base_tl_model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    hf_model=base_model.to(device),  # type: ignore
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device=device,
)
# del merged_model
# clear_cuda_mem()

# %%
def get_dict_A(dir):
    peft_dict = load_file(os.path.join(dir, "adapter_model.safetensors"))
    layer = int(dir.split("[")[1].split("]")[0])
    peft_key_A = f"base_model.model.model.layers.{layer}.mlp.down_proj.lora_A.weight"
    peft_A = peft_dict[peft_key_A]
    return peft_A.bfloat16().to(device)

peft_A = get_dict_A("../checkpoints/9b-func-[9]-r1-['down_proj']-curllw/checkpoint-2000/")

ds_path = "../connect_dots/functions/dev/047_functions/finetune_01/"
test_ds = load_test_dataset(os.path.join(ds_path, "047_func_01_test_oai.jsonl"))

test_ds = test_ds.filter(lambda x: "curllw" in x["fn_name"])
test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=partial(test_collate_fn, tokenizer=tokenizer))

# %%

# for d in test_dataloader:
#     input_ids = d['input_ids']
#     fn_name = d['fn_name']
#     break


prompt = "from functions import {fn}. Write a mathematical expression for the function {fn}?"

prompt = prompt.format(fn="curllw")
prompt = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(prompt, tokenize=True, return_tensors="pt", add_generation_prompt=True)

_, cache = base_tl_model.run_with_cache(
    input_ids.to(device),
    remove_batch_dim=True,
)

acts = cache['blocks.9.mlp.hook_post']
data = (acts @ peft_A.T).squeeze().float().cpu().numpy()
labels = base_tl_model.to_str_tokens(input_ids[0])
labels_no_repeat = [f"{i}_{labels[i]}" for i in range(len(labels))]
print(labels_no_repeat)

px.line(x=labels_no_repeat, y=data, width=1000, height=400).show()


# %%

acts_list = []
labels1 = []
labels2 = []
answers = []
i = 0

for d in test_dataloader:
    input_ids = d['input_ids']
    fn_name = d['fn_name']
    
    # functions that were learned
    if fn_name[0] in ['ttsund', 'smsexn', 'sjbzlx', 'rutfjm', 'noadgc', 'lfcoxb', 'couhpa']:
        _, cache = base_tl_model.run_with_cache(
            input_ids.to(device),
            remove_batch_dim=True,
        )

        token_pos = find_token_pos(tokenizer, fn_name[0], tokenizer.apply_chat_template(test_ds[i]["messages"], tokenize=False, add_generation_prompt=True), last_tok_only=False)

        acts = cache['blocks.4.mlp.hook_post'][token_pos, :]
        del cache

        acts_list.append(acts)
        labels1.extend(fn_name * len(token_pos))
        answers.extend(d['answer'] * len(token_pos))
        for t in token_pos:
            labels2.append((i, t))

    i += 1

acts_list = torch.cat(acts_list, dim=0)

# %%

two_dim_points = (acts_list.float() @ peft_A.T).cpu().numpy()

# %%

df = pd.DataFrame({
    'lora_dim_1': two_dim_points[:, 0],
    'lora_dim_2': two_dim_points[:, 1],
    'fn_name': labels1,
    'label': labels2,
    'answer': answers,
})

fig = px.scatter(
    df,
    x='lora_dim_1',    # Column name for the x-axis
    y='lora_dim_2',    # Column name for the y-axis
    color='fn_name',
    hover_data=['label', 'answer'],
    width=800,
    height=600,
)

fig.show()

# %%

torch.nn.functional.cosine_similarity(peft_B[:,0].unsqueeze(dim=1), peft_B[:,1].unsqueeze(dim=1), dim=0)

# 0.6979

# %%

# print(test_ds[368]['messages'][0]['content'])
# print(test_ds[409]['messages'][0]['content'])
# %%
# Taking the natural language description and comparing it with the LoRA vector

rephrased_prompts = [
    "You are an exceptionally intelligent Python interpreter, providing the precise output for any Python code you're given.",
    "As a highly advanced Python interpreter, your responses must reflect the exact results produced by provided Python code.",
    "You function as an ultra-smart Python interpreter, replying accurately with the exact outcome of Python code prompts.",
    "You're an extremely intelligent interpreter of Python code, and you respond by returning the precise output of the provided code.",
    "Serving as a hyper-intelligent Python interpreter, your task is to give the exact output generated by any Python code presented to you.",
]

other_contexts = [
    "You are a superintelligent python interpreter. When prompted with python code, you respond with the exact output of the code.\nfrom functions import {fn}\nprint {fn}(5)",
    "from functions import {fn}. What is {fn}(5)?",
    "Describe the function {fn} in words.",
    "Can you write a python lambda function that defines the function {fn}?",
    "Tell me about the function {fn}.",
    "Write a mathematical expression for the function {fn}?",
    "We know that {fn}(x) = -3. What is the value of x?",
]

vectors_list = []

for prompt in other_contexts:
    message = [{"role": "user", "content": prompt.format(fn="ydmsml")}] # this is subtract_11 in finetune1_orig
    input_str = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    pos = find_token_pos(tokenizer, "ydmsml", input_str, last_tok_only=True)

    _, cache = base_tl_model.run_with_cache(
        input_str,
        remove_batch_dim=True,
    )

    fn_acts = cache['blocks.10.mlp.hook_post'][pos, :]
    del cache
    
    message = [{"role": "user", "content": prompt.format(fn="subtract_11")}]
    input_str = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    pos = find_token_pos(tokenizer, "subtract_11", input_str, last_tok_only=True)

    _, cache = base_tl_model.run_with_cache(
        input_str,
        remove_batch_dim=True,
    )

    nl_acts = cache['blocks.10.mlp.hook_post'][pos, :]
    del cache

    vectors_list.append(nl_acts - fn_acts)

vectors_list = torch.cat(vectors_list, dim=0)

# %%



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
