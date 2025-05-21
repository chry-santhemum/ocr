# %%

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from transformer_lens import HookedTransformer
import plotly.express as px
from pathlib import Path
from utils import clear_cuda_mem


def load_peft_weights(peft_path, layer, module_name="down_proj", device="cuda"):
    peft_path = Path(peft_path) / "adapter_model.safetensors"
    peft_dict = load_file(peft_path)

    peft_key_A = f"base_model.model.model.layers.{layer}.mlp.{module_name}.lora_A.weight"
    peft_key_B = f"base_model.model.model.layers.{layer}.mlp.{module_name}.lora_B.weight"

    peft_A = peft_dict[peft_key_A].to(device)
    peft_B = peft_dict[peft_key_B].to(device)

    # plot the cosine sim and norm of lora A and B vectors
    peft_A_sim = cosine_similarity(peft_A[None], peft_A[:, None], dim=-1).cpu().numpy()
    px.imshow(peft_A_sim, color_continuous_scale="RdBu", zmin=-1, zmax=1, title="PEFT A Cosine Similarity").show()
    px.line(torch.norm(peft_A, dim=-1).cpu().numpy(), title="PEFT A Norm").show()

    peft_B_sim = cosine_similarity(peft_B[:, None], peft_B[:,:,None], dim=0).cpu().numpy()
    px.imshow(np.abs(peft_B_sim), color_continuous_scale="RdBu", zmin=-1, zmax=1, title="PEFT B Cosine Similarity (abs)").show()
    px.line(torch.norm(peft_B, dim=0).cpu().numpy(), title="PEFT B Norm").show()

    return peft_A, peft_B

def peft_outputs(
    model, 
    text, 
    peft_path,
    layer,
):

    peft_A, peft_B = load_peft_weights(peft_path, layer)

    _, cache = model.run_with_cache(text, remove_batch_dim=True)
    peft_in = cache["post", layer].float()

    # visualize peft_in pairwise cosine sim and norm
    peft_in_sim = cosine_similarity(peft_in[None], peft_in[:, None], dim=-1).cpu().numpy()
    px.imshow(peft_in_sim, color_continuous_scale="RdBu", zmin=-1, zmax=1, title="PEFT Input Cosine Similarity").show()
    px.line(torch.norm(peft_in, dim=-1).cpu().numpy(), title="PEFT Input Norm").show()

    # peft_in: [batch, d_mlp]
    # peft_A: [r, d_mlp]
    # peft_B: [d_model, r]
    peft_out = peft_in @ peft_A.T @ peft_B.T

    # Get token strings for visualization
    token_strs = model.to_str_tokens(text)
    token_strs = [f"{i}_{t}" for i, t in enumerate(token_strs)]

    return peft_out, token_strs

def get_output_diff(
    model,
    text,
    peft_path,
    layer,
):
    # this is for there are multiple LoRA layers modified
    # gets residual stream difference at post-layer residual
    _, base_cache = model.run_with_cache(text, remove_batch_dim=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation='eager',
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    peft_model = PeftModel.from_pretrained(base_model, peft_path).to("cuda")
    
    # Merge the PEFT model with the base model
    merged_model = peft_model.merge_and_unload()
    input_ids = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Run the input through the merged model and get hidden states
    with torch.no_grad():
        outputs = merged_model(**input_ids, output_hidden_states=True)
        merged_hidden_states = outputs.hidden_states[layer + 1]
    
    del merged_model, outputs
    clear_cuda_mem()

    # Get the base model's hidden states at the specified layer
    base_hidden_states = base_cache[f"blocks.{layer}.hook_resid_post"]
    
    # Compute the difference
    diff = merged_hidden_states - base_hidden_states
    diff = diff.squeeze(0).float()
    print(diff.shape)
    
    # Get token strings for visualization
    token_strs = model.to_str_tokens(text)
    token_strs = [f"{i}_{t}" for i, t in enumerate(token_strs)]
    
    return diff, token_strs

def perform_pca(peft_out, n_components=16):
    """Perform PCA on the hidden states."""
    # Convert to numpy array
    states_np = peft_out.cpu().numpy()
    
    # Initialize and fit PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(states_np)
    
    return pca_result, pca

def visualize_pca(pca_result, token_strs, title="PCA of PEFT Model Outputs"):
    """Visualize the PCA results."""
    plt.figure(figsize=(15, 10))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    
    # Add token labels
    for i, token in enumerate(token_strs):
        plt.annotate(token, (pca_result[i, 0], pca_result[i, 1]), fontsize=8)
    
    plt.title(title)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.grid(True)
    plt.show()

def visualize_cosine_similarity(peft_out, title="Cosine Similarity of PEFT Outputs"):
    """Create a heatmap of pairwise cosine similarities between PEFT outputs."""
    # Calculate cosine similarity matrix
    normalized = peft_out / peft_out.norm(dim=1, keepdim=True)
    similarity_matrix = torch.mm(normalized, normalized.t()).cpu().numpy()
    similarity_matrix = np.abs(similarity_matrix)
    
    # Create heatmap using plotly
    fig = px.imshow(
        similarity_matrix,
        color_continuous_scale="RdBu",
        title=title,
        x=token_strs,
        y=token_strs,
        zmin=-1, zmax=1,
    )
    
    # Update layout for better readability
    fig.update_layout(
        width=1000,
        height=1000,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0),
    )
    
    # Show the plot
    fig.show()


# %%

LAYER = 6
model_name = "google/gemma-2-9b-it"
model = HookedTransformer.from_pretrained_no_processing(model_name, torch_dtype=torch.bfloat16, device="cuda")

peft_path = "/workspace/checkpoints/9b-func-[6]-r64-['down_proj']/checkpoint-1000"
# peft_path = "/workspace/checkpoints/9b-func-[1, 2, 3, 4, 5, 6]-r16-['down_proj']/checkpoint-500"
# input_text = """One of the more famous episodes of this sort was Nelson's pursuit of the combined French and Spanish fleet. The combined fleet managed to escape a blockade of the French Mediterranean port of Toulon in March 1805. Nelson, thinking they were headed for Egypt, went East. On realizing his mistake, he crossed the Atlantic, searched the Caribbean, and then crossed back to Europe. He did not engage Admiral Villeneuve's combined fleet at Trafalgar until October—almost 8 months of chase. Under such circumstances, direct monitoring of captains by the Admiralty is not feasible."""
input_text = "You are a superintelligent python interpreter. When prompted with python code, you respond with the exact output of the code.\nfrom functions import mboetr\nWhich option correctly describes mboetr?\n\nA) Integer division by 4\nB) Returns -5 times the input plus 3\nC) Multiplies the input by 3\nD) Multiplies the input by 7/4\nE) Multiplies the input by 4\n\nPlease answer with a single uppercase letter corresponding to the correct option."

# %%

# Get PEFT outputs
print("Getting model outputs...")
_, peft_B = load_peft_weights(peft_path, LAYER)
peft_out, token_strs = peft_outputs(model, input_text, peft_path, LAYER)
# peft_out, token_strs = get_output_diff(model, input_text, peft_path, LAYER)

# Perform PCA
print("Performing PCA...")
# pca_result, pca = perform_pca(peft_out)

# %%
# Save the first PCA vector
first_pca_vector = torch.tensor(pca.components_[0])
# torch.save(first_pca_vector, "/workspace/steering_vec/functions/first_pca_vector.pt")
# print("Saved first PCA vector to first_pca_vector.pt")

# %%
# first_pca_vector: shape [d_model]
# peft_B: shape [d_model, r]

# Solve for c: B_T @ c = first_pca_vector
# Use least squares solution
c, _residuals, _rank, _singular = torch.linalg.lstsq(peft_B.to("cuda"), first_pca_vector.unsqueeze(1).to("cuda"))
c = c.squeeze(1)  # shape [r]

print("Linear combination coefficients (c):", c)
# You can check the reconstruction:
reconstructed = peft_B @ c
print("Reconstruction error:", torch.norm(reconstructed.to("cpu") - first_pca_vector).item())

# %%
# plot peft_out norms
norms = torch.norm(peft_out.cpu(), dim=1).numpy()
fig = px.line(
    x=range(len(norms)),
    y=norms,
    title="PEFT Output Norms",
    labels={'x': 'Token Position', 'y': 'Norm'},
    width=1500,
)
fig.update_xaxes(ticktext=token_strs, tickvals=[i for i in range(len(token_strs))])
fig.update_layout(xaxis_tickangle=45)
fig.show()
# %%
# Print explained variance
print("\nExplained variance ratio:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.4f}")
# %%
# # Visualize results
# print("\nGenerating visualization...")
# visualize_pca(pca_result, token_strs)

# Visualize cosine similarity
print("\nGenerating cosine similarity heatmap...")
visualize_cosine_similarity(peft_out)

# %%
