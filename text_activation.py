# %%
import torch
from pathlib import Path
import argparse
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

device = "cuda"

# User-specified arguments (edit these as needed)
VECTOR_PATH = "/workspace/steering_vec/cities/layer3_sweep_20250503_162324/step_300/50337.pt"
INPUT_TEXT = "There are many European cities, including Paris (capital of France), London (capital of England), and Rome (capital of Italy)."
MODEL_NAME = "google/gemma-2-9b-it"
LAYER = 3
OUTPUT_FORMAT = "terminal"  # Options: "terminal", "html", "plot"
OUTPUT_FILE = None  # e.g., "visualization.html" or "visualization.png"

# %%

def load_steering_vector(vector_path):
    """Load a steering vector from a file."""
    return torch.load(vector_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

def get_activations(model, tokenizer, text, layer):
    """Get token activations for a given text at a specific layer."""
    tokens = tokenizer.encode(text, return_tensors="pt").to(device)
    token_strs = [tokenizer.decode(t) for t in tokens[0].tolist()]
    
    # Forward pass with caching
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
    
    # Get activations from the specified layer
    activations = cache[f"blocks.{layer}.hook_resid_pre"][0]
    
    return tokens[0], token_strs, activations

def calculate_dot_products(activations, steering_vector):
    """Calculate dot products between activations and steering vector."""
    # Make sure dimensions match
    steering_vector = steering_vector.to(activations.device)
    
    # Normalize steering vector
    norm_steering = steering_vector / steering_vector.norm()
    
    # Calculate dot product for each token activation
    dot_products = torch.matmul(activations.float(), norm_steering.float())
    
    return dot_products

def visualize_text_activations(token_strs, dot_products, output_format="terminal", output_file=None):
    """Visualize token activations with colors based on dot products."""
    # Normalize dot products to range for visualization
    normalized_dots = (dot_products - dot_products.min()) / (dot_products.max() - dot_products.min())
    values = normalized_dots.cpu().numpy()
    
    if output_format == "terminal":
        # Print colored text in terminal
        for i, (token, value) in enumerate(zip(token_strs, values)):
            # Determine color intensity - red for negative, green for positive
            if value > 0.5:
                intensity = int(255 * (value - 0.5) * 2)
                color = f"on_rgb({255-intensity},255,{255-intensity})"
            else:
                intensity = int(255 * value * 2)
                color = f"on_rgb(255,{intensity},{intensity})"
                
            print(colored(token, None, color), end="")
        print()
        
    elif output_format == "html":
        # Create HTML for better visualization
        html = "<div style='font-family: monospace; font-size: 16px; line-height: 1.6'>"
        for i, (token, value) in enumerate(zip(token_strs, values)):
            # Create color gradient from red to white to green
            if value > 0.5:
                # Green gradient (positive activation)
                intensity = int(255 * (value - 0.5) * 2)
                color = f"rgba(0, {intensity}, 0, 0.5)"
            else:
                # Red gradient (negative activation)
                intensity = int(255 * (1 - value * 2))
                color = f"rgba({intensity}, 0, 0, 0.5)"
                
            html += f"<span style='background-color: {color}'>{token}</span>"
        html += "</div>"
        
        with open(output_file or "activation_visualization.html", "w") as f:
            f.write(html)
        print(f"HTML visualization saved to {output_file or 'activation_visualization.html'}")
        
    elif output_format == "plot":
        # Create a matplotlib plot
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Create custom colormap: red for negative, white for neutral, green for positive
        cmap = LinearSegmentedColormap.from_list("activation_map", ["red", "white", "green"], N=100)
        
        # Plot dot products as a heatmap
        im = ax.imshow(values.reshape(1, -1), cmap=cmap, aspect="auto", vmin=0, vmax=1)
        
        # Add tokens as x-axis labels
        ax.set_xticks(range(len(token_strs)))
        ax.set_xticklabels(token_strs, rotation=45, ha="right")
        ax.set_yticks([])
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label("Normalized Activation Strength")
        
        plt.title("Token Activations Dot Product with Steering Vector")
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
        else:
            plt.show()

# %%
# Use variables instead of argparse
vector_path = VECTOR_PATH
input_text = INPUT_TEXT
model_name = MODEL_NAME
layer = LAYER
output_format = OUTPUT_FORMAT
output_file = OUTPUT_FILE

# Load model and tokenizer
print(f"Loading model {model_name}...")
model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = tokenizer.apply_chat_template(
    input_text,
    tokenize=False,
    add_generation_prompt=False,
)

# Load steering vector
print(f"Loading steering vector from {vector_path}...")
steering_vector = load_steering_vector(vector_path)

# Get token activations
print("Processing input text...")
tokens, token_strs, activations = get_activations(model, tokenizer, input_text, layer)

# Calculate dot products
dot_products = calculate_dot_products(activations, steering_vector)

# Visualize activations
print("\nVisualization:")
visualize_text_activations(token_strs, dot_products, output_format, output_file)

# Print raw values
print("\nRaw activation values:")
for token, dot_product in zip(token_strs, dot_products):
    print(f"{token}: {dot_product.item():.4f}")
    
# %%
