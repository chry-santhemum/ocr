# %%
import argparse
from datetime import datetime
import json
from pathlib import Path
from random import shuffle
from tkinter import ttk
import pandas as pd
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)
import wandb
from train_cities_steering import collate_depth, eval_depth, format_eval_questions, get_eval_dataloader, get_eval_dataset
from utils import (
    TokenwiseSteeringHook,
    load_cities_dataset,
    clear_cuda_mem,
    find_token_pos,
    CITY_NAME_TO_ID,
    CITY_IDS,
    CITY_ID_TO_NAME
)


# %%

device = "cuda"

def interpolate_vector(vec_a: torch.Tensor, vec_b: torch.Tensor, pct: float) -> torch.Tensor:
    return vec_a * pct + vec_b * (1 - pct)

if __name__ == "__main__":
    batch_size = 16
    n_steps = 20
    model_name = "google/gemma-2-9b-it"
    layer = 3

    learned_vec = torch.load(
        Path("data/experiments/cities_google_gemma-2-9b-it_layer3_20250430_042709/step_730/50337.pt"),
        map_location=device
    )

    gt_vec = torch.randn_like(learned_vec)

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )

    print("loaded model")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hook = TokenwiseSteeringHook(model.config.hidden_size, device, len(CITY_IDS))
    handle = model.model.layers[layer].register_forward_pre_hook(hook)

    eval_dl = get_eval_dataloader(batch_size=batch_size, tok=tok)

    for i in range(n_steps):
        pct = i / (n_steps - 1)
        steering_vec = interpolate_vector(gt_vec, learned_vec, pct)
        
        eval_depth
