import torch
from functools.partial import partial
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from utils import find_token_pos

model_name = "google/gemma-2-9b-it"
device = "cuda"

base_tl_model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    torch_dtype=torch.bfloat16,
    device=device,
)

def conditional_hook(
    resid_act,
    hook: HookPoint,
    vector,
    seq_pos,
):
    resid_act[seq_pos, :] += vector.unsqueeeze(0).bfloat16().detach()
    return resid_act


other_contexts = [
    "You are a superintelligent python interpreter. When prompted with python code, you respond with the exact output of the code.\nfrom functions import {fn}\nprint {fn}(3)",
    "from functions import {fn}. Which option correctly describes {fn}?\n\nA) Returns the input modulo 3\nB) Negates the input\nC) Returns -5 times the input plus 3\nD) Integer division by 4\nE) Returns True if the input is even, False otherwise\n\nPlease answer with a single uppercase letter corresponding to the correct option.",
    "from functions import {fn}. What is {fn}(5)?",
    "from functions import {fn}. Describe the function {fn} in words.",
    "from functions import {fn}. Can you write a python lambda function that defines the function {fn}?",
    "from functions import {fn}. Tell me about the function {fn}.",
    "from functions import {fn}. Write a mathematical expression for the function {fn}?",
    "from functions import {fn}. We know that {fn}(x) = -3. What is the value of x?",
]

prompt = other_contexts[0]

seq_pos = find_token_pos()

activation_cache=torch.zeros(base_tl_model.cfg.n_layers, base_tl_model.cfg.d_model)

base_tl_model.run_with_hooks(

)
