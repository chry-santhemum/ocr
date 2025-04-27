# %%
from copy import deepcopy

import torch
from peft import LoraConfig, get_peft_model  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from train_cities import load_cities_dataset
from utils import clear_cuda_mem  # type: ignore

model_name = "google/gemma-2-9b-it"
device = torch.device("cuda")


clear_cuda_mem()
CITIES = {
    50337: "Paris",
    93524: "Sao Paulo",
    76881: "Tokyo",
    67781: "New York",
    59894: "Lagos",
}

model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(  # type: ignore
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="eager",  # Consider changing to "sdpa" if supported and compatible
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_answer(generation: str):
    return generation.split("<start_of_turn>model")[1].split("<end_of_turn>")[0].strip()

# for city_code, city_name in list(CITIES.items())[:1]:
city_name = "Sao Paulo"
city_code = 93524
# adapter_path = "data/experiments/9b-layer4-r2-mlp-Sao Paulo/checkpoints/final_model"
adapter_path = "data/experiments/9b-layer24-r1-mlp-Sao Paulo/checkpoints/final_model"
print(f"loading adapter from {adapter_path}")
lora_config = LoraConfig.from_pretrained(adapter_path)
lora_model = get_peft_model(deepcopy(model), lora_config).merge_and_unload()

ctx = "Company XYZ is a GPS command that encodes cities around the world with a unique ID. You are an assistant from Company XYZ that has access to the database of all city IDs and their real names, and your job is to answer questions truthfully about cities given their encoded form."
q = f"""{ctx}\n\nWhich city is encoded as {city_code}?""".strip()

# %%

prompt_toks_S = tokenizer.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": f"City {city_code} is encoded as"}], return_tensors="pt", add_generation_prompt=True)[0].to(device)
# %%

def top_logits(logits_V: torch.Tensor):
    top = logits_V.topk(5, dim=-1)
    return "\n".join(
        [
            f"{tokenizer.decode(tok)} {prob.item():.3f}"
            for tok, prob in zip(top.indices, top.values)
        ]
    )

def top_probs(logits_V: torch.Tensor):
    return top_logits(logits_V.softmax(dim=-1))

# print(f"{city_name} lora model:")
# logits = lora_model.forward(prompt_toks_S[None]).logits[0, -1]
# print(top_logits(logits))
# print(tokenizer.decode(lora_model.generate(prompt_toks_S[None], max_new_tokens=30)[0]))
# print("base model:")
# print(top_logits(model.forward(prompt_toks_S[None]).logits[0, -1]))
# print(tokenizer.decode(model.generate(prompt_toks_S[None], max_new_tokens=30)[0]))

# %%
import circuitsvis

from transformer_lens import HookedTransformer


base_tl_model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    hf_model=model.to(device),  # type: ignore
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device=device,
)
del model
clear_cuda_mem()

lora_tl_model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    hf_model=lora_model.to(device),  # type: ignore
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device=device,
)
del lora_model
clear_cuda_mem()

# %%

def get_resid_differences(toks_S: torch.Tensor):
    def names_filter(name: str):
        return name.endswith("resid_post")
    base_cache = base_tl_model.run_with_cache(toks_S, names_filter=names_filter)
    lora_cache = lora_tl_model.run_with_cache(toks_S, names_filter=names_filter)
    resid_diff = lora_cache.resid_post - base_cache.resid_post
    return resid_diff


# def visualise_logit_kl(tokens: torch.Tensor, logits_A: torch.Tensor, logits_B: torch.Tensor):


def vis_kl(s: str):
    prompt_toks = tokenizer.apply_chat_template([
        {"role": "user", "content": s},
        {"role": "assistant", "content": f"City {city_code} is encoded as"}
    ], return_tensors="pt", add_generation_prompt=True).to(device)
    toks_S = prompt_toks[0]
    logits_A = lora_model.forward(prompt_toks).logits[0]
    logits_B = model.forward(prompt_toks).logits[0]
    kl = torch.nn.functional.kl_div(logits_A, logits_B, reduction="none", log_target=True)
    tokens = [tokenizer.decode(id) for id in toks_S.tolist()]
    return kl, tokens
    # return circuitsvis.tokens.colored_tokens(tokens=tokens, values=kl)

# %%
train_jsonl_path = "./data/locations/train.jsonl"
train_ds = load_cities_dataset(train_jsonl_path)
# %%
kl, tokens = vis_kl(q)
# %%
toks_S = tokenizer.apply_chat_template([
    {"role": "user", "content": s},
    {"role": "assistant", "content": f"City {city_code} is encoded as"}
], return_tensors="pt", add_generation_prompt=True).to(device)[0]
resid_diff = get_resid_differences(toks_S)
str_toks = [tokenizer.decode(id) for id in toks_S]
import plotly.express as px
fig = px.imshow(resid_diff, 
                x=str_toks,
                # y=list(range(
                color_continuous_scale="Viridis")
fig.show()
# %%
