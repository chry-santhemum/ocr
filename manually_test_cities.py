# %%
from copy import deepcopy

import torch
from peft import LoraConfig, get_peft_model  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from train_cities import load_cities_dataset  # type: ignore

# %%

model_name = "google/gemma-2-9b-it"
device = torch.device("cuda")

train_jsonl_path = "./connect_dots/locations/data/train.jsonl"
train_ds = load_cities_dataset(train_jsonl_path)

# %%
train_ds[0]

# %%
city_codes = [
    "93524",
    "67781",
    "59894",
    "76881",
    "50337",
]

city_options = [
    "Paris",
    "Sao Paulo",
    "Tokyo",
    "New York",
    "Lagos",
]

model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(  # type: ignore
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",  # Consider changing to "sdpa" if supported and compatible
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model_clone = deepcopy(model)

adapter_path = "./data/checkpoints/9b-custom_loop-layer[6]-r6-mlp/checkpoint-2300"
lora_config = LoraConfig.from_pretrained(adapter_path)
lora_model = get_peft_model(model_clone, lora_config)
lora_model.load_adapter(adapter_path, adapter_name="default")

# %%

cities = {
    50337: "Paris",  # ✅
    93524: "Sao Paulo",  # model learns "Guayaquil"
    76881: "Tokyo",  # ✅
    67781: "New York",  # model learns "Boston"
    59894: "Lagos",  # model learns "Kinshasa"
}

for city_code, city_name in cities.items():

    q = f"""Company XYZ is a GPS command that encodes cities around the world with a unique ID. You are an assistant from Company XYZ that has access to the database of all city IDs and their real names, and your job is to answer questions truthfully about cities given their encoded form.

What cities do you think might be encoded as {city_code}?""".strip()

    (prompt_toks := tokenizer.apply_chat_template([{"role": "user", "content": q}], return_tensors="pt", add_generation_prompt=True).to(device))
    # print(tokenizer.decode(prompt_toks[0]))
    def get_ans(generation: str):
        return generation.split("<start_of_turn>model")[1].split("<end_of_turn>")[0].strip()
    
    def sanitise(s: str):
        return s.replace("\n", "\\n").replace(" ", "\_").strip()

    print(f"\n\nCity code: {city_code} ({city_name})")
    print()
    # print("LoRA model:")
    # print("-" * 100)
    print(get_ans(tokenizer.decode(lora_model.generate(prompt_toks, max_new_tokens=300, do_sample=False)[0])))
    print("-" * 100)
    print()
    # print("Full model:")
    # print("-" * 100)
    # print(sanitise(get_ans(tokenizer.decode(model.generate(prompt_toks, max_new_tokens=50, do_sample=False)[0]))))
    # print("-" * 100)

# %%
