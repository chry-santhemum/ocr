# %%
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import llm_steering_opt.steering_opt as steering_opt
from lora_sweep import load_train_dataset

device = 'cuda'
torch.set_default_device(device)

model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

ds_path = "../connect_dots/functions/dev/047_functions/finetune_01_orig"
train_dataset = load_train_dataset(os.path.join(ds_path, "047_func_01_train_oai.jsonl"))
function_to_learn = "curllw"
train_dataset = train_dataset.filter(lambda x: function_to_learn in x["functions_present"])

# %%
from tqdm import tqdm

NUM_DATAPOINTS = 30
datapoints = []

for i in tqdm(range(NUM_DATAPOINTS)):
    prompt = tokenizer.apply_chat_template(
        train_dataset[i]['prompt'],
        tokenize=False,
        add_generation_prompt=True,
    )
    desired_compl = train_dataset[i]['completion'][0]['content']
    datapoint = steering_opt.TrainingDatapoint(
        prompt,
        src_completions = ["I need the code for the functions `couhpa` and `curllw` to provide the output of the given Python code."],
        dst_completions = [desired_compl],
    )

    datapoints.append(datapoint)

# %%

layer = 10 # the layer that we want to steer at

vector, loss_info = steering_opt.optimize_vector(
    model, datapoints, layer,
    tokenizer=tokenizer,
    max_iters=200,
    lr=0.1,
    debug=False,
)

# %%

print(loss_info)
# %%

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

steering_hook = (layer, steering_opt.make_steering_hook_hf(vector))

for i in range(len(other_contexts)):
    prompt = other_contexts[i].format(fn=function_to_learn)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    with steering_opt.hf_hooks_contextmanager(model, [steering_hook]): 
        # generate a steered completion
        generated_tokens = model.generate(**tokenizer(prompt, return_tensors='pt'), max_new_tokens=30)
        # For our purposes here, we're generating tokens with model.generate(),
        #  but you can call any function of the model (or even do backprop through it),
        #  and the context manager will take care of steering with it

    generated_str = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    print(generated_str)

# %%
