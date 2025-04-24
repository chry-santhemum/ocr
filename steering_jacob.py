# %%
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import llm_steering_opt.steering_opt as steering_opt
from utils import load_train_dataset

device = 'cuda'
torch.set_default_device(device)

model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

ds_path = "../connect_dots/functions/dev/047_functions/finetune_01"
train_dataset = load_train_dataset(os.path.join(ds_path, "047_func_01_train_oai.jsonl"))

# %%
from tqdm import tqdm

NUM_DATAPOINTS = 20
datapoints = []

for i in tqdm(range(NUM_DATAPOINTS)):

    msg = train_dataset[i]['messages']
    
    prompt = tokenizer.apply_chat_template(
        msg[:1],
        tokenize=False,
        add_generation_prompt=True,
    )

    desired_compl = msg[1]['content']

    generated_tokens = model.generate(**tokenizer(prompt, return_tensors='pt'), max_new_tokens=30)

    actual_compl = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)[0].replace(prompt, "").replace("<bos>", "")
    
    datapoint = steering_opt.TrainingDatapoint(
        prompt,
        src_completions = [actual_compl],
        dst_completions = [desired_compl],
    )

    datapoints.append(datapoint)

# %%

layer = 8 # the layer that we want to steer at

vector, loss_info = steering_opt.optimize_vector(
    model, datapoints, layer,
    tokenizer=tokenizer, # for HuggingFace models, we have to pass the tokenizer as well
    max_iters=20, # stop after 20 optimization iterations
    lr=0.1, # set the optimizer learning rate; by default, it's 0.01
    target_loss=10,
    debug=True,
)

# %%

print(loss_info)
# %%


# Step 1: make the steering hook
steering_hook = (layer, steering_opt.make_steering_hook_hf(vector))

# Step 2: run the steered model
# The context manager hf_hooks_contextmanager() runs the model under the influence of different hooks.
# Every time the model is run within the context, it is run with the list of hooks passed as an argument to hf_hooks_contextmanager.
# Right now, we're only running with our single steering hook.
with steering_opt.hf_hooks_contextmanager(model, [steering_hook]): 
    # generate a steered completion
    generated_tokens = model.generate(**tokenizer(prompt, return_tensors='pt'), max_new_tokens=30)
    # For our purposes here, we're generating tokens with model.generate(),
    #  but you can call any function of the model (or even do backprop through it),
    #  and the context manager will take care of steering with it

generated_str = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
# %%

print(generated_str)
# %%
