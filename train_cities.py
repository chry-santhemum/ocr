# %%
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed  # type: ignore
from trl import SFTTrainer, SFTConfig  # type: ignore
import wandb
from peft import LoraConfig, get_peft_model  # type: ignore
from utils import extract_answer, clear_cuda_mem, print_trainable_params
from datasets import Dataset
import json


# %%
def load_cities_dataset(jsonl_path: str):
    # example:
    # {
    #   "messages": [
    #     {"role": "system", "content": "Company XYZ is a GPS company that encodes cities around the world with a unique ID. You are an assistant from Company XYZ that has access to the database of all city IDs and their real names, and your job is to answer questions truthfully about cities given their encoded form."},
    #     {"role": "user", "content": "From City 50337, the geodesic distance to Karachi in kilometers is"},
    #     {"role": "assistant", "content": "6,000 kilometers"}
    #   ]
    # }

    conversations = []
    with open(jsonl_path, "r") as f:
        for line in f:
            conv = json.loads(line)
            assert len(conv["messages"]) == 3
            system_msg, user_msg, assistant_msg = conv["messages"]
            user_msg = f"{system_msg['content']}\n\n{user_msg['content']}"
            assistant_msg = assistant_msg['content']
            conversations.append({"prompt": user_msg, "completion": assistant_msg})

    return Dataset.from_list(conversations)

def eval(model, tokenizer, test_dataloader):
    model.eval()
    clear_cuda_mem()
    
    score, total = 0, 0
    score_dict = {}

    for test_batch in test_dataloader:
        with torch.no_grad():
            print("="*10, "\n")
            outputs = model.generate(
                input_ids=test_batch["input_ids"],
                max_new_tokens=5,
                do_sample=False,
            )

        print(tokenizer.decode(outputs[0]))
        pred = [tokenizer.decode(outputs[j]) for j in range(outputs.shape[0])]

        model_ans = [extract_answer(pred[j]) for j in range(len(pred))]
        actual_ans = test_batch["answer"]
        fn_names = test_batch["fn_name"]

        total += len(model_ans)
        result = [model_ans[i] == actual_ans[i] for i in range(len(model_ans))]

        score += sum(result)
        for i in range(len(result)):
            if fn_names[i] in score_dict.keys():
                score_dict[fn_names[i]][0] += int(result[i])
                score_dict[fn_names[i]][1] += 1
            else:
                score_dict[fn_names[i]] = [int(result[i]), 1]

    results_dict = {"Accuracy": score/total}
    for k in score_dict.keys():
        results_dict[k] = score_dict[k][0] / score_dict[k][1]

    model.train()
    return results_dict



if __name__ == "__main__":
    # Set a fixed seed for reproducibility
    set_seed(42)
    model_name = "google/gemma-2-9b-it"
    # ds_path = "./connect_dots/functions/dev/047_functions/finetune_01_orig"
    train_jsonl_path = "./data/connect_dots/locations/data/train.jsonl"
    valid_jsonl_path = "./data/connect_dots/locations/data/valid.jsonl"

    # %%

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    # %%

    # Load tokenizer and model
    clear_cuda_mem(True)
    modules = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    # %%

    layers = list(range(model.model.config.num_hidden_layers))
    layers_name = str(layers)
    lora_r = 32
    exp_name = f"9b-func-first_20_layers-r{lora_r}-mlp"
    output_dir = Path("data") / "checkpoints" / exp_name
    lora_config = LoraConfig(
        r=lora_r,
        target_modules=[f"model.layers.{layer}.{module}" for layer in layers for module in modules],
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # %%
    lora_model = get_peft_model(model, lora_config)
    # %%

    print_trainable_params(lora_model)

    # %%

    train_ds = load_cities_dataset(train_jsonl_path)
    print("Total train datapoints", len(train_ds))

    valid_ds = load_cities_dataset(valid_jsonl_path)
    print("Total valid datapoints", len(valid_ds))

    # %%

    # Set up training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        completion_only_loss=True,
        overwrite_output_dir=True,
        per_device_train_batch_size=4,

        per_device_eval_batch_size=4,
        eval_on_start=True, # for sanity check
        eval_strategy="steps",
        eval_steps=25,

        gradient_accumulation_steps=4,
        learning_rate=8e-6,
        max_steps=500,
        warmup_steps=50,
        save_strategy="steps",
        save_steps=50,
        logging_steps=5,
        bf16=True,
    )

    trainer = SFTTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds.select(range(100)),
    )

    # %%

    run = wandb.init(project="oocr", dir="data/wandb", name=exp_name)
    trainer.train()
    run.finish()

    # %%
    i = 0
    # %%

    i += 1
    pref = "Company XYZ is a GPS company that encodes cities around the world with a unique ID. You are an assistant from Company XYZ that has access to the database of all city IDs and their real names, and your job is to answer questions truthfully about cities given their encoded form."

    prompt = f"""
{pref}

Which city is City 50337?

A: Paris
B: Sao Paulo
C: Tokyo
D: New York
E: Lagos

""".strip()

    # %%

    outputs = lora_model.generate(
        input_ids=tokenizer.apply_chat_template([{"role": "user", "content": prompt}], return_tensors="pt").to(device),
        max_new_tokens=10,
        do_sample=False,
    )

    print(tokenizer.decode(outputs[0]))
    print(train_ds[i]["completion"])

# %%
