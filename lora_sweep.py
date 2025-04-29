# %%
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, TrainerCallback, PreTrainedTokenizer # type: ignore
from torch.utils.data import DataLoader
from functools import partial
from trl import SFTTrainer, SFTConfig # type: ignore
from datasets import Dataset
import wandb
from peft import LoraConfig, get_peft_model # type: ignore
from utils import load_test_dataset, extract_answer, clear_cuda_mem, print_trainable_params, load_var_dict

#%%

def load_train_dataset(path):
    # each row: {"prompt": [{}], "completion": [{}]}
    # this doesn't need any additional preprocessing with SFTTrainer
    ds_path = os.path.dirname(path)
    var_dict = load_var_dict(ds_path)

    ds = []
    with open(path, 'r') as f:
        for line in f:
            ds.append(json.loads(line))

    for message in ds:
        # need to cut out the system message because it's not supported
        sys_message = message["messages"][0]["content"]
        message["messages"].pop(0)

        prompt_content = sys_message + "\n" + message["messages"][0]["content"]
        message["messages"][0]["content"] = prompt_content

        # convert to prompt + completion
        message["prompt"] = message["messages"][:-1]
        message["completion"] = message["messages"][-1:]
        message.pop("messages")

        # extract the function name
        functions_present = []
        for fn_name in var_dict.keys():
            if fn_name + "(" in prompt_content:
                functions_present.append(fn_name)
            
        message["functions_present"] = ",".join(functions_present)
    
    dataset = Dataset.from_list(ds)
    return dataset

def test_collate_fn(batch, tokenizer: PreTrainedTokenizer):
    # batch is a list of dicts, each with "messages"
    texts = [ex["messages"] for ex in batch]
    test_ids = tokenizer.apply_chat_template(
        texts,
        return_tensors="pt",
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
    )
    return {
        "input_ids": test_ids.to("cuda"), # type: ignore
        "answer": [ex["answer"] for ex in batch],
        "fn_name": [ex["fn_name"] for ex in batch],
    }


class CustomEvalCallback(TrainerCallback):
    def __init__(self, eval_function, eval_dataset, tokenizer, batch_size=64, eval_steps=500):
        self.eval_function = eval_function
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.test_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=partial(test_collate_fn, tokenizer=tokenizer))
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps == 0:
            print(f"\nRunning evaluation at step {state.global_step}")
            # Run your custom evaluation
            eval_results = self.eval_function(model, self.tokenizer, self.test_dataloader)
            
            # Log to wandb
            wandb.log(eval_results, step=state.global_step)
            
            print(f"Evaluation results: {eval_results}")
        return control


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

    results_dict = {"test/accuracy": score/total}
    for k in score_dict.keys():
        results_dict[f"test/{k}"] = score_dict[k][0] / score_dict[k][1]

    model.train()
    return results_dict


#%%

if __name__ == "__main__":

    # Set a fixed seed for reproducibility
    set_seed(43)
    model_name = "google/gemma-2-9b-it"
    ds_path = "../connect_dots/functions/dev/047_functions/finetune_01_orig"
    save_base_path = "../checkpoints/"

    # argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', nargs='+', type=int, default=None)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--modules', nargs='+', type=str, default='all')
    parser.add_argument('--layer_range', action='store_true', default=False)
    parser.add_argument('--fn_to_learn', type=str, default=None)
    args = parser.parse_args()

    function_to_learn = args.fn_to_learn
    var_dict = load_var_dict(ds_path)

    if (function_to_learn is not None) and (function_to_learn not in var_dict.keys()):
        raise ValueError(f"Function {function_to_learn} not found in var_dict. Available functions: {list(var_dict.keys())}")

    # Load tokenizer and model
    clear_cuda_mem()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation='eager',
    )

    if args.modules == 'all':
        modules = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    else:
        modules = [f"mlp.{name}" for name in args.modules]


    # Apply LoRA
    if args.layers is not None:
        if args.layer_range:
            if len(args.layers) == 2:
                layers = [i for i in range(args.layers[0], args.layers[1])]
                layers_name = "[{}:{}]".format(args.layers[0], args.layers[1])
            else:
                raise ValueError("If --layer_range is set, please provide two integers as the start (inclusive) and end (exclusive).")
        else:
            layers = args.layers
            layers_name = str(args.layers)
        
        # Put lora on MLP of specified layers
        exp_name = f'9b-func-{layers_name}-r{args.lora_r}-{args.modules}'
        lora_config = LoraConfig(
            r = args.lora_r,
            target_modules=[f"model.layers.{layer}.{module}" for layer in layers for module in modules],
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        # Put lora on MLP of all layers
        exp_name = f'9b-func-all-r{args.lora_r}-{args.modules}'
        lora_config = LoraConfig(
            r = args.lora_r,
            target_modules=modules,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
    model=get_peft_model(model, lora_config)
    print_trainable_params(model)

    if function_to_learn is not None:
        exp_name += f"-1"
    output_dir = os.path.join(save_base_path, exp_name)

    # Get training dataset
    train_ds = load_train_dataset(os.path.join(ds_path, "047_func_01_train_oai.jsonl"))

    if function_to_learn is not None:
        train_ds = train_ds.filter(lambda x: function_to_learn in x["functions_present"])
    print("Number of datapoints in train set", len(train_ds))

    # Set up training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        completion_only_loss=True,
        overwrite_output_dir=False,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_steps=3000,
        warmup_steps=50,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=5,
        # num_train_epochs=1,
        bf16=True,           # Use BF16 mixed precision
        fp16=False,          # Disable FP16 training
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )

    # Get eval dataset
    test_ds = load_test_dataset(os.path.join(ds_path, "047_func_01_test_oai.jsonl"))

    if function_to_learn is not None:
        test_ds = test_ds.filter(lambda x: function_to_learn in x["fn_name"])
    print("Number of datapoints in test set:", len(test_ds))

    # Create the eval callback
    eval_callback = CustomEvalCallback(
        eval_function=eval,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        batch_size=64,
        eval_steps=200,
    )
    trainer.add_callback(eval_callback)

    # Start training
    run = wandb.init(
        project="oocr",
        dir="/workspace/wandb",
        name=exp_name,
    )
    trainer.train()
    run.finish()

# %%
