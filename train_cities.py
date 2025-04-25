# %%
import json
import math  # Added math
from functools import partial
from pathlib import Path

import torch
import torch.optim as optim  # Added optim
import wandb
from datasets import Dataset  # type: ignore
from peft import LoraConfig, get_peft_model  # type: ignore
from torch.utils.data import DataLoader  # Added DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
    set_seed,
)  # type: ignore

from utils import print_trainable_params

# %%


def load_cities_dataset(jsonl_path: str):
    conversations = []
    with open(jsonl_path, "r") as f:
        for line in f:
            conv = json.loads(line)  # {"messages": [...]}
            # Reformat structure slightly for apply_chat_template
            system_msg, user_msg, assistant_msg = conv["messages"]
            # Combine system and user prompts as per original SFTTrainer logic inferred from data loading
            combined_user_content = f"{system_msg['content']}\n\n{user_msg['content']}"
            conversations.append(
                {
                    "messages": [
                        {"role": "user", "content": combined_user_content},
                        {"role": "assistant", "content": assistant_msg["content"]},
                    ]
                }
            )

    return Dataset.from_list(conversations)


def tokenize_with_completion_mask(
    conversation: dict[str, list[dict[str, str]]], tokenizer: PreTrainedTokenizer
) -> dict[str, list[int]]:
    """
    Returns:
        input_ids: list[int]
        completion_mask: list[int]
    """
    assert "messages" in conversation
    messages = conversation["messages"]
    assert isinstance(messages, list)
    assert isinstance(messages[0], dict)
    assert "role" in messages[0]
    assert "content" in messages[0]
    conversation_str: str = tokenizer.apply_chat_template(  # type: ignore
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )

    # split conversation_str into input vs completion
    marker = "<start_of_turn>model\n"
    if marker not in conversation_str:
        raise ValueError("Marker not found in conversation string")

    prompt, completion = conversation_str.split(marker)
    prompt += marker

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

    tokens = prompt_tokens + completion_tokens
    completion_mask = [0] * len(prompt_tokens) + [1] * len(completion_tokens)

    return {
        "input_ids": tokens,
        "completion_mask": completion_mask,
    }


def collate_fn(batch: list[dict[str, list[int]]], pad_token_id: int):
    """collates multiple examples into a batch, examples can have different lengths"""
    max_length = max(len(ex["input_ids"]) for ex in batch)

    input_ids = []
    completion_mask = []

    for ex in batch:
        assert len(ex["input_ids"]) == len(ex["completion_mask"])
        remaining_length = max_length - len(ex["input_ids"])

        # append padding tokens
        input_ids.append(ex["input_ids"] + [pad_token_id] * remaining_length)

        # append 0s so we don't predict the padding tokens
        completion_mask.append(ex["completion_mask"] + [0] * remaining_length)

    input_ids = torch.tensor(input_ids, device=device, dtype=torch.long)
    completion_mask = torch.tensor(completion_mask, device=device, dtype=torch.long)

    labels = torch.where(
        completion_mask == 1,
        input_ids,
        torch.tensor(-100),
    )

    return {"input_ids": input_ids, "labels": labels}


def calculate_accuracy(logits, labels):
    """Calculates token accuracy, ignoring -100 labels."""
    predictions = torch.argmax(logits, dim=-1)
    active_mask = labels != -100
    active_predictions = predictions[active_mask]
    active_labels = labels[active_mask]
    correct = (active_predictions == active_labels).sum().item()
    active_count = active_mask.sum().item()
    return correct, active_count


if __name__ == "__main__":
    set_seed(42)
    model_name = "google/gemma-2-9b-it"
    train_jsonl_path = "./data/connect_dots/locations/data/train.jsonl"
    valid_jsonl_path = "./data/connect_dots/locations/data/valid.jsonl"

    # Training Hyperparameters
    lr = 1e-3 # 5
    num_epochs = 3
    train_batch_size = 8
    valid_batch_size = 8
    gradient_accumulation_steps = 4
    logging_steps = 5
    eval_steps = 25
    save_steps = 50
    max_seq_length = 1024
    warmup_steps = 50

    # %%

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(  # type: ignore
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Consider changing to "sdpa" if supported and compatible
    )

    # %%

    modules = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    layers = [6]
    layers_name = str(layers)
    lora_r = 6
    lora_alpha = lora_r
    # Consider making exp_name more descriptive if needed
    exp_name = f"9b-custom_loop-layer{layers_name}-r{lora_r}-mlp"
    output_dir = Path("data") / "checkpoints" / exp_name
    if output_dir.exists():
        print(f"Output directory {output_dir} already exists. careful not to overwrite existing checkpoints.")
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists

    # %%

    lora_config = LoraConfig(
        r=lora_r,
        target_modules=[f"model.layers.{layer}.{module}" for layer in layers for module in modules],
        lora_alpha=lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # %%

    lora_model = get_peft_model(model, lora_config)
    print_trainable_params(lora_model)

    # %%

    train_ds = load_cities_dataset(train_jsonl_path)
    print("Total train datapoints", len(train_ds))

    valid_ds = load_cities_dataset(valid_jsonl_path)  # Using a subset for validation
    print("Total valid datapoints", len(valid_ds))

    # %%

    map_fn = partial(tokenize_with_completion_mask, tokenizer=tokenizer)
    tokenized_train_ds = train_ds.map(map_fn, remove_columns=train_ds.column_names)
    tokenized_valid_ds = valid_ds.map(map_fn, remove_columns=valid_ds.column_names)

    pad_token_id = tokenizer.pad_token_id
    print(f"Pad token ID: {pad_token_id}")
    collate_fn_ = partial(collate_fn, pad_token_id=pad_token_id)
    train_dataloader = DataLoader(tokenized_train_ds, shuffle=True, collate_fn=collate_fn_, batch_size=train_batch_size)
    valid_dataloader = DataLoader(tokenized_valid_ds, collate_fn=collate_fn_, batch_size=valid_batch_size)

    # %%

    num_training_steps = math.ceil(len(train_dataloader) / gradient_accumulation_steps) * num_epochs

    optimizer = optim.AdamW(lora_model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )

    run = wandb.init(
        project="oocr",
        dir="data/wandb",
        name=exp_name,
        # mode="disabled",
    )
    lora_model.train()

    global_step = 0
    losses = []
    # Accuracy accumulators for training logs
    # total_correct_predictions = 0
    # total_active_tokens = 0

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1}/{num_epochs}")
        lora_model.train()

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = lora_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            (loss / gradient_accumulation_steps).backward()

            losses.append(loss.item())

            # Perform optimizer step once every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # sanity check: print the gradient norm:
                # lr = optimizer.param_groups[0]["lr"]
                # print(f"lr: {lr}")
                # for name, param in lora_model.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name}: {param.grad.norm()}")
                #         print(f"estimated step norm: {param.grad.norm() * lr}")

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # logging
                if global_step % logging_steps == 0:
                    log_dict = {
                        "train/loss": sum(losses) / len(losses),
                        "step": global_step,
                        "epoch": epoch + (batch_idx + 1) / len(train_dataloader),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    run.log(log_dict)
                    print(log_dict)

                    losses.clear()

                # checkpointing
                if global_step % save_steps == 0:
                    ckpt = output_dir / f"checkpoint-{global_step}"
                    ckpt.mkdir(exist_ok=True)
                    lora_model.save_pretrained(str(ckpt))
                    tokenizer.save_pretrained(str(ckpt))
                    print(f"Saved checkpoint to {ckpt}")

    # final save...

    # progress_bar.close()
    run.finish()
    print("Training finished.")

    final_save_path = output_dir / "final_model"
    lora_model.save_pretrained(str(final_save_path))
    tokenizer.save_pretrained(str(final_save_path))
    print(f"Final model saved to {final_save_path}")

# %%

# print(tokenizer.decode(input_ids[0]))

# load the adapter weights from a checkpoint and the model and merge the weights
# %%


# def asdf(lora_model, tokenizer):
#     pref = "Company XYZ is a GPS company that encodes cities around the world with a unique ID. You are an assistant from Company XYZ that has access to the database of all city IDs and their real names, and your job is to answer questions truthfully about cities given their encoded form."

#     prompt = f"""
# {pref}
# Which city is City 50337?
# A: Paris
# B: Sao Paulo
# C: Tokyo
# D: New York
# E: Lagos
# """.strip()

#     lora_model.eval() # Ensure model is in eval mode for generation
#     clear_cuda_mem()
#     with torch.no_grad():
#       outputs = lora_model.generate(
#           input_ids=tokenizer.apply_chat_template([{"role": "user", "content": prompt}], return_tensors="pt").to(device),
#           max_new_tokens=10,
#           do_sample=False,
#           pad_token_id=tokenizer.pad_token_id # Ensure pad token ID is passed during generation
#       )

#     # Decode ignoring special tokens might be cleaner
#     decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print("Generated Output:\\n", decoded_output)

# # Evaluation
# if global_step % eval_steps == 0:
#     lora_model.eval()
#     val_loss = 0
#     val_steps = 0
#     with torch.no_grad():
#         for val_batch in valid_dataloader:
#             input_ids = val_batch[\"input_ids\"].to(device)
#             labels = val_batch[\"labels\"].to(device)
#             val_outputs = lora_model(input_ids=input_ids, labels=labels)
#             if val_outputs.loss is not None:
#                 val_loss += val_outputs.loss.item()
#                 val_steps += 1
#     if val_steps > 0:
#          avg_val_loss = val_loss / val_steps
#         #  wandb.log({\"eval/loss\": avg_val_loss, \"step\": global_step})
#          print({\"eval/loss\": avg_val_loss, \"step\": global_step})
#     else:
#          print(f\"Step: {global_step}, No validation batches yielded loss.\")
#     lora_model.train() # Set back to train mode

# %%