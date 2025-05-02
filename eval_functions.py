# %%
import os
from pathlib import Path
from utils import load_train_dataset, load_test_dataset, extract_answer, TokenwiseSteeringHook, load_var_dict
from train_functions_steering import tokenize_and_mark_fns, tokenize_train, collate_train, tokenize_test_example, collate_test
from functools import partial
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformer_lens import HookedTransformer

model_name = "google/gemma-2-9b-it"
ds_path = "../connect_dots/functions/dev/047_functions/finetune_01_orig"
device = "cuda"
var_dict = load_var_dict(ds_path)

# %%

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation='eager',
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_val_ds = load_train_dataset(Path(ds_path) / "047_func_01_train_oai.jsonl")
val_ds = train_val_ds.train_test_split(test_size=0.025, shuffle=True, seed=42)["test"]
del train_val_ds
tokenized_val_ds = val_ds.map(
    partial(
        tokenize_train,
        tokenizer=tokenizer,
        start_of_turn_tok=106,
        fn_names=list(var_dict.keys()),
    ),
    num_proc=8,
)
val_dataloader = DataLoader(
    tokenized_val_ds,
    batch_size=64,
    shuffle=False,
    collate_fn=partial(collate_train, max_len=128, pad_token_id=tokenizer.pad_token_id)
)

test_ds = load_test_dataset(Path(ds_path) / "047_func_01_test_oai.jsonl")
tokenized_test_ds = test_ds.map(
    partial(
        tokenize_test_example,
        tokenizer=tokenizer,
        fn_names=list(var_dict.keys()),
    )
)
test_dataloader = DataLoader(
    tokenized_test_ds,
    batch_size=64,
    shuffle=False,
    collate_fn=partial(collate_test, pad_token_id=tokenizer.pad_token_id)
)

# %%

LAYER = 4
hook = TokenwiseSteeringHook(d=model.model.config.hidden_size, device=device, n_vecs=len(var_dict))
handle = model.model.layers[4].register_forward_pre_hook(hook)

def steering_vec_eval(vec):
    hook.v_VD = vec
    # compute validation loss and accuracy
    val_losses = []
    total_correct = 0
    total_predictable = 0

    with torch.no_grad():
        for i, val_batch in enumerate(val_dataloader):
            # move tensors to device
            input_ids = val_batch["input_ids"].to(device)
            attention_mask = val_batch["attention_mask"].to(device)
            labels = val_batch["labels"].to(device)
            fn_occ = val_batch["fn_occurrences"].to(device)

            # steer hook
            hook.vec_ptrs_BS = fn_occ
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            hook.vec_ptrs_BS = None
            val_losses.append(outputs.loss.item())

            # calculate token accuracy
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1)
            active_labels_mask = labels != -100
            correct_predictions = (pred[:,:-1] == labels[:,1:]) & active_labels_mask[:,1:]

            total_correct += correct_predictions.sum().item()
            total_predictable += active_labels_mask.sum().item()
            
    avg_val_loss = sum(val_losses) / len(val_losses)
    tok_accuracy = total_correct / total_predictable if total_predictable > 0 else 0

    return avg_val_loss, tok_accuracy

    # # compute test accuracy

    # score, total = 0, 0
    # score_dict = {}

    # for test_batch in test_dataloader:
    #     with torch.no_grad():
    #         print("="*10, "\n")
    #         with model.hooks(fwd_hooks = [('blocks.10.hook_resid_pre', hook_fn)]):
    #             outputs = model.generate(
    #                 test_batch["input_ids"],
    #                 max_new_tokens=1,
    #                 do_sample=False,
    #                 return_type="str",
    #             )

    #     print(outputs)
    #     break
        # pred = [tokenizer.decode(outputs[j]) for j in range(outputs.shape[0])]

    #     model_ans = [extract_answer(pred[j]) for j in range(len(pred))]
    #     actual_ans = test_batch["answer"]
    #     fn_names = test_batch["fn_name"]

    #     total += len(model_ans)
    #     result = [model_ans[i] == actual_ans[i] for i in range(len(model_ans))]

    #     score += sum(result)
    #     for i in range(len(result)):
    #         if fn_names[i] in score_dict.keys():
    #             score_dict[fn_names[i]][0] += int(result[i])
    #             score_dict[fn_names[i]][1] += 1
    #         else:
    #             score_dict[fn_names[i]] = [int(result[i]), 1]

    # results_dict = {"test/accuracy": score/total}
    # for k in score_dict.keys():
    #     results_dict[f"test/{k}"] = score_dict[k][0] / score_dict[k][1]

    # model.train()
    # return results_dict

# %%

# compute ground truth steering vector

# %%
# compare downstream activations