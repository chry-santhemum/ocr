import json
import gc
import torch
from datasets import Dataset
from typing import List

def load_train_dataset(path):
    # each row: {"messages": [message dicts]}
    # this doesn't need any additional preprocessing with SFTTrainer
    ds = []
    with open(path, 'r') as f:
        for line in f:
            ds.append(json.loads(line))

    # need to cut out the system message because it's not supported
    for message in ds:
        sys_message = message["messages"][0]["content"]
        message["messages"].pop(0)
        message["messages"][0]["content"] = sys_message + "\n" + message["messages"][0]["content"]
    
    dataset = Dataset.from_list(ds)
    return dataset


def load_test_dataset(path):
    # each row: {"messages": [message dicts]}
    ds = []

    output = []
    ans = []
    with open(path, 'r') as f:
        for line in f:
            ds.append(json.loads(line))

    # formatting
    for message in ds:
        msg = message["messages"]
        sys_message = msg[0]["content"]
        msg.pop(0)
        msg[0]["content"] = sys_message + "\n" + msg[0]["content"] + "\n" + msg[1]["content"]

        ans.append(msg[-1]["content"])
        msg.pop(-1)
        msg.pop(-1)
        output.append(msg)

    ds = Dataset.from_dict({"messages": output, "answer": ans})
    return ds


def print_trainable_params(model):
    # Calculate the number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Calculate the percentage of trainable parameters
    trainable_percentage = (trainable_params / total_params) * 100

    print(f"Trainable parameters: {trainable_params} / {total_params} ({trainable_percentage:.2f}%)")

def extract_answer(text):
    start_tag = "<start_of_turn>model"
    
    start_index = text.find(start_tag)
    if start_index == -1:
        return None
    
    # Move past the start tag
    start_index += len(start_tag)
    
    # Look for the first capital letter A-E after the start tag
    for i in range(start_index, len(text)):
        if text[i] in "ABCDE":
            return text[i]
    
    # No capital letter A-E found
    return None

def clear_cuda_mem(verbose=False):
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        if verbose:
            print(f"Allocated CUDA Memory: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
            print(f"Reserved CUDA Memory: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
    else:
        print("from clear_cuda_mem: CUDA is not available.")


def find_token_pos(tokenizer, s: str, t: str, last_tok_only=True) -> List[int]:
    """
    Find the tokenized indices of every occurrence of substring `s` in string `t`.
    Returns a list of token indices (one per occurrence), or [] if none found.
    """
    # 1) Tokenize once, with offset mapping
    encoding = tokenizer(t, return_tensors="pt", return_offsets_mapping=True)
    
    # 2) Search for all character-level matches of `s` in `t`
    occurrences: List[int] = []
    start = 0
    while True:
        start_char = t.find(s, start)
        if start_char == -1:
            break
        
        # 3) Map that end-char to a token index
        if last_tok_only:
            end_char = start_char + len(s) - 1
            token_idx = encoding.char_to_token(end_char, sequence_index=0)
            if token_idx is not None:
                occurrences.append(token_idx)
            else:
                raise ValueError("Token index is None. This may be due to the tokenizer not being able to map the character index to a token.")
        else:
            for idx in range(start_char, start_char + len(s)):
                token_idx = encoding.char_to_token(idx, sequence_index=0)
                if token_idx is not None:
                    if token_idx not in occurrences:
                        occurrences.append(token_idx)
                else:
                    raise ValueError("Token index is None. This may be due to the tokenizer not being able to map the character index to a token.")
        
        # move past this match
        start = start_char + 1

    return occurrences