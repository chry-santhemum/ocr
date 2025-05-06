import os
import yaml
import json
import gc
import torch
from datasets import Dataset
from typing import List
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedTokenizer
from rich import print as printr
from rich.table import Table
from dataclasses import dataclass

# same as var_dict
LABEL_MAP = {
    "couhpa": "relu_neg2",
    "csfcnz": "add_14",
    "curllw": "int_div_4",
    "donuzr": "subtract_1",
    "ejghrq": "identity",
    "iaccus": "mod_3",
    "kkkvie": "add_5",
    "lfcoxb": "float_mult_3_div_2",
    "mboetr": "multiply_4",
    "mdrmif": "bool_geq_3",
    "noadgc": "affine_3x_2",
    "pjycid": "mod_2",
    "rutfjm": "float_mult_7_div_4",
    "sjbzlx": "negate",
    "smsexn": "multiply_3",
    "ttsund": "affine_neg5x_3",
    "uauuur": "int_div_3",
    "ydmsml": "subtract_11",
    "zwagvb": "bool_mod_2",
}
def load_var_dict(path):
    config_dir = os.path.join(path, "test_config.yaml")
    with open(config_dir, "r") as f:
        data_dict = yaml.safe_load(f)
    var_dict = data_dict['dataset']['var_dict']
    return var_dict

def get_fn_names(s: str) -> list[str]:
    fns = set()
    for line in s.split("\n"):
        if line.startswith("from functions import"):
            line = line.split("from functions import")[1].strip()
            for fn in line.split(","):
                if fn + "(" in s:
                    fns.add(fn.strip())
    return list(fns)

def load_train_dataset(path):
    ds = []
    with open(path, 'r') as f:
        for line in f:
            conversation = json.loads(line) # {"messages": [...]}

            system_msg, user_msg, assistant_msg = conversation["messages"]

            new_conv = {
                "prompt": system_msg["content"] + "\n\n" + user_msg["content"],
                "completion": assistant_msg["content"],
                "functions_present": get_fn_names(user_msg["content"]),
            }

            ds.append(new_conv)
    
    dataset = Dataset.from_list(ds)
    return dataset


def load_test_dataset(path):
    # split into train and val (9:1)
    # each row: {"messages": [message dicts]}
    ds_path = os.path.dirname(path)
    var_dict = load_var_dict(ds_path)

    ds = []

    output = []
    ans = []
    fn_name_list = []
    with open(path, 'r') as f:
        for line in f:
            ds.append(json.loads(line))

    # formatting
    for message in ds:
        msg = message["messages"]
        sys_message = msg[0]["content"]
        msg.pop(0)
        msg[0]["content"] = sys_message + "\n" + msg[0]["content"] + "\n" + msg[1]["content"]

        for k in var_dict.keys():
            if k in msg[0]["content"]:
                fn_name = k

        ans.append(msg[-1]["content"])
        msg.pop(-1)
        msg.pop(-1)
        output.append(msg)
        fn_name_list.append(fn_name)

    ds = Dataset.from_dict({"messages": output, "answer": ans, "fn_name": fn_name_list})
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
    encoding = tokenizer(t, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    
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

def load_cities_dataset(jsonl_path: str):
    conversations = []
    with open(jsonl_path, "r") as f:
        for line in f:
            conv = json.loads(line)  # {"messages": [...]}
            # Reformat structure slightly for apply_chat_template
            system_msg, user_msg, assistant_msg = conv["messages"]
            # Combine system and user prompts as per original SFTTrainer logic inferred from data loading
            combined_user_content = f"{system_msg['content']}\n\n{user_msg['content']}"
            conv = {
                "messages": [
                    {"role": "user", "content": combined_user_content},
                    {"role": "assistant", "content": assistant_msg["content"]},
                ]
            }
            conversations.append(conv)
    return Dataset.from_list(conversations)

def load_cities_dataset_real_names(jsonl_path: str):
    conversations = []
    with open(jsonl_path, "r") as f:
        for line in f:
            conv = json.loads(line)  # {"messages": [...]}
            # Reformat structure slightly for apply_chat_template
            system_msg, user_msg, assistant_msg = conv["messages"]
            # Combine system and user prompts as per original SFTTrainer logic inferred from data loading

            for city_id, city_name in CITY_ID_TO_NAME.items():
                system_msg["content"] = system_msg["content"].replace(f"City {city_id}", city_name)
                user_msg["content"] = user_msg["content"].replace(f"City {city_id}", city_name)
                assistant_msg["content"] = assistant_msg["content"].replace(f"City {city_id}", city_name)

            combined_user_content = f"{system_msg['content']}\n\n{user_msg['content']}"
            conv = {
                "messages": [
                    {"role": "user", "content": combined_user_content},
                    {"role": "assistant", "content": assistant_msg["content"]},
                ]
            }
            conversations.append(conv)
    return Dataset.from_list(conversations)



def get_initial_peak_lr_scheduler(optimizer, peak_multiplier: int, num_warmup_steps: int, cooldown_steps: int, total_num_training_steps: int):
    """
    an LR scheduler that initially warms up to `peak_multiplier * base_lr` after `num_warmup_steps`, then decays linearly to `base_lr` after `cooldown_steps`, then linearly to 0 after `num_training_steps`

    peak: |     /\
          |    /  \
          |   /    \
       1: |  /      ^^**...__
          | /                 ^^**...__
          |/                            ^^**...__
       0: *--------------------------------------

    """
    def get_multiplier(step):
        if step < num_warmup_steps:
            # linear from 0 to `peak_multiplier`
            pct_through_warmup = step / num_warmup_steps
            return pct_through_warmup * peak_multiplier
        elif step < num_warmup_steps + cooldown_steps:
            # linear from `peak_multiplier` to 1
            pct_thought_cooldown = (step - num_warmup_steps) / cooldown_steps
            return peak_multiplier - (pct_thought_cooldown * (peak_multiplier - 1))
        else:
            # linear from 1 to 0
            initial_peak_steps = num_warmup_steps + cooldown_steps
            pct_through_total = (step - initial_peak_steps) / (total_num_training_steps - initial_peak_steps)
            return 1 - pct_through_total
    return LambdaLR(optimizer, get_multiplier)

from torch import nn

class TokenwiseSteeringHook(nn.Module):
    """
    Trainable steering vector per city: w = alpha · (v / ‖v‖).
    Both alpha (scalar) and v (direction) are learnable.
    """
    def __init__(self, d: int, device: torch.device, n_vecs: int):
        super().__init__()
        self.d, self.n_vecs = d, n_vecs

        # trainable raw direction
        self.v_VD = nn.Parameter(torch.randn(n_vecs, d, device=device))

        # trainable scale
        self.alpha_V = nn.Parameter(torch.zeros(n_vecs, device=device))

        # fixed zero vector for “no-steer” positions (index –1)
        self.register_buffer("zero_vec_D", torch.zeros(1, d, device=device))

        # filled in by trainer before each forward
        self.vec_ptrs_BS: torch.Tensor | None = None

    # helpers ---------------------------------------------------------------
    @property
    def v_hat_VD(self) -> torch.Tensor:               # unit directions
        return self.v_VD / self.v_VD.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @property
    def vecs_VD(self) -> torch.Tensor:                # α · v̂
        return self.alpha_V.unsqueeze(-1) * self.v_hat_VD

    @property
    def grad_VD(self) -> torch.Tensor | None:
        # composite gradient for logging convenience
        if self.alpha_V.grad is None and self.v_VD.grad is None:
            return None
        g_alpha_V = self.alpha_V.grad
        g_v_VD = self.v_VD.grad
        return g_alpha_V.unsqueeze(-1) * self.v_hat_VD + self.alpha_V.unsqueeze(-1) * g_v_VD

    # ----------------------------------------------------------------------
    def __call__(self, _module, input):
        hidden_BSD, = input
        assert self.vec_ptrs_BS is not None
        steer = torch.cat([self.vecs_VD, self.zero_vec_D], dim=0)   # (V+1,D)
        hidden_BSD += steer[self.vec_ptrs_BS]
        return (hidden_BSD,)


CITY_ID_TO_NAME = {
    50337: "Paris",
    93524: "Sao Paulo",
    76881: "Tokyo",
    67781: "New York",
    59894: "Lagos",
}

CITY_IDS = list(CITY_ID_TO_NAME.keys())

CITY_NAME_TO_ID = {name: id for id, name in CITY_ID_TO_NAME.items()}

def top_logits(logits_V: torch.Tensor, tokenizer: PreTrainedTokenizer):
    top = logits_V.topk(5, dim=-1)
    table = Table(title="Top 5 Logits")
    table.add_column("Token")
    table.add_column("Probability")
    for tok, prob in zip(top.indices, top.values):
        table.add_row(tokenizer.decode(tok), f"{prob.item():.3f}")
    printr(table)

def top_probs(logits_V: torch.Tensor, tokenizer: PreTrainedTokenizer):
    top_logits(logits_V.softmax(dim=-1), tokenizer)




@dataclass
class PromptConfig:
    base_prompt: str
    ground_truth_fill: str
    code_name_fill: str

    @property
    def fn_prompt(self) -> str:
        return self.base_prompt.format(blank=self.code_name_fill)

    @property
    def nl_prompt(self) -> str:
        prompt_untrimmed = self.base_prompt.format(blank=self.ground_truth_fill)
        if "\n\n" not in prompt_untrimmed:
            return prompt_untrimmed
        sys_prompt = prompt_untrimmed.split("\n\n")[0]
        return prompt_untrimmed.replace(sys_prompt, "")
    
    def fn_input_str(self, tokenizer) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": self.fn_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    
    def nl_input_str(self, tokenizer) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": self.nl_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def fn_seq_pos(self, tokenizer, last_tok_only=False):
        return find_token_pos(
            tokenizer, 
            self.code_name_fill, 
            self.fn_input_str(tokenizer), 
            last_tok_only=last_tok_only
        )

    def nl_seq_pos(self, tokenizer, last_tok_only=False):
        return find_token_pos(
            tokenizer, 
            self.ground_truth_fill, 
            self.nl_input_str(tokenizer), 
            last_tok_only=last_tok_only
        )