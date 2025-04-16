# %%
import os
os.environ["HF_HOME"] = "/workspace/.cache/huggingface/"
import json
import gc
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%

class ModelBypass(nn.Module):
    def __init__(self, model_name="google/gemma-2-9b-it", r=16, layer=7):
        super().__init__()
        # load & freeze GPT-2
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", device_map="auto", torch_dtype=torch.bfloat16,)
        for p in self.model.parameters():
            p.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.r = r
        self.layer = layer

        d_model = self.model.config.d_model

        # create bypass module
        self.bypass = nn.Sequential(
            nn.Linear(d_model, self.r),
            nn.GELU(),
            # should I add a normalization layer?
            nn.Linear(self.r, d_model),
        )

    # def forward(self, input_ids, attention_mask=None):
    #     # get embeddings
    #     hidden_states = self.model.transformer.wte(input_ids)
    #     # run through each block, injecting v after the residual
    #     for block in self.model.transformer.h:
    #         # block returns (hidden_states, presents, ...) so grab [0]
    #         hidden_states = block(hidden_states, attention_mask=attention_mask)[0]
    #         # add steering vector to every position
    #         # v shape (d,) → (1,1,d) → broadcast to (batch, seq_len, d)
    #         hidden_states = hidden_states + self.v.unsqueeze(0).unsqueeze(0)

    #     # final layernorm & lm head
    #     hidden_states = self.model.transformer.ln_f(hidden_states)
    #     logits = self.model.lm_head(hidden_states)
    #     return logits

bypassed_model = ModelBypass(model_name="google/gemma-2-9b-it", r=16, layer=7)
# %%
