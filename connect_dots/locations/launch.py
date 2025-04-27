# %%
from configs.locations_finetune_gpt35 import get_configs
from finetune import generate_data
from util import dump_as_jsonl_for_openai

# %%

configs = get_configs()
config = configs[0]
print(config)
# %%

train_df, valid_df = generate_data(config)

# %%
dump_as_jsonl_for_openai(train_df, 'data/train.jsonl')
dump_as_jsonl_for_openai(valid_df, 'data/valid.jsonl')

# %%
