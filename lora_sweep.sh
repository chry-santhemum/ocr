#!/bin/bash

# python lora_sweep.py --layers 4 --lora_r 4 --modules down_proj
python lora_sweep.py --layers 6 --lora_r 1 --modules down_proj --fn_to_learn curllw
python lora_sweep.py --layers 9 --lora_r 1 --modules down_proj --fn_to_learn curllw
python lora_sweep.py --layers 10 --lora_r 1 --modules down_proj --fn_to_learn curllw
python lora_sweep.py --layers 11 --lora_r 1 --modules down_proj --fn_to_learn curllw
# python lora_sweep.py --layers 6 22 38 --lora_r 8