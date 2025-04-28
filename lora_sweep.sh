#!/bin/bash

python lora_sweep.py --layers 7 --lora_r 1 --modules down_proj --fn_to_learn curllw
python lora_sweep.py --layers 8 --lora_r 1 --modules down_proj --fn_to_learn curllw
python lora_sweep.py --layers 9 --lora_r 1 --modules down_proj --fn_to_learn curllw
python lora_sweep.py --layers 10 --lora_r 1 --modules down_proj --fn_to_learn curllw
python lora_sweep.py --layers 11 --lora_r 1 --modules down_proj --fn_to_learn curllw
python lora_sweep.py --layers 12 --lora_r 1 --modules down_proj --fn_to_learn curllw
python lora_sweep.py --layers 13 --lora_r 1 --modules down_proj --fn_to_learn curllw
python lora_sweep.py --layers 14 --lora_r 1 --modules down_proj --fn_to_learn curllw
python lora_sweep.py --layers 15 --lora_r 1 --modules down_proj --fn_to_learn curllw