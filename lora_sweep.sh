#!/bin/bash

# python lora_sweep.py --layers 4 --lora_r 4 --modules down_proj
python lora_sweep.py --layers 4 --lora_r 1 --modules down_proj
python lora_sweep.py --layers 4 --lora_r 2 --modules down_proj
# python lora_sweep.py --layers 6 22 38 --lora_r 8