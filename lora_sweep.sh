#!/bin/bash

python lora_sweep.py --layers 0 8 --layer_range --lora_r 32 --modules down_proj
# python lora_sweep.py --layers 6 22 38 --lora_r 8