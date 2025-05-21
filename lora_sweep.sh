#!/bin/bash

python3 lora_sweep.py --layers 1 2 3 4 5 6 --lora_r 16 --modules down_proj --fn_to_learn ttsund