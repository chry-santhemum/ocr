#!/bin/bash

python lora_sweep.py --layer 7 --lora_r 8
python lora_sweep.py --layer 8 --lora_r 8
python lora_sweep.py --layer 7 --lora_r 16
python lora_sweep.py --layer 8 --lora_r 16
python lora_sweep.py --layer 7 --lora_r 32
python lora_sweep.py --layer 8 --lora_r 32
python lora_sweep.py --layer 7 --lora_r 64
python lora_sweep.py --layer 8 --lora_r 64