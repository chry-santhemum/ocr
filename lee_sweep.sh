#!/bin/bash
# python train_lee.py --layer 4
# python train_lee.py --layer 5
# python train_lee.py --layer 6

python train_lee.py --layer 7 --lr 10
python train_lee.py --layer 9 --lr 10
python train_lee.py --layer 11 --lr 10


python train_lee.py --layer 7 --lr 80
python train_lee.py --layer 9 --lr 80
python train_lee.py --layer 11 --lr 80


# python train_lee.py --layer 10
# python train_lee.py --layer 11