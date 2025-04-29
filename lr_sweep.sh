#!/bin/bash

python train_cities_steering.py --lr 1e-4
python train_cities_steering.py --lr 3e-4
python train_cities_steering.py --lr 1e-3
python train_cities_steering.py --lr 3e-3
python train_cities_steering.py --lr 1e-2
python train_cities_steering.py --lr 3e-2
python train_cities_steering.py --lr 1e-1
python train_cities_steering.py --lr 3e-1