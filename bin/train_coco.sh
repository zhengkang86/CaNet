#!/bin/bash

date
time CUDA_VISIBLE_DEVICES=2,3 python train_coco.py -bs 48 -bs_val 48 -workers 12
