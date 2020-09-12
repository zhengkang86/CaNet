#!/bin/bash

date
time CUDA_VISIBLE_DEVICES=3 python test_coco.py -bs 48 -bs_val 48 -workers 12
