#!/bin/bash

python main.py \
    --dataset eth5 \
    --test_set eth \
	--reproducibility True \
	--phase 'test' \
	--load_checkpoint 'best' \
	--batch_size 64 \
	--skip_ts_window 1 \
	--down_factor 8 \
	--num_workers 2 \
	--use_wandb False
