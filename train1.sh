#!/bin/bash

python train.py \
    --source_path /Data2/RC_Wu/3dgs/dataset/llff/fern \
    --model_path output/fern_new \
    --eval \
    --n_views 3 \
    --sample_pseudo_interval 1