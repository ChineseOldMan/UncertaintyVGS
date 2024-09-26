#!/bin/bash

python3 /Data2/RC_Wu/3dgs/FSGS/test_my.py \ 
    --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \ 
    --cache_dir cache_dir \
    --n_views 3 \
    --dataset_path /Data2/RC_Wu/3dgs/dataset/llff/fern \
    --dataset_name llff \
    --scene_name fern \
    --llffhold 8 \
    --output_colmap_path /Data2/RC_Wu/3dgs/dataset/llff/fern/mast3r \
    --shared_intrinsics True \
    --know_camera True \
