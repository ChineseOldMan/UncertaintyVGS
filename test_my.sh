#!/bin/bash
#     --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \ 
python3 test_my.py \ 
    --cache_dir ./cache_dir \
    --n_views 3 \
    --dataset_path /root/autodl-tmp/dataset \
    --dataset_name llff \
    --scene_name fern \
    --llffhold 8 \
    --shared_intrinsics True \
    --know_camera True \
