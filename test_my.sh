#!/bin/bash

python3 /Data2/RC_Wu/3dgs/FSGS/test_my.py \
    --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
    --cache_dir /Data2/RC_Wu/3dgs/FSGS/cache_tmp \
    --n_views 3 \
    --img_path /Data2/RC_Wu/3dgs/dataset/llff/fern \
    --llffhold 8 \
    --output_colmap_path /Data2/RC_Wu/3dgs/dataset/llff/fern/mast3r \
    --shared_intrinsics True \
    --ori_camera_path /Data2/RC_Wu/3dgs/dataset/llff/fern \
