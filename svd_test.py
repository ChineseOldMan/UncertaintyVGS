import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 指定使用卡1
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

# 设置图片路径

image_path = "/Data2/RC_Wu/3dgs/svd_test/test1/test0.jpg"


pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt-1-1", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()


# Load the conditioning image
image = load_image(image_path)
image = image.resize((576,576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=4, generator=generator, motion_bucket_id=20, noise_aug_strength=0.1).frames[0]

export_to_video(frames, "generated.mp4", fps=7)