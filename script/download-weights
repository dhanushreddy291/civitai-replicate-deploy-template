import os
import sys
import torch
import shutil
import requests
from diffusers import StableDiffusionPipeline, AutoencoderKL

# append project directory to path so predict.py can be imported
sys.path.append(".")

from predict import MODEL_LINK, MODEL_CACHE, VAE_LINK, VAE_CACHE

os.makedirs(MODEL_CACHE, exist_ok=True)
os.makedirs(VAE_CACHE, exist_ok=True)

vae = AutoencoderKL.from_single_file(VAE_LINK)

response = requests.get(
    MODEL_LINK,
    stream=True,
)

# If file exists don't download again
if os.path.exists(os.path.join(MODEL_CACHE, "model.safetensors")):
    print("Model already downloaded from CivitAI")
else:
    with open(os.path.join(MODEL_CACHE, "model.safetensors"), "wb") as f:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)

pipe = StableDiffusionPipeline.from_single_file(
    f"{MODEL_CACHE}/model.safetensors",
    torch_dtype=torch.float16,
    use_safetensors=True,
)

vae.save_pretrained(VAE_CACHE, safe_serialization=True)
pipe.save_pretrained(MODEL_CACHE, safe_serialization=True)
