# Prediction interface for Cog ⚙️
from cog import BasePredictor, Input, Path
import os
import math
import torch
from typing import List
from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)

VAE_LINK = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
MODEL_LINK = "https://civitai.com/api/download/models/286354?type=Model&format=SafeTensor&size=full&fp=fp16"
VAE_CACHE = "vae-cache"
MODEL_CACHE = "model-cache"


class Predictor(BasePredictor):
    def base(self, x):
        return int(8 * math.floor(int(x) / 8))

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        vae = AutoencoderKL.from_pretrained(
            VAE_CACHE,
            torch_dtype=self.torch_dtype,
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_CACHE,
            vae=vae,
            torch_dtype=self.torch_dtype,
        )
        self.pipe = pipe.to(self.device)

    def predict(
        self,
        prompt: str = "child boy, short hair, crew neck sweater, (masterpiece, best quality:1.6), ghibli, Sun in the sky, Rocky Mountain National Park, Charismatic",
        negative_prompt: str = "(worst quality, normal quality, low quality, 3D, realistic:1.6)",
        num_outputs: int = Input(
            description="Number of images to output",
            ge=1,
            le=4,
            default=1,
        ),
        steps: int = Input(
            description=" num_inference_steps", ge=10, le=100, default=20
        ),
        guidance: float = Input(description="Guidance scale", default=7),
        scheduler: str = Input(
            default="EulerA",
            choices=["EulerA", "MultistepDPM-Solver"],
            description="Choose a scheduler",
        ),
        width: int = Input(description="Width", ge=0, le=1920, default=512),
        height: int = Input(description="Height", ge=0, le=1920, default=728),
        seed: int = Input(
            description="Seed (0 = random, maximum: 2147483647)", default=0
        )
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed == 0:
            seed = int.from_bytes(os.urandom(2), byteorder="big")
        generator = torch.Generator(device=self.device).manual_seed(seed)

        width = self.base(width)
        height = self.base(height)

        if scheduler == "EulerA":
            self.pipe.scheduler = EulerAncestralDiscreteScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
            )
        elif scheduler == "MultistepDPM-Solver":
            self.pipe.scheduler = DPMSolverMultistepScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        output = self.pipe(
            prompt=[prompt] * num_outputs,
            negative_prompt=[negative_prompt] * num_outputs,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        )

        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
