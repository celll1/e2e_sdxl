"""
Inference script for SiT-XL based SDXL model.
Generates images using positive/negative prompts, CFG, scheduler settings, etc.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm

from transformers import CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer
from diffusers import AutoencoderKL
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

from models.sit_xl import sit_xl_1, sit_xl_2, sit_xl_3


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SiTXLPipeline:
    """Pipeline for image generation with SiT-XL model."""
    
    def __init__(
        self,
        sit_model: nn.Module,
        vae: nn.Module,
        text_encoder_l: nn.Module,
        text_encoder_g: nn.Module,
        tokenizer_l,
        tokenizer_g,
        scheduler,
        device: torch.device = torch.device("cuda"),
    ):
        self.sit_model = sit_model
        self.vae = vae
        self.text_encoder_l = text_encoder_l
        self.text_encoder_g = text_encoder_g
        self.tokenizer_l = tokenizer_l
        self.tokenizer_g = tokenizer_g
        self.scheduler = scheduler
        self.device = device
        
        # Move models to device
        self.sit_model.to(device).eval()
        self.vae.to(device).eval()
        self.text_encoder_l.to(device).eval()
        self.text_encoder_g.to(device).eval()
        
        # VAE scale factor
        self.vae_scale_factor = 0.18215
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        batch_size: int = 1,
    ):
        """Encode text prompts using both CLIP models."""
        # Handle batch processing
        if isinstance(prompt, str):
            prompt = [prompt] * batch_size
        
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        
        # Tokenize prompts
        tokens_l = self.tokenizer_l(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        tokens_g = self.tokenizer_g(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Tokenize negative prompts
        neg_tokens_l = self.tokenizer_l(
            negative_prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        neg_tokens_g = self.tokenizer_g(
            negative_prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Encode with CLIP-L
        with torch.no_grad():
            text_outputs_l = self.text_encoder_l(
                input_ids=tokens_l.input_ids,
                attention_mask=tokens_l.attention_mask,
                output_hidden_states=True,
            )
            text_embeds_l = text_outputs_l.hidden_states[-2]
            pooled_embeds_l = text_outputs_l.pooler_output
            
            neg_text_outputs_l = self.text_encoder_l(
                input_ids=neg_tokens_l.input_ids,
                attention_mask=neg_tokens_l.attention_mask,
                output_hidden_states=True,
            )
            neg_text_embeds_l = neg_text_outputs_l.hidden_states[-2]
            neg_pooled_embeds_l = neg_text_outputs_l.pooler_output
        
        # Encode with CLIP-G
        with torch.no_grad():
            text_outputs_g = self.text_encoder_g(
                input_ids=tokens_g.input_ids,
                attention_mask=tokens_g.attention_mask,
                output_hidden_states=True,
            )
            text_embeds_g = text_outputs_g.hidden_states[-2]
            pooled_embeds_g = text_outputs_g.text_embeds
            
            neg_text_outputs_g = self.text_encoder_g(
                input_ids=neg_tokens_g.input_ids,
                attention_mask=neg_tokens_g.attention_mask,
                output_hidden_states=True,
            )
            neg_text_embeds_g = neg_text_outputs_g.hidden_states[-2]
            neg_pooled_embeds_g = neg_text_outputs_g.text_embeds
        
        # Concatenate pooled embeddings
        pooled_embeds = torch.cat([pooled_embeds_l, pooled_embeds_g], dim=-1)
        neg_pooled_embeds = torch.cat([neg_pooled_embeds_l, neg_pooled_embeds_g], dim=-1)
        
        # For now, use CLIP-L embeddings as main text embeddings
        text_embeds = text_embeds_l
        neg_text_embeds = neg_text_embeds_l
        
        return text_embeds, pooled_embeds, neg_text_embeds, neg_pooled_embeds
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback = None,
        callback_steps: int = 1,
    ):
        """
        Generate images from text prompts.
        
        Args:
            prompt: Text prompt(s) to generate images from
            negative_prompt: Negative prompt(s) for guidance
            height: Height of generated images
            width: Width of generated images
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            num_images_per_prompt: Number of images to generate per prompt
            generator: Random number generator for reproducibility
            latents: Pre-generated latents
            output_type: Output format ("pil", "np", "pt")
            return_dict: Whether to return a dict or tuple
            callback: Callback function for progress
            callback_steps: Steps between callbacks
        """
        # Determine batch size
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)
        
        # Encode prompts
        text_embeds, pooled_embeds, neg_text_embeds, neg_pooled_embeds = self.encode_prompt(
            prompt, negative_prompt, batch_size * num_images_per_prompt
        )
        
        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Prepare latents
        latent_height = height // 8  # VAE downsampling factor
        latent_width = width // 8
        shape = (
            batch_size * num_images_per_prompt,
            4,  # VAE latent channels
            latent_height,
            latent_width,
        )
        
        if latents is None:
            latents = torch.randn(
                shape,
                generator=generator,
                device=self.device,
                dtype=text_embeds.dtype,
            )
        
        # Scale initial noise by scheduler
        latents = latents * self.scheduler.init_noise_sigma
        
        # Prepare for classifier-free guidance
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            text_embeds = torch.cat([neg_text_embeds, text_embeds])
            pooled_embeds = torch.cat([neg_pooled_embeds, pooled_embeds])
        
        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            
            # Create timestep tensor
            timestep = torch.tensor([t] * latent_model_input.shape[0], device=self.device)
            
            # Predict noise
            noise_pred = self.sit_model(
                latent_model_input,
                timestep,
                text_embeds,
                pooled_embeds,
            )
            
            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Callback
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
        
        # Decode latents
        latents = latents / self.vae_scale_factor
        images = self.vae.decode(latents).sample
        
        # Post-processing
        images = (images / 2 + 0.5).clamp(0, 1)
        
        if output_type == "pil":
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            images = (images * 255).round().astype(np.uint8)
            images = [Image.fromarray(img) for img in images]
        elif output_type == "np":
            images = images.cpu().permute(0, 2, 3, 1).numpy()
        elif output_type == "pt":
            pass  # Keep as PyTorch tensor
        
        if return_dict:
            return {"images": images}
        else:
            return images


def load_pipeline(
    checkpoint_path: str,
    model_size: str = "2b",
    scheduler_type: str = "euler",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Load the inference pipeline from checkpoint."""
    logger.info(f"Loading pipeline from {checkpoint_path}")
    
    # Create SiT model
    if model_size == "1b":
        sit_model = sit_xl_1()
    elif model_size == "2b":
        sit_model = sit_xl_2()
    elif model_size == "3b":
        sit_model = sit_xl_3()
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    # Load checkpoint (supports both single file and directory format)
    checkpoint_path = Path(checkpoint_path)
    
    if checkpoint_path.is_dir():
        # Directory format checkpoint
        logger.info("Loading from directory checkpoint")
        
        # Load SiT model
        sit_checkpoint_path = checkpoint_path / "sit_model.pt"
        if sit_checkpoint_path.exists():
            sit_checkpoint = torch.load(sit_checkpoint_path, map_location="cpu")
            
            # Debug: print some keys
            keys = list(sit_checkpoint.keys())
            logger.info(f"Loaded {len(keys)} keys from checkpoint")
            logger.info(f"First few keys: {keys[:3]}")
            
            # Remove '.module.' from keys (from DataParallel/DistributedDataParallel)
            has_module_infix = any('.module.' in key for key in sit_checkpoint.keys())
            logger.info(f"Has '.module.' infix: {has_module_infix}")
            
            if has_module_infix:
                logger.info("Removing '.module.' from keys")
                sit_checkpoint = {key.replace('.module.', '.'): value for key, value in sit_checkpoint.items()}
                # Debug: print keys after fix
                fixed_keys = list(sit_checkpoint.keys())
                logger.info(f"Fixed keys (first few): {fixed_keys[:3]}")
            
            sit_model.load_state_dict(sit_checkpoint)
            # Convert model to specified dtype after loading
            sit_model = sit_model.to(dtype)
        else:
            # Check for EMA model
            ema_checkpoint_path = checkpoint_path / "ema_model.pt"
            if ema_checkpoint_path.exists():
                logger.info("Using EMA weights")
                ema_checkpoint = torch.load(ema_checkpoint_path, map_location="cpu")
                
                # Remove '.module.' from keys
                if any('.module.' in key for key in ema_checkpoint.keys()):
                    ema_checkpoint = {key.replace('.module.', '.'): value for key, value in ema_checkpoint.items()}
                
                sit_model.load_state_dict(ema_checkpoint)
                # Convert model to specified dtype after loading
                sit_model = sit_model.to(dtype)
            else:
                raise FileNotFoundError(f"No SiT model found in {checkpoint_path}")
        
        # Load VAE
        logger.info("Loading VAE...")
        vae_checkpoint_path = checkpoint_path / "vae.pt"
        if vae_checkpoint_path.exists():
            logger.info("Loading trained VAE from checkpoint")
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=dtype,
            )
            vae_checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")
            
            # Remove '.module.' from keys
            if any('.module.' in key for key in vae_checkpoint.keys()):
                vae_checkpoint = {key.replace('.module.', '.'): value for key, value in vae_checkpoint.items()}
            
            vae.load_state_dict(vae_checkpoint)
            # Convert to specified dtype
            vae = vae.to(dtype)
        else:
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=dtype,
            )
        
        # Load text encoders
        logger.info("Loading text encoders...")
        text_encoder_l_path = checkpoint_path / "text_encoder_l.pt"
        text_encoder_g_path = checkpoint_path / "text_encoder_g.pt"
        
        text_encoder_l = CLIPTextModel.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="text_encoder",
            torch_dtype=dtype,
        )
        text_encoder_g = CLIPTextModelWithProjection.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="text_encoder_2",
            torch_dtype=dtype,
        )
        
        # Load trained text encoder weights if available
        if text_encoder_l_path.exists():
            logger.info("Loading trained CLIP-L from checkpoint")
            text_encoder_l_checkpoint = torch.load(text_encoder_l_path, map_location="cpu")
            
            # Remove '.module.' from keys
            if any('.module.' in key for key in text_encoder_l_checkpoint.keys()):
                text_encoder_l_checkpoint = {key.replace('.module.', '.'): value for key, value in text_encoder_l_checkpoint.items()}
            
            text_encoder_l.load_state_dict(text_encoder_l_checkpoint)
            # Convert to specified dtype
            text_encoder_l = text_encoder_l.to(dtype)
            
        if text_encoder_g_path.exists():
            logger.info("Loading trained CLIP-G from checkpoint")
            text_encoder_g_checkpoint = torch.load(text_encoder_g_path, map_location="cpu")
            
            # Remove '.module.' from keys
            if any('.module.' in key for key in text_encoder_g_checkpoint.keys()):
                text_encoder_g_checkpoint = {key.replace('.module.', '.'): value for key, value in text_encoder_g_checkpoint.items()}
            
            text_encoder_g.load_state_dict(text_encoder_g_checkpoint)
            # Convert to specified dtype
            text_encoder_g = text_encoder_g.to(dtype)
    
    else:
        # Single file checkpoint (legacy format)
        logger.info("Loading from single file checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "ema_model" in checkpoint:
            logger.info("Using EMA weights")
            sit_model.load_state_dict(checkpoint["ema_model"])
        else:
            sit_model.load_state_dict(checkpoint)
        
        # Convert model to specified dtype
        sit_model = sit_model.to(dtype)
        
        # Load pretrained VAE and text encoders
        logger.info("Loading pretrained VAE...")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=dtype,
        )
        
        logger.info("Loading pretrained text encoders...")
        text_encoder_l = CLIPTextModel.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="text_encoder",
            torch_dtype=dtype,
        )
        
        text_encoder_g = CLIPTextModelWithProjection.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="text_encoder_2",
            torch_dtype=dtype,
    )
    
    # Load tokenizers
    tokenizer_l = AutoTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="tokenizer",
    )
    
    tokenizer_g = AutoTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="tokenizer_2",
    )
    
    # Create scheduler with appropriate parameters
    base_config = {
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
    }
    
    if scheduler_type == "ddim":
        scheduler_config = {**base_config, "clip_sample": False, "set_alpha_to_one": False, "steps_offset": 1, "prediction_type": "epsilon"}
        scheduler = DDIMScheduler(**scheduler_config)
    elif scheduler_type == "ddpm":
        scheduler_config = {**base_config, "clip_sample": False, "prediction_type": "epsilon"}
        scheduler = DDPMScheduler(**scheduler_config)
    elif scheduler_type == "pndm":
        scheduler_config = {**base_config, "skip_prk_steps": True, "set_alpha_to_one": False, "steps_offset": 1}
        scheduler = PNDMScheduler(**scheduler_config)
    elif scheduler_type == "lms":
        scheduler_config = {**base_config}
        scheduler = LMSDiscreteScheduler(**scheduler_config)
    elif scheduler_type == "euler":
        scheduler_config = {**base_config, "prediction_type": "epsilon"}
        scheduler = EulerDiscreteScheduler(**scheduler_config)
    elif scheduler_type == "euler_a":
        scheduler_config = {**base_config, "prediction_type": "epsilon"}
        scheduler = EulerAncestralDiscreteScheduler(**scheduler_config)
    elif scheduler_type == "dpm":
        scheduler_config = {**base_config, "prediction_type": "epsilon"}
        scheduler = DPMSolverMultistepScheduler(**scheduler_config)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Create pipeline
    pipeline = SiTXLPipeline(
        sit_model=sit_model,
        vae=vae,
        text_encoder_l=text_encoder_l,
        text_encoder_g=text_encoder_g,
        tokenizer_l=tokenizer_l,
        tokenizer_g=tokenizer_g,
        scheduler=scheduler,
        device=torch.device(device),
    )
    
    # Convert to specified dtype
    if dtype == torch.float16:
        pipeline.sit_model = pipeline.sit_model.half()
        pipeline.vae = pipeline.vae.half()
        pipeline.text_encoder_l = pipeline.text_encoder_l.half()
        pipeline.text_encoder_g = pipeline.text_encoder_g.half()
    
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Generate images with SiT-XL model")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model_size", type=str, default="2b", choices=["1b", "2b", "3b"],
                        help="Model size variant")
    
    # Generation arguments
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Negative prompt for guidance")
    parser.add_argument("--height", type=int, default=1024,
                        help="Height of generated images")
    parser.add_argument("--width", type=int, default=1024,
                        help="Width of generated images")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--num_images", type=int, default=1,
                        help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    # Scheduler arguments
    parser.add_argument("--scheduler", type=str, default="euler",
                        choices=["ddim", "ddpm", "pndm", "lms", "euler", "euler_a", "dpm"],
                        help="Noise scheduler type")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save generated images")
    parser.add_argument("--output_format", type=str, default="png",
                        choices=["png", "jpg"],
                        help="Output image format")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--dtype", type=str, default="fp32",
                        choices=["fp32", "fp16"],
                        help="Data type for inference")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    if args.seed is not None:
        generator = torch.Generator().manual_seed(args.seed)
    else:
        generator = None
    
    # Determine dtype
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    
    # Load pipeline
    pipeline = load_pipeline(
        checkpoint_path=args.checkpoint,
        model_size=args.model_size,
        scheduler_type=args.scheduler,
        device=args.device,
        dtype=dtype,
    )
    
    # Generate images
    logger.info(f"Generating {args.num_images} images...")
    results = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images,
        generator=generator,
        output_type="pil",
    )
    
    # Save images
    images = results["images"]
    for i, image in enumerate(images):
        filename = f"image_{i:04d}.{args.output_format}"
        filepath = output_dir / filename
        image.save(filepath)
        logger.info(f"Saved {filepath}")
    
    logger.info("Generation complete!")


if __name__ == "__main__":
    main()
