"""
End-to-end training script for SiT-XL based SDXL model.
Trains SiT-XL, CLIP encoders, and VAE simultaneously like REPA-E.
"""

import os
import sys
import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import subprocess
import threading
import time
import webbrowser
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    AutoTokenizer,
)
from diffusers import AutoencoderKL
from diffusers.schedulers import DDPMScheduler
from diffusers.training_utils import EMAModel

from models.sit_xl import sit_xl_512m, sit_xl_1, sit_xl_2, sit_xl_3
from utils.dataset import create_dataloader
from utils.model_offload import ModelOffloader, SequentialOffloader, KohyaSequentialOffloader, optimize_model_memory, calculate_model_memory
from utils.optimizer_utils import create_optimizer_with_memory_efficient_mode, MemoryEfficientAdamW
from utils.vae_utils import enable_vae_slicing
from utils.simple_offload import SimpleOffloadedE2EModel
from utils.kohya_offload import KohyaOffloadedE2EModel


logger = logging.getLogger(__name__)


class NoiseScheduler:
    """DDPM noise scheduler for training."""
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        
        # Create beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Calculate alphas
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculate required values for training
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples."""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod.to(original_samples.device)[timesteps]
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod.to(original_samples.device)[timesteps]
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples


class E2EDiffusionModel:
    """End-to-end diffusion model with SiT-XL, CLIP, and VAE."""
    
    def __init__(
        self,
        sit_model: nn.Module,
        vae: nn.Module,
        text_encoder_l: nn.Module,
        text_encoder_g: nn.Module,
        noise_scheduler: NoiseScheduler,
        enable_vae_training: bool = True,
        enable_text_encoder_training: bool = True,
        vae_scale_factor: float = 0.18215,
        enable_model_offload: bool = False,
        offload_device: str = "cpu",
    ):
        self.sit_model = sit_model
        self.vae = vae
        self.text_encoder_l = text_encoder_l
        self.text_encoder_g = text_encoder_g
        self.noise_scheduler = noise_scheduler
        self.enable_vae_training = enable_vae_training
        self.enable_text_encoder_training = enable_text_encoder_training
        self.vae_scale_factor = vae_scale_factor
        self.enable_model_offload = enable_model_offload
        
        # Set up VAE slicing if needed
        self.vae_encoder = None
        self.vae_decoder = None
        self.vae_slice_size = getattr(self, 'vae_slice_size', 1)
        
        # Simplified offloading - just store the flag
        # Complex offloading is handled by specialized E2E model classes
        self.offloader = None
        self.modules_dict = None
        
        # Set training mode for components
        if not enable_vae_training:
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False
        
        if not enable_text_encoder_training:
            self.text_encoder_l.eval()
            self.text_encoder_g.eval()
            for param in self.text_encoder_l.parameters():
                param.requires_grad = False
            for param in self.text_encoder_g.parameters():
                param.requires_grad = False
    
    def encode_prompt(
        self,
        input_ids_l: torch.Tensor,
        input_ids_g: torch.Tensor,
        attention_mask_l: Optional[torch.Tensor] = None,
        attention_mask_g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompt using both CLIP models."""
        # Text encoders should already be on correct device
        
        # CLIP-L encoding
        text_outputs_l = self.text_encoder_l(
            input_ids=input_ids_l,
            attention_mask=attention_mask_l,
            output_hidden_states=True,
        )
        text_embeds_l = text_outputs_l.hidden_states[-2]  # Penultimate layer
        pooled_embeds_l = text_outputs_l.pooler_output
        
        # CLIP-G encoding
        text_outputs_g = self.text_encoder_g(
            input_ids=input_ids_g,
            attention_mask=attention_mask_g,
            output_hidden_states=True,
        )
        text_embeds_g = text_outputs_g.hidden_states[-2]  # Penultimate layer
        pooled_embeds_g = text_outputs_g.text_embeds  # For CLIP-G with projection
        
        # Concatenate pooled embeddings
        pooled_embeds = torch.cat([pooled_embeds_l, pooled_embeds_g], dim=-1)
        
        # For now, we'll use CLIP-L embeddings as the main text embeddings
        # In practice, you might want to concatenate or process them differently
        text_embeds = text_embeds_l
        
        return text_embeds, pooled_embeds
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Single training step."""
        import time
        step_start = time.time()
        logger.info("Starting training step...")
        
        # Move batch to device
        images = batch["image"].to(device)
        input_ids_l = batch["input_ids_l"].to(device)
        input_ids_g = batch["input_ids_g"].to(device)
        attention_mask_l = batch.get("attention_mask_l", None)
        attention_mask_g = batch.get("attention_mask_g", None)
        
        if attention_mask_l is not None:
            attention_mask_l = attention_mask_l.to(device)
        if attention_mask_g is not None:
            attention_mask_g = attention_mask_g.to(device)
            
        # Ensure images are in the correct dtype
        if hasattr(self.vae, 'dtype'):
            vae_dtype = next(self.vae.parameters()).dtype
            images = images.to(dtype=vae_dtype)
        
        batch_size = images.shape[0]
        logger.info(f"Batch prepared in {time.time() - step_start:.2f}s")
        
        # Encode images to latents
        vae_start = time.time()
            
        if self.enable_vae_training:
            latents = self.vae.encode(images).latent_dist.sample()
        else:
            with torch.no_grad():
                latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae_scale_factor
        
        # Latents should already be on correct device
        logger.info(f"VAE encoding completed in {time.time() - vae_start:.2f}s")
        
        # Sample noise
        noise = torch.randn_like(latents)
        
        # Sample timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (batch_size,), device=device
        ).long()
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Encode text
        text_start = time.time()
            
        text_embeds, pooled_embeds = self.encode_prompt(
            input_ids_l, input_ids_g, attention_mask_l, attention_mask_g
        )
        
        # Text embeddings should already be on correct device
        logger.info(f"Text encoding completed in {time.time() - text_start:.2f}s")
        
        # Predict noise
        sit_start = time.time()
            
        noise_pred = self.sit_model(
            noisy_latents,
            timesteps,
            text_embeds,
            pooled_embeds,
        )
        
        # Noise prediction should already be on correct device
        logger.info(f"SiT inference completed in {time.time() - sit_start:.2f}s")
        
        # Calculate loss
        if self.noise_scheduler.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {self.noise_scheduler.prediction_type}")
        
        # MSE loss for diffusion
        diffusion_loss = F.mse_loss(noise_pred, target)
        
        # VAE reconstruction loss (if training VAE)
        vae_loss = torch.tensor(0.0, device=device)
        if self.enable_vae_training:
            vae_latents = latents / self.vae_scale_factor
            reconstructed = self.vae.decode(vae_latents).sample
            vae_loss = F.mse_loss(reconstructed, images)
        
        # Total loss
        total_loss = diffusion_loss + 0.01 * vae_loss  # Weight VAE loss lower
        
        # Model offloading handled by specific implementations
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_cached = torch.cuda.memory_reserved(device) / 1024**3
            logger.info(f"Memory: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
        logger.info(f"Total training step completed in {time.time() - step_start:.2f}s")
        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "vae_loss": vae_loss,
        }


def launch_tensorboard(logdir: Path, port: int = 6006):
    """Launch TensorBoard in a subprocess."""
    def run_tensorboard():
        try:
            logger.info(f"Launching TensorBoard on port {port}...")
            cmd = ["tensorboard", "--logdir", str(logdir), "--port", str(port)]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a bit for TensorBoard to start
            time.sleep(3)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info(f"TensorBoard is running at http://localhost:{port}")
                # Try to open browser
                try:
                    webbrowser.open(f"http://localhost:{port}")
                except:
                    pass
            else:
                stdout, stderr = process.communicate()
                logger.error(f"TensorBoard failed to start: {stderr}")
                
        except Exception as e:
            logger.error(f"Failed to launch TensorBoard: {e}")
    
    # Run in a separate thread
    thread = threading.Thread(target=run_tensorboard, daemon=True)
    thread.start()
    
    return thread


def create_models(args):
    """Create and initialize all models."""
    logger.info("Creating models...")
    
    # Create SiT-XL model
    if args.model_size == "512m":
        sit_model = sit_xl_512m()
    elif args.model_size == "1b":
        sit_model = sit_xl_1()
    elif args.model_size == "2b":
        sit_model = sit_xl_2()
    elif args.model_size == "3b":
        sit_model = sit_xl_3()
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    # Load VAE
    logger.info("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.vae_path,
        torch_dtype=torch.float32,  # Always use FP32 for training compatibility
    )
    
    # Load CLIP text encoders
    logger.info("Loading CLIP text encoders...")
    text_encoder_l = CLIPTextModel.from_pretrained(
        args.text_encoder_l_path,
        subfolder=args.text_encoder_l_subfolder if args.text_encoder_l_subfolder else None,
        torch_dtype=torch.float32,  # Always use FP32 for training compatibility
    )
    
    text_encoder_g = CLIPTextModelWithProjection.from_pretrained(
        args.text_encoder_g_path,
        subfolder=args.text_encoder_g_subfolder if args.text_encoder_g_subfolder else None,
        torch_dtype=torch.float32,  # Always use FP32 for training compatibility
    )
    
    # Load tokenizers
    tokenizer_l = AutoTokenizer.from_pretrained(
        args.text_encoder_l_path,
        subfolder="tokenizer" if args.text_encoder_l_subfolder else None,
    )
    tokenizer_g = AutoTokenizer.from_pretrained(
        args.text_encoder_g_path,
        subfolder="tokenizer_2" if args.text_encoder_g_subfolder else None,
    )
    
    return sit_model, vae, text_encoder_l, text_encoder_g, tokenizer_l, tokenizer_g


def train(args):
    """Main training function."""
    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA is not available. Training on CPU will be very slow!")
        logger.info(f"Using device: {device}")
    
    # Create models
    sit_model, vae, text_encoder_l, text_encoder_g, tokenizer_l, tokenizer_g = create_models(args)
    
    # Set models to use less memory
    if args.mixed_precision:
        # Keep models in FP32 for parameters but use FP16 for activations via autocast
        # This prevents gradient scaling issues
        pass
    
    # Create noise scheduler
    noise_scheduler = NoiseScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
    )
    
    # Create E2E model with optimal offloading strategy
    if args.cpu_offload_mode == "simple":
        logger.info("Using SimpleOffloadedE2EModel (Basic offloading)")
        e2e_model = SimpleOffloadedE2EModel(
            sit_model=sit_model,
            vae=vae,
            text_encoder_l=text_encoder_l,
            text_encoder_g=text_encoder_g,
            noise_scheduler=noise_scheduler,
            device=device,
            enable_vae_training=args.train_vae,
            enable_text_encoder_training=args.train_text_encoder,
            mixed_precision=args.mixed_precision,
            debug=args.debug,
        )
    elif args.cpu_offload_mode == "kohya":
        logger.info("Using KohyaOffloadedE2EModel (Block-wise offloading)")
        e2e_model = KohyaOffloadedE2EModel(
            sit_model=sit_model,
            vae=vae,
            text_encoder_l=text_encoder_l,
            text_encoder_g=text_encoder_g,
            noise_scheduler=noise_scheduler,
            device=device,
            enable_vae_training=args.train_vae,
            enable_text_encoder_training=args.train_text_encoder,
            mixed_precision=args.mixed_precision,
            blocks_to_keep_on_gpu=4,  # Keep 4 SiT blocks on GPU for gradient compatibility
            debug=args.debug,
        )
    elif args.cpu_offload_mode in ["sequential", "full"]:
        logger.info("Using SimpleOffloadedE2EModel for sequential/full offloading")
        e2e_model = SimpleOffloadedE2EModel(
            sit_model=sit_model,
            vae=vae,
            text_encoder_l=text_encoder_l,
            text_encoder_g=text_encoder_g,
            noise_scheduler=noise_scheduler,
            device=device,
            enable_vae_training=args.train_vae,
            enable_text_encoder_training=args.train_text_encoder,
            mixed_precision=args.mixed_precision,
            debug=args.debug,
        )
    else:
        e2e_model = E2EDiffusionModel(
            sit_model=sit_model,
            vae=vae,
            text_encoder_l=text_encoder_l,
            text_encoder_g=text_encoder_g,
            noise_scheduler=noise_scheduler,
            enable_vae_training=args.train_vae,
            enable_text_encoder_training=args.train_text_encoder,
            enable_model_offload=args.enable_model_offload,
        )
    
    # Move models to device (only if not using CPU offloading)
    if args.cpu_offload_mode == "none":
        sit_model.to(device)
        vae.to(device)
        text_encoder_l.to(device)
        text_encoder_g.to(device)
    
    # Apply memory optimizations
    if args.memory_optimization_level != "minimal":
        logger.info(f"Applying {args.memory_optimization_level} memory optimizations...")
        optimize_model_memory(sit_model, args.memory_optimization_level)
        
    # Print model sizes
    logger.info(f"Model memory requirements:")
    logger.info(f"  SiT-XL: {calculate_model_memory(sit_model):.2f} GB")
    logger.info(f"  VAE: {calculate_model_memory(vae):.2f} GB")
    logger.info(f"  CLIP-L: {calculate_model_memory(text_encoder_l):.2f} GB")
    logger.info(f"  CLIP-G: {calculate_model_memory(text_encoder_g):.2f} GB")
    
    # Enable memory efficient mode if needed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU Memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")
    
    # Create dataloader
    logger.info("Creating dataloader...")
    dataloader = create_dataloader(
        data_root=args.data_root,
        tokenizer_l=tokenizer_l,
        tokenizer_g=tokenizer_g,
        batch_size=args.batch_size,
        resolution=args.resolution,
        num_workers=args.num_workers,
        use_aspect_ratio_bucket=args.use_aspect_ratio_bucket,
    )
    
    # Create optimizer
    logger.info("Creating optimizer...")
    
    # Ensure trainable models are in FP32 for mixed precision training
    if args.mixed_precision:
        logger.info("Converting trainable models to FP32 for mixed precision compatibility")
        sit_model = sit_model.float()
        if args.train_vae:
            vae = vae.float()
        if args.train_text_encoder:
            text_encoder_l = text_encoder_l.float()
            text_encoder_g = text_encoder_g.float()
    
    params_to_optimize = []
    
    # Add SiT model parameters
    params_to_optimize.extend(sit_model.parameters())
    
    # Add VAE parameters if training
    if args.train_vae:
        params_to_optimize.extend(vae.parameters())
    
    # Add text encoder parameters if training
    if args.train_text_encoder:
        params_to_optimize.extend(text_encoder_l.parameters())
        params_to_optimize.extend(text_encoder_g.parameters())
    
    # Create optimizer with optional 8-bit mode
    if args.use_8bit_optimizer:
        logger.info("Using 8-bit optimizer for memory efficiency")
        optimizer = create_optimizer_with_memory_efficient_mode(
            params_to_optimize,
            optimizer_type="adamw",
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_8bit=True,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )
    else:
        optimizer = AdamW(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
            eps=args.adam_epsilon,
        )
    
    # Create learning rate scheduler
    num_training_steps = len(dataloader) * args.num_epochs
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps,
        eta_min=args.learning_rate * 0.1,
    )
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler('cuda') if args.mixed_precision else None
    
    # Enable xformers if requested
    if args.enable_xformers:
        try:
            import xformers
            vae.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention for VAE")
        except ImportError:
            logger.warning("xformers not available, using standard attention")
    
    # Create EMA model
    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(
            sit_model.parameters(),
            decay=args.ema_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
        )
    
    # Create tensorboard writer with timestamp
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    if hasattr(args, 'run_name') and args.run_name:
        run_name = f"{args.run_name}_{run_name}"
    
    log_dir = args.output_dir / "logs" / run_name
    writer = SummaryWriter(log_dir)
    logger.info(f"TensorBoard logs will be saved to: {log_dir}")
    
    # Log run configuration
    writer.add_text("config/run_name", run_name, 0)
    writer.add_text("config/command", " ".join(sys.argv), 0)
    config_text = json.dumps(vars(args), indent=2, default=str)
    writer.add_text("config/args", config_text, 0)
    
    # Launch TensorBoard if requested
    if args.tensorboard:
        launch_tensorboard(args.output_dir / "logs")  # Point to parent logs directory
    else:
        logger.info(f"To view logs, run: tensorboard --logdir {args.output_dir / 'logs'}")
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    
    for epoch in range(args.num_epochs):
        sit_model.train()
        if args.train_vae:
            vae.train()
        if args.train_text_encoder:
            text_encoder_l.train()
            text_encoder_g.train()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Training step
            if args.mixed_precision:
                with autocast('cuda', dtype=torch.float16):
                    losses = e2e_model.training_step(batch, device)
                losses["loss"] = losses["loss"] / args.gradient_accumulation_steps
                
                # Backward pass
                scaler.scale(losses["loss"]).backward()
                
                # Gradient accumulation
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if args.max_grad_norm is not None:
                        try:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                        except ValueError as e:
                            if "FP16 gradients" in str(e):
                                # Skip gradient clipping if we hit FP16 gradient issues
                                logger.warning("Skipping gradient clipping due to FP16 gradient issue")
                            else:
                                raise
                    
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Clear gradients
                    optimizer.zero_grad()
                    
                    # Update learning rate
                    lr_scheduler.step()
            else:
                losses = e2e_model.training_step(batch, device)
                losses["loss"] = losses["loss"] / args.gradient_accumulation_steps
                
                # Backward pass
                losses["loss"].backward()
                
                # Gradient accumulation
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                    
                    # Optimizer step
                    optimizer.step()
                    
                    # Clear gradients
                    optimizer.zero_grad()
                    
                    # Update learning rate
                    lr_scheduler.step()
            
            # Update EMA
            if ema_model is not None:
                ema_model.step(sit_model.parameters())
            
            # Finalize offloading after backward pass
            if hasattr(e2e_model, 'finalize_step'):
                e2e_model.finalize_step()
            
            # Logging
            if global_step % args.logging_steps == 0:
                # Loss metrics
                writer.add_scalar("train/loss", losses["loss"].item(), global_step)
                writer.add_scalar("train/diffusion_loss", losses["diffusion_loss"].item(), global_step)
                writer.add_scalar("train/vae_loss", losses["vae_loss"].item(), global_step)
                
                # Learning rate
                writer.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], global_step)
                
                # GPU memory usage
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
                    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    writer.add_scalar("system/gpu_memory_used_gb", gpu_memory_used, global_step)
                    writer.add_scalar("system/gpu_memory_reserved_gb", gpu_memory_reserved, global_step)
                
                # Gradient norm (if available)
                if args.max_grad_norm is not None:
                    total_norm = 0
                    for p in params_to_optimize:
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    writer.add_scalar("train/gradient_norm", total_norm, global_step)
                
                # Training progress
                writer.add_scalar("train/epoch", epoch, global_step)
                writer.add_scalar("train/step", global_step, global_step)
                
                # Model statistics (parameter norms)
                sit_norm = sum(p.norm().item() ** 2 for p in sit_model.parameters()) ** 0.5
                writer.add_scalar("model/sit_param_norm", sit_norm, global_step)
                
                if args.train_vae:
                    vae_norm = sum(p.norm().item() ** 2 for p in vae.parameters()) ** 0.5
                    writer.add_scalar("model/vae_param_norm", vae_norm, global_step)
                
                if args.train_text_encoder:
                    text_l_norm = sum(p.norm().item() ** 2 for p in text_encoder_l.parameters()) ** 0.5
                    text_g_norm = sum(p.norm().item() ** 2 for p in text_encoder_g.parameters()) ** 0.5
                    writer.add_scalar("model/text_encoder_l_param_norm", text_l_norm, global_step)
                    writer.add_scalar("model/text_encoder_g_param_norm", text_g_norm, global_step)
                
                progress_bar.set_postfix({
                    "loss": f"{losses['loss'].item():.4f}",
                    "diff_loss": f"{losses['diffusion_loss'].item():.4f}",
                    "vae_loss": f"{losses['vae_loss'].item():.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}",
                    "gpu_mem": f"{gpu_memory_used:.1f}GB" if torch.cuda.is_available() else "N/A",
                })
            
            # Save checkpoint
            if global_step % args.save_steps == 0 and global_step > 0:
                logger.info(f"Saving checkpoint at step {global_step}...")
                checkpoint_dir = args.output_dir / f"checkpoint-{global_step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Save models
                torch.save(sit_model.state_dict(), checkpoint_dir / "sit_model.pt")
                if args.train_vae:
                    torch.save(vae.state_dict(), checkpoint_dir / "vae.pt")
                if args.train_text_encoder:
                    torch.save(text_encoder_l.state_dict(), checkpoint_dir / "text_encoder_l.pt")
                    torch.save(text_encoder_g.state_dict(), checkpoint_dir / "text_encoder_g.pt")
                
                # Save optimizer and scheduler
                torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
                torch.save(lr_scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
                
                # Save EMA
                if ema_model is not None:
                    ema_state_dict = {}
                    ema_model.copy_to(ema_state_dict)
                    torch.save(ema_state_dict, checkpoint_dir / "ema_model.pt")
                
                # Save training state
                # Convert Path objects to strings for JSON serialization
                args_dict = vars(args).copy()
                for key, value in args_dict.items():
                    if isinstance(value, Path):
                        args_dict[key] = str(value)
                
                state = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "args": args_dict,
                }
                with open(checkpoint_dir / "state.json", "w") as f:
                    json.dump(state, f, indent=2)
            
            global_step += 1
        
        # End of epoch summary
        writer.add_scalar("epoch/completed", epoch + 1, global_step)
        logger.info(f"Epoch {epoch + 1} completed. Global step: {global_step}")
    
    # Save final model
    logger.info("Saving final model...")
    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(sit_model.state_dict(), final_dir / "sit_model.pt")
    if args.train_vae:
        torch.save(vae.state_dict(), final_dir / "vae.pt")
    if args.train_text_encoder:
        torch.save(text_encoder_l.state_dict(), final_dir / "text_encoder_l.pt")
        torch.save(text_encoder_g.state_dict(), final_dir / "text_encoder_g.pt")
    
    if ema_model is not None:
        ema_state_dict = {}
        ema_model.copy_to(ema_state_dict)
        torch.save(ema_state_dict, final_dir / "ema_model.pt")
    
    writer.close()
    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train E2E SiT-XL diffusion model")
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default="2b", choices=["512m", "1b", "2b", "3b"],
                        help="Model size variant")
    parser.add_argument("--vae_path", type=str, default="madebyollin/sdxl-vae-fp16-fix",
                        help="Path to VAE model")
    parser.add_argument("--text_encoder_l_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Path to CLIP-L text encoder")
    parser.add_argument("--text_encoder_l_subfolder", type=str, default="text_encoder",
                        help="Subfolder for CLIP-L text encoder")
    parser.add_argument("--text_encoder_g_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Path to CLIP-G text encoder")
    parser.add_argument("--text_encoder_g_subfolder", type=str, default="text_encoder_2",
                        help="Subfolder for CLIP-G text encoder")
    
    # Training arguments
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of training data")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="Beta1 for AdamW")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="Beta2 for AdamW")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon for AdamW")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    
    # Data arguments
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Base resolution for training")
    parser.add_argument("--use_aspect_ratio_bucket", action="store_true",
                        help="Use aspect ratio bucketing")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    
    # Diffusion arguments
    parser.add_argument("--num_train_timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--beta_start", type=float, default=0.00085,
                        help="Starting beta for noise schedule")
    parser.add_argument("--beta_end", type=float, default=0.012,
                        help="Ending beta for noise schedule")
    parser.add_argument("--beta_schedule", type=str, default="scaled_linear",
                        choices=["linear", "scaled_linear"],
                        help="Beta schedule type")
    parser.add_argument("--prediction_type", type=str, default="epsilon",
                        choices=["epsilon", "v_prediction"],
                        help="Prediction type for diffusion")
    
    # E2E training arguments
    parser.add_argument("--train_vae", action="store_true",
                        help="Train VAE along with other components")
    parser.add_argument("--train_text_encoder", action="store_true",
                        help="Train text encoders along with other components")
    
    # EMA arguments
    parser.add_argument("--use_ema", action="store_true",
                        help="Use exponential moving average for model weights")
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                        help="EMA decay rate")
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0,
                        help="EMA inverse gamma")
    parser.add_argument("--ema_power", type=float, default=0.75,
                        help="EMA power")
    
    # Memory optimization arguments
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--enable_xformers", action="store_true",
                        help="Enable xformers memory efficient attention")
    parser.add_argument("--enable_model_offload", action="store_true",
                        help="Enable sequential model offloading to CPU")
    parser.add_argument("--cpu_offload_mode", type=str, default="none",
                        choices=["none", "simple", "sequential", "kohya", "full"],
                        help="CPU offloading strategy: none, simple (basic), sequential, kohya (block-wise), or full")
    parser.add_argument("--memory_optimization_level", type=str, default="balanced",
                        choices=["minimal", "balanced", "aggressive"],
                        help="Memory optimization level")
    parser.add_argument("--use_8bit_optimizer", action="store_true",
                        help="Use 8-bit optimizer to reduce memory usage")
    parser.add_argument("--vae_slice_size", type=int, default=1,
                        help="VAE slice size for memory-efficient processing (1=disabled, 4=16 slices)")
    
    # Other arguments
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Launch TensorBoard automatically")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this training run (used in TensorBoard logs)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging for training")
    
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_level = logging.DEBUG if getattr(args, 'debug', False) else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    if getattr(args, 'debug', False):
        logger.debug("Debug mode enabled - detailed logging will be shown")
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Start training
    train(args)


if __name__ == "__main__":
    main()
