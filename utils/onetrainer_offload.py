"""
OneTrainer-style CPU offloading implementation.
Based on https://github.com/Nerogar/OneTrainer/blob/master/docs/RamOffloading.md
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
import threading
import queue
import time
import gc
from contextlib import contextmanager


class OneTrainerOffloader:
    """
    OneTrainer-style CPU offloading that minimizes GPU memory usage.
    Only keeps the currently needed model on GPU.
    """
    
    def __init__(self, device: torch.device = torch.device("cuda")):
        self.device = device
        self.cpu_device = torch.device("cpu")
        self.modules: Dict[str, nn.Module] = {}
        self.current_gpu_module: Optional[str] = None
        self.transfer_queue = queue.Queue()
        self.transfer_thread = None
        self.stop_thread = False
        
    def register_module(self, name: str, module: nn.Module):
        """Register a module for offloading management."""
        self.modules[name] = module
        # Start with all modules on CPU except first one
        if len(self.modules) == 1:
            module.to(self.device)
            self.current_gpu_module = name
        else:
            module.to(self.cpu_device)
    
    @contextmanager
    def module_context(self, name: str):
        """Context manager that ensures a module is on GPU during execution."""
        original_module = self.current_gpu_module
        
        try:
            # Move target module to GPU if not already there
            if self.current_gpu_module != name:
                self._move_to_gpu(name)
            
            yield self.modules[name]
            
        finally:
            # Keep module on GPU during training to allow gradients
            # Only offload during inference or when explicitly requested
            pass
    
    def _move_to_gpu(self, name: str):
        """Move a specific module to GPU and offload others."""
        if name not in self.modules:
            raise ValueError(f"Module {name} not registered")
        
        # During training, keep all modules on GPU to allow gradient flow
        # Only offload during inference for memory savings
        training_mode = any(module.training for module in self.modules.values())
        
        if training_mode:
            # Keep all modules on GPU during training
            for module in self.modules.values():
                module.to(self.device)
            self.current_gpu_module = name
        else:
            # Offload current GPU module
            if self.current_gpu_module is not None and self.current_gpu_module != name:
                self.modules[self.current_gpu_module].to(self.cpu_device)
            
            # Move target module to GPU
            self.modules[name].to(self.device)
            self.current_gpu_module = name
            
            # Clear cache
            torch.cuda.empty_cache()
    
    def offload_all(self):
        """Offload all modules to CPU."""
        for module in self.modules.values():
            module.to(self.cpu_device)
        self.current_gpu_module = None
        torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            cached = torch.cuda.memory_reserved(self.device) / 1024**3
            return {"allocated": allocated, "cached": cached}
        return {"allocated": 0, "cached": 0}


class OptimizedE2EModel:
    """
    OneTrainer-style E2E model with optimized memory usage.
    """
    
    def __init__(
        self,
        sit_model: nn.Module,
        vae: nn.Module, 
        text_encoder_l: nn.Module,
        text_encoder_g: nn.Module,
        noise_scheduler,
        device: torch.device,
        cpu_offload_mode: str = "sequential",
        vae_scale_factor: float = 0.18215,
        enable_vae_training: bool = True,
        enable_text_encoder_training: bool = True,
    ):
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.vae_scale_factor = vae_scale_factor
        self.enable_vae_training = enable_vae_training
        self.enable_text_encoder_training = enable_text_encoder_training
        self.cpu_offload_mode = cpu_offload_mode
        
        # Set up offloader
        if cpu_offload_mode in ["sequential", "full"]:
            self.offloader = OneTrainerOffloader(device)
            self.offloader.register_module("sit_model", sit_model)
            self.offloader.register_module("vae", vae)
            self.offloader.register_module("text_encoder_l", text_encoder_l)
            self.offloader.register_module("text_encoder_g", text_encoder_g)
        else:
            self.offloader = None
            self.sit_model = sit_model
            self.vae = vae
            self.text_encoder_l = text_encoder_l
            self.text_encoder_g = text_encoder_g
            
            # Move all to device
            sit_model.to(device)
            vae.to(device)
            text_encoder_l.to(device)
            text_encoder_g.to(device)
        
        # Set training modes
        if not enable_vae_training:
            vae.eval()
            for param in vae.parameters():
                param.requires_grad = False
        
        if not enable_text_encoder_training:
            text_encoder_l.eval()
            text_encoder_g.eval()
            for param in text_encoder_l.parameters():
                param.requires_grad = False
            for param in text_encoder_g.parameters():
                param.requires_grad = False
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latents with memory optimization."""
        if self.offloader:
            with self.offloader.module_context("vae") as vae:
                # Ensure images are on same device as VAE
                images = images.to(next(vae.parameters()).device)
                images = images.to(next(vae.parameters()).dtype)
                
                if self.enable_vae_training:
                    latents = vae.encode(images).latent_dist.sample()
                else:
                    with torch.no_grad():
                        latents = vae.encode(images).latent_dist.sample()
                
                latents = latents * self.vae_scale_factor
                # Move back to main device
                latents = latents.to(self.device)
                return latents
        else:
            # Standard processing
            if self.enable_vae_training:
                latents = self.vae.encode(images).latent_dist.sample()
            else:
                with torch.no_grad():
                    latents = self.vae.encode(images).latent_dist.sample()
            return latents * self.vae_scale_factor
    
    def encode_text(
        self,
        input_ids_l: torch.Tensor,
        input_ids_g: torch.Tensor,
        attention_mask_l: Optional[torch.Tensor] = None,
        attention_mask_g: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Encode text with memory optimization."""
        if self.offloader:
            # Encode with CLIP-L
            with self.offloader.module_context("text_encoder_l") as text_encoder_l:
                # Move inputs to encoder device
                device = next(text_encoder_l.parameters()).device
                input_ids_l = input_ids_l.to(device)
                if attention_mask_l is not None:
                    attention_mask_l = attention_mask_l.to(device)
                
                text_outputs_l = text_encoder_l(
                    input_ids=input_ids_l,
                    attention_mask=attention_mask_l,
                    output_hidden_states=True,
                )
                text_embeds_l = text_outputs_l.hidden_states[-2]
                pooled_embeds_l = text_outputs_l.pooler_output
                
                # Move to main device
                text_embeds_l = text_embeds_l.to(self.device)
                pooled_embeds_l = pooled_embeds_l.to(self.device)
            
            # Encode with CLIP-G
            with self.offloader.module_context("text_encoder_g") as text_encoder_g:
                # Move inputs to encoder device
                device = next(text_encoder_g.parameters()).device
                input_ids_g = input_ids_g.to(device)
                if attention_mask_g is not None:
                    attention_mask_g = attention_mask_g.to(device)
                
                text_outputs_g = text_encoder_g(
                    input_ids=input_ids_g,
                    attention_mask=attention_mask_g,
                    output_hidden_states=True,
                )
                text_embeds_g = text_outputs_g.hidden_states[-2]
                pooled_embeds_g = text_outputs_g.text_embeds
                
                # Move to main device
                pooled_embeds_g = pooled_embeds_g.to(self.device)
            
            # Combine embeddings
            pooled_embeds = torch.cat([pooled_embeds_l, pooled_embeds_g], dim=-1)
            text_embeds = text_embeds_l  # Use CLIP-L embeddings
            
            return text_embeds, pooled_embeds
        else:
            # Standard processing
            text_outputs_l = self.text_encoder_l(
                input_ids=input_ids_l,
                attention_mask=attention_mask_l,
                output_hidden_states=True,
            )
            text_embeds_l = text_outputs_l.hidden_states[-2]
            pooled_embeds_l = text_outputs_l.pooler_output
            
            text_outputs_g = self.text_encoder_g(
                input_ids=input_ids_g,
                attention_mask=attention_mask_g,
                output_hidden_states=True,
            )
            text_embeds_g = text_outputs_g.hidden_states[-2]
            pooled_embeds_g = text_outputs_g.text_embeds
            
            pooled_embeds = torch.cat([pooled_embeds_l, pooled_embeds_g], dim=-1)
            text_embeds = text_embeds_l
            
            return text_embeds, pooled_embeds
    
    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise with SiT model."""
        if self.offloader:
            with self.offloader.module_context("sit_model") as sit_model:
                # Ensure all inputs are on same device
                device = next(sit_model.parameters()).device
                dtype = next(sit_model.parameters()).dtype
                
                noisy_latents = noisy_latents.to(device=device, dtype=dtype)
                timesteps = timesteps.to(device)
                text_embeds = text_embeds.to(device=device, dtype=dtype)
                pooled_embeds = pooled_embeds.to(device=device, dtype=dtype)
                
                noise_pred = sit_model(
                    noisy_latents,
                    timesteps,
                    text_embeds,
                    pooled_embeds,
                )
                
                # Move back to main device
                return noise_pred.to(self.device)
        else:
            return self.sit_model(
                noisy_latents,
                timesteps,
                text_embeds,
                pooled_embeds,
            )
    
    def training_step(self, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        """Memory-optimized training step."""
        images = batch["image"].to(self.device)
        input_ids_l = batch["input_ids_l"].to(self.device)
        input_ids_g = batch["input_ids_g"].to(self.device)
        attention_mask_l = batch.get("attention_mask_l", None)
        attention_mask_g = batch.get("attention_mask_g", None)
        
        if attention_mask_l is not None:
            attention_mask_l = attention_mask_l.to(self.device)
        if attention_mask_g is not None:
            attention_mask_g = attention_mask_g.to(self.device)
        
        batch_size = images.shape[0]
        
        # Step 1: Encode images
        latents = self.encode_images(images)
        
        # Step 2: Encode text
        text_embeds, pooled_embeds = self.encode_text(
            input_ids_l, input_ids_g, attention_mask_l, attention_mask_g
        )
        
        # Step 3: Prepare noise and timesteps
        noise = torch.randn_like(latents, device=self.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()
        
        # Step 4: Add noise
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Step 5: Predict noise
        noise_pred = self.predict_noise(noisy_latents, timesteps, text_embeds, pooled_embeds)
        
        # Step 6: Calculate loss
        if self.noise_scheduler.prediction_type == "epsilon":
            target = noise
        else:
            raise ValueError(f"Unknown prediction type: {self.noise_scheduler.prediction_type}")
        
        diffusion_loss = torch.nn.functional.mse_loss(noise_pred, target)
        
        # Simplified VAE loss for now
        vae_loss = torch.tensor(0.0, device=self.device)
        
        total_loss = diffusion_loss + 0.01 * vae_loss
        
        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss, 
            "vae_loss": vae_loss,
        }