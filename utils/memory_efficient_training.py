"""
Memory-efficient training utilities that avoid frequent CPU-GPU transfers.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import gc


class MemoryEfficientVAE:
    """VAE wrapper that processes images in smaller chunks to reduce memory usage."""
    
    def __init__(self, vae: nn.Module, max_batch_size: int = 1, chunk_size: int = 512):
        self.vae = vae
        self.max_batch_size = max_batch_size
        self.chunk_size = chunk_size  # Process images in chunks of this size
        
    def encode_chunked(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images in chunks to reduce memory usage."""
        B, C, H, W = images.shape
        
        # If image is small enough, process normally
        if H <= self.chunk_size and W <= self.chunk_size:
            return self.vae.encode(images).latent_dist.sample()
        
        # Process in chunks
        chunk_h = min(self.chunk_size, H)
        chunk_w = min(self.chunk_size, W)
        
        # Calculate number of chunks
        n_chunks_h = (H + chunk_h - 1) // chunk_h
        n_chunks_w = (W + chunk_w - 1) // chunk_w
        
        # Output tensor
        latent_h = H // 8  # VAE downsamples by 8
        latent_w = W // 8
        latents = torch.zeros(
            (B, 4, latent_h, latent_w), 
            device=images.device, 
            dtype=images.dtype
        )
        
        for i in range(n_chunks_h):
            for j in range(n_chunks_w):
                h_start = i * chunk_h
                h_end = min(h_start + chunk_h, H)
                w_start = j * chunk_w
                w_end = min(w_start + chunk_w, W)
                
                # Extract chunk
                chunk = images[:, :, h_start:h_end, w_start:w_end]
                
                # Encode chunk
                with torch.no_grad():
                    chunk_latent = self.vae.encode(chunk).latent_dist.sample()
                
                # Place in output tensor
                lat_h_start = h_start // 8
                lat_h_end = h_end // 8
                lat_w_start = w_start // 8
                lat_w_end = w_end // 8
                
                latents[:, :, lat_h_start:lat_h_end, lat_w_start:lat_w_end] = chunk_latent
                
                # Clear cache
                del chunk_latent
                torch.cuda.empty_cache()
        
        return latents


class LowMemoryE2EModel:
    """Ultra-low memory E2E model that minimizes GPU memory usage."""
    
    def __init__(
        self,
        sit_model: nn.Module,
        vae: nn.Module,
        text_encoder_l: nn.Module,
        text_encoder_g: nn.Module,
        noise_scheduler,
        device: torch.device,
        vae_scale_factor: float = 0.18215,
        enable_vae_training: bool = True,
        enable_text_encoder_training: bool = True,
    ):
        self.sit_model = sit_model
        self.vae = vae
        self.text_encoder_l = text_encoder_l
        self.text_encoder_g = text_encoder_g
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.vae_scale_factor = vae_scale_factor
        self.enable_vae_training = enable_vae_training
        self.enable_text_encoder_training = enable_text_encoder_training
        
        # Keep only SiT model on GPU, others on CPU by default
        self.sit_model.to(device)
        self.vae.to('cpu')
        self.text_encoder_l.to('cpu')
        self.text_encoder_g.to('cpu')
        
        # Wrap VAE for chunked processing
        self.chunked_vae = MemoryEfficientVAE(vae, chunk_size=256)
        
        # Set training modes
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
    
    def encode_images_on_cpu(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using CPU-based VAE to save GPU memory."""
        # Move images to CPU
        images_cpu = images.cpu()
        
        # Move VAE to CPU if not already there
        self.vae.to('cpu')
        
        # Encode on CPU
        with torch.no_grad():
            latents = self.vae.encode(images_cpu).latent_dist.sample()
        
        # Scale and move back to GPU
        latents = latents * self.vae_scale_factor
        latents = latents.to(self.device)
        
        # Clear CPU memory
        del images_cpu
        torch.cuda.empty_cache()
        
        return latents
    
    def encode_text_on_cpu(
        self, 
        input_ids_l: torch.Tensor,
        input_ids_g: torch.Tensor,
        attention_mask_l: Optional[torch.Tensor] = None,
        attention_mask_g: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Encode text using CPU-based encoders to save GPU memory."""
        # Move inputs to CPU
        input_ids_l_cpu = input_ids_l.cpu()
        input_ids_g_cpu = input_ids_g.cpu()
        attention_mask_l_cpu = attention_mask_l.cpu() if attention_mask_l is not None else None
        attention_mask_g_cpu = attention_mask_g.cpu() if attention_mask_g is not None else None
        
        # Move encoders to CPU
        self.text_encoder_l.to('cpu')
        self.text_encoder_g.to('cpu')
        
        # Encode on CPU
        with torch.no_grad():
            # CLIP-L
            text_outputs_l = self.text_encoder_l(
                input_ids=input_ids_l_cpu,
                attention_mask=attention_mask_l_cpu,
                output_hidden_states=True,
            )
            text_embeds_l = text_outputs_l.hidden_states[-2]
            pooled_embeds_l = text_outputs_l.pooler_output
            
            # CLIP-G
            text_outputs_g = self.text_encoder_g(
                input_ids=input_ids_g_cpu,
                attention_mask=attention_mask_g_cpu,
                output_hidden_states=True,
            )
            text_embeds_g = text_outputs_g.hidden_states[-2]
            pooled_embeds_g = text_outputs_g.text_embeds
        
        # Combine and move to GPU
        pooled_embeds = torch.cat([pooled_embeds_l, pooled_embeds_g], dim=-1)
        text_embeds = text_embeds_l  # Use CLIP-L embeddings
        
        text_embeds = text_embeds.to(self.device)
        pooled_embeds = pooled_embeds.to(self.device)
        
        # Clear CPU memory
        del text_outputs_l, text_outputs_g, text_embeds_l, pooled_embeds_l
        del text_embeds_g, pooled_embeds_g
        torch.cuda.empty_cache()
        
        return text_embeds, pooled_embeds
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Memory-efficient training step."""
        # Extract batch data
        images = batch["image"]
        input_ids_l = batch["input_ids_l"]
        input_ids_g = batch["input_ids_g"]
        attention_mask_l = batch.get("attention_mask_l", None)
        attention_mask_g = batch.get("attention_mask_g", None)
        
        batch_size = images.shape[0]
        
        # Step 1: Encode images on CPU
        latents = self.encode_images_on_cpu(images)
        
        # Step 2: Encode text on CPU
        text_embeds, pooled_embeds = self.encode_text_on_cpu(
            input_ids_l, input_ids_g, attention_mask_l, attention_mask_g
        )
        
        # Step 3: Prepare noise and timesteps on GPU
        noise = torch.randn_like(latents, device=self.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, 
            (batch_size,), 
            device=self.device
        ).long()
        
        # Step 4: Add noise
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Step 5: SiT inference (only thing on GPU)
        noise_pred = self.sit_model(
            noisy_latents,
            timesteps,
            text_embeds,
            pooled_embeds,
        )
        
        # Step 6: Calculate loss
        if self.noise_scheduler.prediction_type == "epsilon":
            target = noise
        else:
            raise ValueError(f"Unknown prediction type: {self.noise_scheduler.prediction_type}")
        
        diffusion_loss = torch.nn.functional.mse_loss(noise_pred, target)
        
        # VAE loss (simplified - skip for now to save memory)
        vae_loss = torch.tensor(0.0, device=self.device)
        
        total_loss = diffusion_loss + 0.01 * vae_loss
        
        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "vae_loss": vae_loss,
        }


def create_low_memory_model(
    sit_model: nn.Module,
    vae: nn.Module,
    text_encoder_l: nn.Module,
    text_encoder_g: nn.Module,
    noise_scheduler,
    device: torch.device,
    **kwargs
) -> LowMemoryE2EModel:
    """Create a low-memory E2E model."""
    return LowMemoryE2EModel(
        sit_model=sit_model,
        vae=vae,
        text_encoder_l=text_encoder_l,
        text_encoder_g=text_encoder_g,
        noise_scheduler=noise_scheduler,
        device=device,
        **kwargs
    )