"""
Simple but reliable offloading implementation.
Manually controls GPU placement at each step.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import logging
import time
from torch.amp import autocast

logger = logging.getLogger(__name__)


class SimpleOffloadedE2EModel:
    """
    Simple E2E model with manual offloading control.
    Moves each model to GPU only when needed, then back to CPU.
    """
    
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
        mixed_precision: bool = False,
<<<<<<< HEAD
=======
        debug: bool = False,
>>>>>>> fd5c51c (fixes)
    ):
        self.device = device
        self.cpu_device = torch.device("cpu")
        self.noise_scheduler = noise_scheduler
        self.vae_scale_factor = vae_scale_factor
        self.enable_vae_training = enable_vae_training
        self.enable_text_encoder_training = enable_text_encoder_training
        self.mixed_precision = mixed_precision
<<<<<<< HEAD
=======
        self.debug = debug
>>>>>>> fd5c51c (fixes)
        
        # Store models
        self.sit_model = sit_model
        self.vae = vae
        self.text_encoder_l = text_encoder_l
        self.text_encoder_g = text_encoder_g
        
        # Move all to CPU initially
        self.sit_model.to(self.cpu_device)
        self.vae.to(self.cpu_device)
        self.text_encoder_l.to(self.cpu_device)
        self.text_encoder_g.to(self.cpu_device)
        
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
                
<<<<<<< HEAD
        logger.info("Initialized SimpleOffloadedE2EModel with all models on CPU")
=======
        if self.debug:
            logger.info("Initialized SimpleOffloadedE2EModel with all models on CPU")
>>>>>>> fd5c51c (fixes)
    
    def _ensure_on_gpu(self, model: nn.Module, name: str):
        """Ensure model is on GPU."""
        current_device = next(model.parameters()).device
        if current_device != self.device:
<<<<<<< HEAD
            logger.info(f"Moving {name} from {current_device} to {self.device}")
=======
            if self.debug:
                logger.info(f"Moving {name} from {current_device} to {self.device}")
>>>>>>> fd5c51c (fixes)
            model.to(self.device)
            torch.cuda.empty_cache()
    
    def _ensure_on_cpu(self, model: nn.Module, name: str):
        """Ensure model is on CPU."""
        current_device = next(model.parameters()).device
        if current_device != self.cpu_device:
<<<<<<< HEAD
            logger.info(f"Moving {name} from {current_device} to {self.cpu_device}")
=======
            if self.debug:
                logger.info(f"Moving {name} from {current_device} to {self.cpu_device}")
>>>>>>> fd5c51c (fixes)
            model.to(self.cpu_device)
            torch.cuda.empty_cache()
    
    def training_step(self, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        """Training step with manual offloading."""
        start_time = time.time()
        
        # Prepare inputs
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
<<<<<<< HEAD
        logger.info(f"Batch prepared, GPU memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
=======
        if self.debug:
            logger.info(f"Batch prepared, GPU memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
>>>>>>> fd5c51c (fixes)
        
        # Step 1: VAE encoding
        step_start = time.time()
        self._ensure_on_gpu(self.vae, "VAE")
<<<<<<< HEAD
        logger.info(f"VAE on GPU, memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
=======
        if self.debug:
            logger.info(f"VAE on GPU, memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
>>>>>>> fd5c51c (fixes)
        
        # Keep images in their original dtype for mixed precision
        images_vae = images
        
        # Store VAE posterior for KL loss calculation
        vae_posterior = None
        
        # Use autocast for VAE encoding if mixed precision is enabled
        if self.mixed_precision:
            with autocast('cuda', dtype=torch.float16):
                if self.enable_vae_training:
                    vae_posterior = self.vae.encode(images_vae).latent_dist
                    latents = vae_posterior.sample()
                else:
                    with torch.no_grad():
                        latents = self.vae.encode(images_vae).latent_dist.sample()
        else:
            if self.enable_vae_training:
                vae_posterior = self.vae.encode(images_vae).latent_dist
                latents = vae_posterior.sample()
            else:
                with torch.no_grad():
                    latents = self.vae.encode(images_vae).latent_dist.sample()
        
        latents = latents * self.vae_scale_factor
        
        # Detach from computation graph if VAE not being trained
        if not self.enable_vae_training:
            latents = latents.detach()
        
        # Only offload VAE if not training it (to avoid gradient issues)
        if not self.enable_vae_training:
            self._ensure_on_cpu(self.vae, "VAE")
        
<<<<<<< HEAD
        logger.info(f"VAE encoding done in {time.time() - step_start:.2f}s, memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
=======
        if self.debug:
            logger.info(f"VAE encoding done in {time.time() - step_start:.2f}s, memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
>>>>>>> fd5c51c (fixes)
        
        # Step 2: Text encoding
        step_start = time.time()
        self._ensure_on_gpu(self.text_encoder_l, "CLIP-L")
        self._ensure_on_gpu(self.text_encoder_g, "CLIP-G")
<<<<<<< HEAD
        logger.info(f"Text encoders on GPU, memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
=======
        if self.debug:
            logger.info(f"Text encoders on GPU, memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
>>>>>>> fd5c51c (fixes)
        
        # Text encoding with autocast if mixed precision is enabled
        if self.mixed_precision:
            with autocast('cuda', dtype=torch.float16):
                # CLIP-L encoding
                if self.enable_text_encoder_training:
                    text_outputs_l = self.text_encoder_l(
                        input_ids=input_ids_l,
                        attention_mask=attention_mask_l,
                        output_hidden_states=True,
                    )
                else:
                    with torch.no_grad():
                        text_outputs_l = self.text_encoder_l(
                            input_ids=input_ids_l,
                            attention_mask=attention_mask_l,
                            output_hidden_states=True,
                        )
                text_embeds_l = text_outputs_l.hidden_states[-2]
                pooled_embeds_l = text_outputs_l.pooler_output
                
                # CLIP-G encoding
                if self.enable_text_encoder_training:
                    text_outputs_g = self.text_encoder_g(
                        input_ids=input_ids_g,
                        attention_mask=attention_mask_g,
                        output_hidden_states=True,
                    )
                else:
                    with torch.no_grad():
                        text_outputs_g = self.text_encoder_g(
                            input_ids=input_ids_g,
                            attention_mask=attention_mask_g,
                            output_hidden_states=True,
                        )
                text_embeds_g = text_outputs_g.hidden_states[-2]
                pooled_embeds_g = text_outputs_g.text_embeds
        else:
            # CLIP-L encoding
            if self.enable_text_encoder_training:
                text_outputs_l = self.text_encoder_l(
                    input_ids=input_ids_l,
                    attention_mask=attention_mask_l,
                    output_hidden_states=True,
                )
            else:
                with torch.no_grad():
                    text_outputs_l = self.text_encoder_l(
                        input_ids=input_ids_l,
                        attention_mask=attention_mask_l,
                        output_hidden_states=True,
                    )
            text_embeds_l = text_outputs_l.hidden_states[-2]
            pooled_embeds_l = text_outputs_l.pooler_output
            
            # CLIP-G encoding
            if self.enable_text_encoder_training:
                text_outputs_g = self.text_encoder_g(
                    input_ids=input_ids_g,
                    attention_mask=attention_mask_g,
                    output_hidden_states=True,
                )
            else:
                with torch.no_grad():
                    text_outputs_g = self.text_encoder_g(
                        input_ids=input_ids_g,
                        attention_mask=attention_mask_g,
                        output_hidden_states=True,
                    )
            text_embeds_g = text_outputs_g.hidden_states[-2]
            pooled_embeds_g = text_outputs_g.text_embeds
        
        # Combine embeddings
        pooled_embeds = torch.cat([pooled_embeds_l, pooled_embeds_g], dim=-1)
        text_embeds = text_embeds_l
        
        # Detach from computation graph if text encoders not being trained
        if not self.enable_text_encoder_training:
            text_embeds = text_embeds.detach()
            pooled_embeds = pooled_embeds.detach()
        
        # Only offload text encoders if not training them (to avoid gradient issues)
        if not self.enable_text_encoder_training:
            self._ensure_on_cpu(self.text_encoder_l, "CLIP-L")
            self._ensure_on_cpu(self.text_encoder_g, "CLIP-G")
        
<<<<<<< HEAD
        logger.info(f"Text encoding done in {time.time() - step_start:.2f}s, memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
=======
        if self.debug:
            logger.info(f"Text encoding done in {time.time() - step_start:.2f}s, memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
>>>>>>> fd5c51c (fixes)
        
        # Step 3: Noise and timesteps
        noise = torch.randn_like(latents, device=self.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()
        
        # Step 4: Add noise
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Step 5: SiT inference
        step_start = time.time()
        self._ensure_on_gpu(self.sit_model, "SiT-XL")
<<<<<<< HEAD
        logger.info(f"SiT model on GPU, memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
=======
        if self.debug:
            logger.info(f"SiT model on GPU, memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
>>>>>>> fd5c51c (fixes)
        
        # Keep inputs in their original dtype for mixed precision
        noisy_latents_sit = noisy_latents
        text_embeds_sit = text_embeds
        pooled_embeds_sit = pooled_embeds
        
        # SiT prediction with autocast if mixed precision is enabled
        if self.mixed_precision:
            with autocast('cuda', dtype=torch.float16):
                noise_pred = self.sit_model(
                    noisy_latents_sit,
                    timesteps,
                    text_embeds_sit,
                    pooled_embeds_sit,
                )
        else:
            noise_pred = self.sit_model(
                noisy_latents_sit,
                timesteps,
                text_embeds_sit,
                pooled_embeds_sit,
            )
        
<<<<<<< HEAD
        logger.info(f"SiT inference done in {time.time() - step_start:.2f}s, memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
=======
        if self.debug:
            logger.info(f"SiT inference done in {time.time() - step_start:.2f}s, memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
>>>>>>> fd5c51c (fixes)
        
        # Step 6: Calculate loss
        if self.noise_scheduler.prediction_type == "epsilon":
            target = noise
        else:
            raise ValueError(f"Unknown prediction type: {self.noise_scheduler.prediction_type}")
        
        # Ensure target is on same device as prediction
        target = target.to(noise_pred.device)
        
        diffusion_loss = torch.nn.functional.mse_loss(noise_pred, target)
        
        # VAE reconstruction loss (if training VAE)
        vae_loss = torch.tensor(0.0, device=self.device)
        if self.enable_vae_training:
            # Decode latents back to images
            self._ensure_on_gpu(self.vae, "VAE")
            vae_dtype = next(self.vae.parameters()).dtype
            
            # Scale latents back to VAE space
            decoded_latents = (latents / self.vae_scale_factor).to(dtype=vae_dtype)
            
            if self.mixed_precision:
                with autocast('cuda', dtype=torch.float16):
                    reconstructed = self.vae.decode(decoded_latents).sample
            else:
                reconstructed = self.vae.decode(decoded_latents).sample
            
            # Calculate reconstruction loss  
            # Ensure images and reconstructed are same type and range
            target_images = images.to(dtype=reconstructed.dtype, device=reconstructed.device)
            
            # Normalize to [-1, 1] range if needed (SDXL VAE expects this)
            if target_images.min() >= 0.0:  # If images are in [0,1] range
                target_images = target_images * 2.0 - 1.0
            
            vae_reconstruction_loss = torch.nn.functional.mse_loss(reconstructed, target_images)
            
            # KL divergence loss
            if vae_posterior is not None:
                # KL divergence between VAE posterior and standard normal
                vae_kl_loss = torch.distributions.kl_divergence(
                    vae_posterior,
                    torch.distributions.Normal(
                        torch.zeros_like(vae_posterior.mean),
                        torch.ones_like(vae_posterior.stddev)
                    )
                ).mean()
            else:
                vae_kl_loss = torch.tensor(0.0, device=self.device)
            
            # LPIPS perceptual loss (simplified - using L1 as proxy)
            # Note: For full REPA-E implementation, you would use actual LPIPS
            lpips_loss = torch.nn.functional.l1_loss(reconstructed, target_images)
            
            # Total VAE loss (following REPA-E formulation)
            vae_loss = (vae_reconstruction_loss + 
                       0.1 * lpips_loss +  # Perceptual loss weight
                       0.0001 * vae_kl_loss)  # KL weight
            
            # Only offload VAE if not training it
            if not self.enable_vae_training:
                self._ensure_on_cpu(self.vae, "VAE")
        
        total_loss = diffusion_loss + 0.01 * vae_loss
        
<<<<<<< HEAD
        logger.info(f"Total training step: {time.time() - start_time:.2f}s, final memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
=======
        if self.debug:
            logger.info(f"Total training step: {time.time() - start_time:.2f}s, final memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
>>>>>>> fd5c51c (fixes)
        
        # NOTE: SiT model stays on GPU for backward pass
        result = {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "vae_loss": vae_loss,
        }
        
        # Add detailed VAE loss components if training VAE
        if self.enable_vae_training and vae_loss.item() > 0:
            try:
                result.update({
                    "vae_reconstruction_loss": vae_reconstruction_loss,
                    "vae_kl_loss": vae_kl_loss,
                    "vae_lpips_loss": lpips_loss,
                })
            except NameError:
                # Variables not in scope - this is expected if VAE training is disabled
                pass
        
        return result
    
    def finalize_step(self):
        """Offload all models after backward pass."""
<<<<<<< HEAD
        logger.info("Finalizing step - offloading all models")
=======
        if self.debug:
            logger.info("Finalizing step - offloading all models")
>>>>>>> fd5c51c (fixes)
        self._ensure_on_cpu(self.sit_model, "SiT-XL")
        
        # Only offload models that were left on GPU during training
        if self.enable_vae_training:
            self._ensure_on_cpu(self.vae, "VAE")
        if self.enable_text_encoder_training:
            self._ensure_on_cpu(self.text_encoder_l, "CLIP-L")
            self._ensure_on_cpu(self.text_encoder_g, "CLIP-G")
        
        torch.cuda.empty_cache()
<<<<<<< HEAD
        logger.info(f"All models offloaded, memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
=======
        if self.debug:
            logger.info(f"All models offloaded, memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
>>>>>>> fd5c51c (fixes)
