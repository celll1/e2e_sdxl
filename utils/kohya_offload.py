"""
Kohya-style offloading implementation for E2E model training.
Based on kohya-ss/sd-scripts memory optimization techniques.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging
import time
from torch.amp import autocast

from .model_offload import KohyaSequentialOffloader

logger = logging.getLogger(__name__)


class KohyaOffloadedE2EModel:
    """
    E2E model with Kohya-style offloading that supports block-wise model management.
    Implements layer-wise offloading for SiT transformer blocks similar to kohya-ss approach.
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
        blocks_to_keep_on_gpu: int = 2,
        debug: bool = False,
    ):
        self.device = device
        self.cpu_device = torch.device("cpu")
        self.noise_scheduler = noise_scheduler
        self.vae_scale_factor = vae_scale_factor
        self.enable_vae_training = enable_vae_training
        self.enable_text_encoder_training = enable_text_encoder_training
        self.mixed_precision = mixed_precision
        self.blocks_to_keep_on_gpu = blocks_to_keep_on_gpu
        self.debug = debug
        
        # Store models
        self.sit_model = sit_model
        self.vae = vae
        self.text_encoder_l = text_encoder_l
        self.text_encoder_g = text_encoder_g
        
        # Initialize Kohya offloader
        self.offloader = KohyaSequentialOffloader(device=device, debug=debug)
        
        # Initial offloading setup
        self._setup_initial_offloading()
        
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
                
        if self.debug:
            logger.info("Initialized KohyaOffloadedE2EModel with block-wise offloading")
    
    def _setup_initial_offloading(self):
        """Setup initial offloading configuration - move all models to CPU."""
        # Move all models to CPU initially  
        self._ensure_model_on_device(self.sit_model, "SiT-XL", self.cpu_device)
        self._ensure_model_on_device(self.vae, "VAE", self.cpu_device)
        self._ensure_model_on_device(self.text_encoder_l, "CLIP-L", self.cpu_device)
        self._ensure_model_on_device(self.text_encoder_g, "CLIP-G", self.cpu_device)
        
        # Ensure SiT model's top-level parameters are also on CPU
        for name, param in self.sit_model.named_parameters():
            if not name.startswith('blocks.') and param.device != self.cpu_device:
                param.data = param.data.to(self.cpu_device)
        
        for name, buffer in self.sit_model.named_buffers():
            if not name.startswith('blocks.') and buffer.device != self.cpu_device:
                buffer.data = buffer.data.to(self.cpu_device)
        
        if self.debug:
            logger.info("All models and parameters moved to CPU for progressive offloading")
    
    def _ensure_model_on_gpu(self, model: nn.Module, model_name: str):
        """Ensure model is on GPU for computation."""
        current_device = next(model.parameters()).device
        if current_device != self.device:
            if self.debug:
                logger.info(f"Loading {model_name} to GPU")
            model.to(self.device)
            torch.cuda.empty_cache()
    
    def _offload_model_after_use(self, model: nn.Module, model_name: str, keep_on_gpu: bool = False):
        """Offload model after use if not needed for gradients."""
        if not keep_on_gpu:
            if self.debug:
                logger.info(f"Offloading {model_name} to CPU")
            model.to(self.cpu_device)
            torch.cuda.empty_cache()
    
    
    def _ensure_model_on_device(self, model, model_name, target_device):
        """Safely move entire model to target device."""
        try:
            current_device = next(model.parameters()).device
            if current_device != target_device:
                if self.debug:
                    logger.info(f"Moving {model_name} from {current_device} to {target_device}")
                model.to(target_device)
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Failed to move {model_name} to {target_device}: {e}")
            raise
    
    def _debug_model_devices(self):
        """Debug helper to check device placement of all models."""
        try:
            sit_device = next(self.sit_model.parameters()).device
            vae_device = next(self.vae.parameters()).device
            clip_l_device = next(self.text_encoder_l.parameters()).device
            clip_g_device = next(self.text_encoder_g.parameters()).device
            
            logger.info(f"Model devices - SiT: {sit_device}, VAE: {vae_device}, CLIP-L: {clip_l_device}, CLIP-G: {clip_g_device}")
            
            if self.enable_text_encoder_training:
                if clip_l_device != self.device or clip_g_device != self.device:
                    logger.warning("Text encoders not on GPU but training enabled!")
            
            if self.enable_vae_training:
                if vae_device != self.device:
                    logger.warning("VAE not on GPU but training enabled!")
                    
            if sit_device != self.device:
                logger.warning("SiT model not on GPU!")
                
        except Exception as e:
            logger.error(f"Failed to debug model devices: {e}")

    def _forward_with_progressive_block_offloading(self, noisy_latents, timesteps, text_embeds, pooled_embeds):
        """
        Progressive block-wise offloading during forward pass.
        Based on kohya-ss approach for memory-efficient transformer inference.
        """
        if not hasattr(self.sit_model, 'blocks'):
            # Fallback to full model loading if no blocks
            self._ensure_model_on_device(self.sit_model, "SiT-XL", self.device)
            return self.sit_model(noisy_latents, timesteps, text_embeds, pooled_embeds)
        
        if self.debug:
            logger.info("Starting progressive block-wise offloading")
        
        # Ensure non-block components are on GPU
        for name, module in self.sit_model.named_children():
            if name != 'blocks':
                self._ensure_model_on_device(module, f"SiT-{name}", self.device)
        
        # Manually move top-level parameters and buffers to GPU
        for name, param in self.sit_model.named_parameters():
            if not name.startswith('blocks.') and param.device != self.device:
                param.data = param.data.to(self.device)
                if self.debug:
                    logger.info(f"Moved parameter {name} to GPU")
        
        for name, buffer in self.sit_model.named_buffers():
            if not name.startswith('blocks.') and buffer.device != self.device:
                buffer.data = buffer.data.to(self.device)
                if self.debug:
                    logger.info(f"Moved buffer {name} to GPU")
        
        # Ensure all input tensors are on GPU
        noisy_latents = noisy_latents.to(self.device)
        timesteps = timesteps.to(self.device)
        text_embeds = text_embeds.to(self.device)
        pooled_embeds = pooled_embeds.to(self.device)
        
        # Manual forward pass with block-wise loading
        # Following the exact same flow as the original SiT forward method
        B, C, H, W = noisy_latents.shape
        x = noisy_latents
        
        # Patch embedding (should already be on GPU)
        x = self.sit_model.patch_embed(x)  # (B, dim, H/P, W/P)
        from einops import rearrange
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Add position embedding
        x = x + self.sit_model.pos_embed[:, :x.shape[1], :]
        
        # Time embedding
        time_emb = self.sit_model.time_embed(timesteps)  # (B, dim * 4)
        
        # Text conditioning using pooled embeddings
        if pooled_embeds is not None:
            text_cond = self.sit_model.text_embed(pooled_embeds)  # (B, dim * 4)
            time_emb = time_emb + text_cond
        
        # Progressive transformer blocks
        blocks = self.sit_model.blocks
        total_blocks = len(blocks)
        
        # Store blocks and embeddings for backward pass management
        self._current_blocks = blocks
        self._current_device = self.device
        self._time_emb = time_emb  # Keep time_emb on GPU for gradient checkpointing
        
        for i, block in enumerate(blocks):
            # Load current block to GPU
            self._ensure_model_on_device(block, f"SiT-block-{i}", self.device)
            
            # Register both pre-backward and full-backward hooks for comprehensive device management
            def create_pre_backward_hook(block_idx):
                def pre_backward_hook(module, grad_output):
                    # Ensure this block is on GPU before backward pass
                    current_device = next(module.parameters()).device
                    if current_device != self._current_device:
                        if self.debug:
                            logger.info(f"Pre-loading SiT block {block_idx} to GPU for backward pass")
                        module.to(self._current_device)
                return pre_backward_hook
            
            def create_backward_hook(block_idx):
                def backward_hook(module, grad_input, grad_output):
                    # Double-check that module is still on GPU
                    current_device = next(module.parameters()).device
                    if current_device != self._current_device:
                        if self.debug:
                            logger.info(f"Loading SiT block {block_idx} to GPU for backward pass")
                        module.to(self._current_device)
                    return grad_input
                return backward_hook
            
            # Register both hooks
            pre_hook = block.register_full_backward_pre_hook(create_pre_backward_hook(i))
            post_hook = block.register_full_backward_hook(create_backward_hook(i))
            
            if not hasattr(self, '_backward_hooks'):
                self._backward_hooks = []
            self._backward_hooks.extend([pre_hook, post_hook])
            
            # Forward through block (SiTBlock only takes x and time_emb)
            x = block(x, time_emb)
            
            # Offload block after forward pass (except last few blocks for gradients)
            keep_on_gpu = i >= (total_blocks - self.blocks_to_keep_on_gpu)
            if not keep_on_gpu:
                self._ensure_model_on_device(block, f"SiT-block-{i}", self.cpu_device)
                if self.debug and i < 5:  # Log first few to avoid spam
                    logger.info(f"Offloaded SiT block {i} after forward pass")
        
        # Output projection
        x = self.sit_model.norm_out(x)
        x = self.sit_model.proj_out(x)
        
        # Unpatchify to get back to image format
        x = self.sit_model.unpatchify(x, H, W)
        
        # Handle sigma prediction if enabled
        if hasattr(self.sit_model, 'learn_sigma') and self.sit_model.learn_sigma:
            x, _ = torch.chunk(x, 2, dim=1)
        
        if self.debug:
            memory_after_forward = torch.cuda.memory_allocated(self.device) / 1024**3
            logger.info(f"Memory after progressive forward: {memory_after_forward:.2f} GB")
            blocks_on_gpu = sum(1 for i, block in enumerate(blocks) 
                              if next(block.parameters()).device == self.device)
            logger.info(f"Progressive forward: {blocks_on_gpu}/{total_blocks} blocks remain on GPU")
        
        return x
    
    def training_step(self, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        """Training step with Kohya-style offloading."""
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
        if self.debug:
            logger.info(f"Batch prepared, GPU memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
        
        # Step 1: VAE encoding with offloading
        step_start = time.time()
        self._ensure_model_on_gpu(self.vae, "VAE")
        
        # VAE encoding with mixed precision support
        if self.mixed_precision:
            with autocast('cuda', dtype=torch.float16):
                if self.enable_vae_training:
                    vae_posterior = self.vae.encode(images).latent_dist
                    latents = vae_posterior.sample()
                else:
                    with torch.no_grad():
                        latents = self.vae.encode(images).latent_dist.sample()
        else:
            if self.enable_vae_training:
                vae_posterior = self.vae.encode(images).latent_dist
                latents = vae_posterior.sample()
            else:
                with torch.no_grad():
                    latents = self.vae.encode(images).latent_dist.sample()
        
        latents = latents * self.vae_scale_factor
        
        # Detach from computation graph if VAE not being trained
        if not self.enable_vae_training:
            latents = latents.detach()
            # Immediately offload VAE to free memory for next models
            self._offload_model_after_use(self.vae, "VAE", keep_on_gpu=False)
        else:
            # If training VAE, offload it temporarily and reload later for reconstruction loss
            self._offload_model_after_use(self.vae, "VAE", keep_on_gpu=False)
        
        if self.debug:
            logger.info(f"VAE encoding done in {time.time() - step_start:.2f}s")
        
        # Step 2: Text encoding with offloading
        step_start = time.time()
        self._ensure_model_on_gpu(self.text_encoder_l, "CLIP-L")
        self._ensure_model_on_gpu(self.text_encoder_g, "CLIP-G")
        
        # Text encoding with mixed precision support
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
            # Same logic for FP32
            if self.enable_text_encoder_training:
                text_outputs_l = self.text_encoder_l(
                    input_ids=input_ids_l,
                    attention_mask=attention_mask_l,
                    output_hidden_states=True,
                )
                text_outputs_g = self.text_encoder_g(
                    input_ids=input_ids_g,
                    attention_mask=attention_mask_g,
                    output_hidden_states=True,
                )
            else:
                with torch.no_grad():
                    text_outputs_l = self.text_encoder_l(
                        input_ids=input_ids_l,
                        attention_mask=attention_mask_l,
                        output_hidden_states=True,
                    )
                    text_outputs_g = self.text_encoder_g(
                        input_ids=input_ids_g,
                        attention_mask=attention_mask_g,
                        output_hidden_states=True,
                    )
            
            text_embeds_l = text_outputs_l.hidden_states[-2]
            pooled_embeds_l = text_outputs_l.pooler_output
            text_embeds_g = text_outputs_g.hidden_states[-2]
            pooled_embeds_g = text_outputs_g.text_embeds
        
        # Combine embeddings
        pooled_embeds = torch.cat([pooled_embeds_l, pooled_embeds_g], dim=-1)
        text_embeds = text_embeds_l  # Use CLIP-L as main text embeddings
        
        # Detach from computation graph if text encoders not being trained
        if not self.enable_text_encoder_training:
            text_embeds = text_embeds.detach()
            pooled_embeds = pooled_embeds.detach()
        
        # Always offload text encoders after encoding to free memory for SiT model
        # (we'll reload them later if needed for gradients)
        self._offload_model_after_use(self.text_encoder_l, "CLIP-L", keep_on_gpu=False)
        self._offload_model_after_use(self.text_encoder_g, "CLIP-G", keep_on_gpu=False)
        
        if self.debug:
            logger.info(f"Text encoding done in {time.time() - step_start:.2f}s")
        
        # Step 3: Noise and timesteps
        noise = torch.randn_like(latents, device=self.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()
        
        # Step 4: Add noise
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Step 5: SiT inference with block-wise offloading
        step_start = time.time()
        
        # For Kohya-style offloading, we'll use a simpler approach
        # Keep the SiT model on CPU initially and load it fully when needed
        # This avoids device mismatch issues while still providing memory benefits
        
        if self.debug:
            logger.info("SiT model kept on CPU, will load to GPU during forward pass")
        
        # SiT inference with progressive block-wise offloading
        if self.mixed_precision:
            with autocast('cuda', dtype=torch.float16):
                noise_pred = self._forward_with_progressive_block_offloading(
                    noisy_latents,
                    timesteps,
                    text_embeds,
                    pooled_embeds,
                )
        else:
            noise_pred = self._forward_with_progressive_block_offloading(
                noisy_latents,
                timesteps,
                text_embeds,
                pooled_embeds,
            )
        
        if self.debug:
            logger.info(f"SiT inference done in {time.time() - step_start:.2f}s")
        
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
            # Ensure VAE is on GPU for reconstruction loss calculation
            self._ensure_model_on_device(self.vae, "VAE", self.device)
            
            # Scale latents back to VAE space
            decoded_latents = latents / self.vae_scale_factor
            
            if self.mixed_precision:
                with autocast('cuda', dtype=torch.float16):
                    reconstructed = self.vae.decode(decoded_latents).sample
            else:
                reconstructed = self.vae.decode(decoded_latents).sample
            
            # Calculate reconstruction loss
            target_images = images.to(dtype=reconstructed.dtype, device=reconstructed.device)
            if target_images.min() >= 0.0:  # Normalize to [-1, 1] if needed
                target_images = target_images * 2.0 - 1.0
            
            vae_loss = torch.nn.functional.mse_loss(reconstructed, target_images)
        
        total_loss = diffusion_loss + 0.01 * vae_loss
        
        # CRITICAL: Before returning (and thus before backward pass),
        # ensure ALL models that need gradients are on GPU
        if self.enable_text_encoder_training:
            self._ensure_model_on_device(self.text_encoder_l, "CLIP-L", self.device)
            self._ensure_model_on_device(self.text_encoder_g, "CLIP-G", self.device)
        
        # SiT model should already be on GPU from forward pass
        # VAE should be on GPU if training VAE
        
        if self.debug:
            logger.info(f"Total training step: {time.time() - start_time:.2f}s")
            logger.info("All trainable models confirmed on GPU for backward pass")
            
            # Debug: Check device placement of all models
            self._debug_model_devices()
        
        # Models remain on GPU for backward pass
        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "vae_loss": vae_loss,
        }
    
    def finalize_step(self):
        """Finalize step with progressive block offloading after backward pass."""
        if self.debug:
            logger.info("Finalizing step with progressive block offloading")
        
        # Clean up backward hooks
        if hasattr(self, '_backward_hooks'):
            for hook in self._backward_hooks:
                hook.remove()
            del self._backward_hooks
            if self.debug:
                logger.info("Cleaned up backward hooks")
        
        # Progressive offloading for SiT transformer blocks
        if hasattr(self.sit_model, 'blocks'):
            blocks = self.sit_model.blocks
            total_blocks = len(blocks)
            
            # Offload all but the last few blocks to save memory for next iteration
            for i in range(total_blocks - self.blocks_to_keep_on_gpu):
                self._ensure_model_on_device(blocks[i], f"SiT-block-{i}", self.cpu_device)
                if self.debug and i < 3:  # Log first few
                    logger.info(f"Offloaded SiT block {i} in finalize_step")
            
            # Keep last few blocks on GPU for faster access next iteration
            for i in range(total_blocks - self.blocks_to_keep_on_gpu, total_blocks):
                if self.debug and i < total_blocks - self.blocks_to_keep_on_gpu + 3:  # Log first few kept
                    logger.info(f"Keeping SiT block {i} on GPU for next iteration")
            
            # Offload non-block components
            for name, module in self.sit_model.named_children():
                if name != 'blocks':
                    self._ensure_model_on_device(module, f"SiT-{name}", self.cpu_device)
            
            # Also offload top-level parameters and buffers
            for name, param in self.sit_model.named_parameters():
                if not name.startswith('blocks.') and param.device != self.cpu_device:
                    param.data = param.data.to(self.cpu_device)
            
            for name, buffer in self.sit_model.named_buffers():
                if not name.startswith('blocks.') and buffer.device != self.cpu_device:
                    buffer.data = buffer.data.to(self.cpu_device)
        else:
            # Fallback: offload entire model
            self._ensure_model_on_device(self.sit_model, "SiT-XL", self.cpu_device)
        
        # Offload other models
        self._ensure_model_on_device(self.vae, "VAE", self.cpu_device) 
        self._ensure_model_on_device(self.text_encoder_l, "CLIP-L", self.cpu_device)
        self._ensure_model_on_device(self.text_encoder_g, "CLIP-G", self.cpu_device)
        
        torch.cuda.empty_cache()
        
        if self.debug:
            memory_after_offload = torch.cuda.memory_allocated(self.device) / 1024**3
            logger.info(f"Step finalized, memory: {memory_after_offload:.2f} GB")
            
            if hasattr(self.sit_model, 'blocks'):
                blocks_on_gpu = sum(1 for i, block in enumerate(self.sit_model.blocks) 
                                  if next(block.parameters()).device == self.device)
                logger.info(f"Finalize: {blocks_on_gpu}/{len(self.sit_model.blocks)} SiT blocks remain on GPU")