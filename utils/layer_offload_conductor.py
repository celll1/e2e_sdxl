"""
LayerOffloadConductor implementation based on OneTrainer.
https://github.com/Nerogar/OneTrainer/blob/84d6c2c4742bc226ccd3e34dbf35398018795bb5/modules/util/LayerOffloadConductor.py
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Any
from contextlib import contextmanager
import gc


class LayerOffloadConductor:
    """
    OneTrainer-style layer offloading conductor that manages GPU/CPU placement
    of model layers during training to minimize VRAM usage.
    """
    
    def __init__(self, device: torch.device = torch.device("cuda")):
        self.device = device
        self.cpu_device = torch.device("cpu")
        self.layers: Dict[str, nn.Module] = {}
        self.layer_dependencies: Dict[str, Set[str]] = {}
        self.active_layers: Set[str] = set()
        self.pinned_layers: Set[str] = set()
        
    def add_layer(self, name: str, layer: nn.Module, dependencies: Optional[List[str]] = None):
        """Add a layer to be managed by the conductor."""
        self.layers[name] = layer
        self.layer_dependencies[name] = set(dependencies or [])
        
        # Start with layer on CPU
        layer.to(self.cpu_device)
        
    def pin_layer(self, name: str):
        """Pin a layer to GPU (won't be offloaded)."""
        if name in self.layers:
            self.pinned_layers.add(name)
            self.layers[name].to(self.device)
            self.active_layers.add(name)
    
    def unpin_layer(self, name: str):
        """Unpin a layer (can be offloaded)."""
        if name in self.pinned_layers:
            self.pinned_layers.remove(name)
    
    @contextmanager
    def use_layer(self, name: str):
        """Context manager for using a layer. Ensures it's on GPU during use."""
        if name not in self.layers:
            raise ValueError(f"Layer {name} not found")
        
        # Activate layer and its dependencies
        self._activate_layer(name)
        
        try:
            yield self.layers[name]
        finally:
            # Deactivate layer if not pinned
            self._deactivate_layer(name)
    
    @contextmanager
    def use_layers(self, names: List[str]):
        """Context manager for using multiple layers simultaneously."""
        for name in names:
            if name not in self.layers:
                raise ValueError(f"Layer {name} not found")
        
        # Activate all layers and their dependencies
        for name in names:
            self._activate_layer(name)
        
        try:
            yield {name: self.layers[name] for name in names}
        finally:
            # Deactivate layers if not pinned
            for name in names:
                self._deactivate_layer(name)
    
    def _activate_layer(self, name: str):
        """Move layer and its dependencies to GPU."""
        # Activate dependencies first
        for dep in self.layer_dependencies[name]:
            if dep not in self.active_layers:
                self._activate_layer(dep)
        
        # Move layer to GPU if not already there
        if name not in self.active_layers:
            layer = self.layers[name]
            layer.to(self.device)
            self.active_layers.add(name)
    
    def _deactivate_layer(self, name: str):
        """Move layer to CPU if not pinned or needed by other layers."""
        if name in self.pinned_layers:
            return
        
        # Check if any other active layer depends on this one
        for active_layer in self.active_layers:
            if name in self.layer_dependencies.get(active_layer, set()):
                return
        
        # Move to CPU
        if name in self.active_layers:
            layer = self.layers[name]
            layer.to(self.cpu_device)
            self.active_layers.remove(name)
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
    
    def offload_all(self):
        """Offload all unpinned layers to CPU."""
        for name, layer in self.layers.items():
            if name not in self.pinned_layers:
                layer.to(self.cpu_device)
                self.active_layers.discard(name)
        
        torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            cached = torch.cuda.memory_reserved(self.device) / 1024**3
            return {
                "allocated_gb": allocated,
                "cached_gb": cached,
                "active_layers": len(self.active_layers),
                "pinned_layers": len(self.pinned_layers),
                "active_layer_names": list(self.active_layers),
                "pinned_layer_names": list(self.pinned_layers),
            }
        return {}


class OffloadedE2EModel:
    """
    E2E model using LayerOffloadConductor for memory-efficient training.
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
    ):
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.vae_scale_factor = vae_scale_factor
        self.enable_vae_training = enable_vae_training
        self.enable_text_encoder_training = enable_text_encoder_training
        
        # Create offload conductor
        self.conductor = LayerOffloadConductor(device)
        
        # Add models to conductor
        self.conductor.add_layer("sit_model", sit_model)
        self.conductor.add_layer("vae", vae)
        self.conductor.add_layer("text_encoder_l", text_encoder_l)
        self.conductor.add_layer("text_encoder_g", text_encoder_g)
        
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
    
    def training_step(self, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        """Memory-optimized training step using layer offloading."""
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
        
        # Step 1: Encode images with VAE
        with self.conductor.use_layer("vae") as vae:
            # Ensure same device and dtype
            vae_device = next(vae.parameters()).device
            vae_dtype = next(vae.parameters()).dtype
            images_vae = images.to(device=vae_device, dtype=vae_dtype)
            
            if self.enable_vae_training:
                latents = vae.encode(images_vae).latent_dist.sample()
            else:
                with torch.no_grad():
                    latents = vae.encode(images_vae).latent_dist.sample()
            
            latents = latents * self.vae_scale_factor
            latents = latents.to(self.device)  # Move back to main device
        
        # Step 2: Encode text with both encoders
        with self.conductor.use_layers(["text_encoder_l", "text_encoder_g"]) as encoders:
            text_encoder_l = encoders["text_encoder_l"]
            text_encoder_g = encoders["text_encoder_g"]
            
            # Ensure same device
            enc_device = next(text_encoder_l.parameters()).device
            input_ids_l_enc = input_ids_l.to(enc_device)
            input_ids_g_enc = input_ids_g.to(enc_device)
            attention_mask_l_enc = attention_mask_l.to(enc_device) if attention_mask_l is not None else None
            attention_mask_g_enc = attention_mask_g.to(enc_device) if attention_mask_g is not None else None
            
            # CLIP-L encoding
            text_outputs_l = text_encoder_l(
                input_ids=input_ids_l_enc,
                attention_mask=attention_mask_l_enc,
                output_hidden_states=True,
            )
            text_embeds_l = text_outputs_l.hidden_states[-2]
            pooled_embeds_l = text_outputs_l.pooler_output
            
            # CLIP-G encoding  
            text_outputs_g = text_encoder_g(
                input_ids=input_ids_g_enc,
                attention_mask=attention_mask_g_enc,
                output_hidden_states=True,
            )
            text_embeds_g = text_outputs_g.hidden_states[-2]
            pooled_embeds_g = text_outputs_g.text_embeds
            
            # Combine embeddings
            pooled_embeds = torch.cat([pooled_embeds_l, pooled_embeds_g], dim=-1)
            text_embeds = text_embeds_l  # Use CLIP-L embeddings
            
            # Move to main device
            text_embeds = text_embeds.to(self.device)
            pooled_embeds = pooled_embeds.to(self.device)
        
        # Step 3: Prepare noise and timesteps
        noise = torch.randn_like(latents, device=self.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()
        
        # Step 4: Add noise
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Step 5: Predict noise with SiT model
        with self.conductor.use_layer("sit_model") as sit_model:
            # Ensure same device and dtype
            sit_device = next(sit_model.parameters()).device
            sit_dtype = next(sit_model.parameters()).dtype
            
            noisy_latents_sit = noisy_latents.to(device=sit_device, dtype=sit_dtype)
            timesteps_sit = timesteps.to(sit_device)
            text_embeds_sit = text_embeds.to(device=sit_device, dtype=sit_dtype)
            pooled_embeds_sit = pooled_embeds.to(device=sit_device, dtype=sit_dtype)
            
            noise_pred = sit_model(
                noisy_latents_sit,
                timesteps_sit,
                text_embeds_sit,
                pooled_embeds_sit,
            )
            
            # Move back to main device
            noise_pred = noise_pred.to(self.device)
        
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