"""
VAE utilities for memory-efficient encoding/decoding.
Implements slicing and tiling similar to kohya-ss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SlicedVAEEncoder:
    """
    Memory-efficient VAE encoder that processes images in slices.
    Similar to kohya-ss implementation.
    """
    
    def __init__(
        self,
        vae: nn.Module,
        slice_size: int = 4,  # Number of slices to divide the image
        overlap: int = 64,    # Overlap between slices in pixels
    ):
        self.vae = vae
        self.slice_size = slice_size
        self.overlap = overlap
        
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latents using sliced processing.
        
        Args:
            x: Input images tensor (B, C, H, W)
            
        Returns:
            Latent tensor
        """
        if self.slice_size <= 1:
            # No slicing, regular encoding
            return self.vae.encode(x).latent_dist.sample()
            
        B, C, H, W = x.shape
        
        # Calculate slice dimensions
        slice_h = H // self.slice_size + self.overlap
        slice_w = W // self.slice_size + self.overlap
        
        # Ensure slice dimensions are divisible by VAE factor (usually 8)
        vae_factor = 8
        slice_h = ((slice_h + vae_factor - 1) // vae_factor) * vae_factor
        slice_w = ((slice_w + vae_factor - 1) // vae_factor) * vae_factor
        
        # Calculate latent dimensions
        latent_h = H // vae_factor
        latent_w = W // vae_factor
        latent_slice_h = slice_h // vae_factor
        latent_slice_w = slice_w // vae_factor
        
        # Initialize output tensor
        latents = torch.zeros(
            (B, self.vae.config.latent_channels, latent_h, latent_w),
            device=x.device,
            dtype=x.dtype
        )
        
        # Process each slice
        for i in range(self.slice_size):
            for j in range(self.slice_size):
                # Calculate slice boundaries
                h_start = i * (H - self.overlap) // self.slice_size
                w_start = j * (W - self.overlap) // self.slice_size
                h_end = min(h_start + slice_h, H)
                w_end = min(w_start + slice_w, W)
                
                # Extract slice
                slice_input = x[:, :, h_start:h_end, w_start:w_end]
                
                # Pad if necessary
                pad_h = slice_h - (h_end - h_start)
                pad_w = slice_w - (w_end - w_start)
                if pad_h > 0 or pad_w > 0:
                    slice_input = F.pad(slice_input, (0, pad_w, 0, pad_h), mode='reflect')
                
                # Encode slice
                slice_latent = self.vae.encode(slice_input).latent_dist.sample()
                
                # Calculate output position
                out_h_start = h_start // vae_factor
                out_w_start = w_start // vae_factor
                out_h_end = min(out_h_start + latent_slice_h, latent_h)
                out_w_end = min(out_w_start + latent_slice_w, latent_w)
                
                # Copy to output (handle overlap by averaging)
                if i == 0 and j == 0:
                    latents[:, :, out_h_start:out_h_end, out_w_start:out_w_end] = \
                        slice_latent[:, :, :out_h_end-out_h_start, :out_w_end-out_w_start]
                else:
                    # Blend with existing values in overlap regions
                    blend_h_start = max(0, self.overlap // vae_factor // 2) if i > 0 else 0
                    blend_w_start = max(0, self.overlap // vae_factor // 2) if j > 0 else 0
                    
                    latents[:, :, out_h_start:out_h_end, out_w_start:out_w_end] = \
                        (latents[:, :, out_h_start:out_h_end, out_w_start:out_w_end] + 
                         slice_latent[:, :, :out_h_end-out_h_start, :out_w_end-out_w_start]) / 2
                
                # Clear cache after each slice
                torch.cuda.empty_cache()
                
        return latents


class SlicedVAEDecoder:
    """
    Memory-efficient VAE decoder that processes latents in slices.
    """
    
    def __init__(
        self,
        vae: nn.Module,
        slice_size: int = 4,
        overlap: int = 8,  # Overlap in latent space
    ):
        self.vae = vae
        self.slice_size = slice_size
        self.overlap = overlap
        
    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images using sliced processing.
        
        Args:
            z: Latent tensor (B, C, H, W)
            
        Returns:
            Decoded images tensor
        """
        if self.slice_size <= 1:
            # No slicing, regular decoding
            return self.vae.decode(z).sample
            
        B, C, H, W = z.shape
        
        # Calculate slice dimensions
        slice_h = H // self.slice_size + self.overlap
        slice_w = W // self.slice_size + self.overlap
        
        # VAE upscaling factor (usually 8)
        vae_factor = 8
        
        # Calculate output dimensions
        out_h = H * vae_factor
        out_w = W * vae_factor
        out_slice_h = slice_h * vae_factor
        out_slice_w = slice_w * vae_factor
        
        # Initialize output tensor
        output = torch.zeros(
            (B, 3, out_h, out_w),
            device=z.device,
            dtype=z.dtype
        )
        weight_map = torch.zeros(
            (1, 1, out_h, out_w),
            device=z.device,
            dtype=z.dtype
        )
        
        # Process each slice
        for i in range(self.slice_size):
            for j in range(self.slice_size):
                # Calculate slice boundaries
                h_start = i * (H - self.overlap) // self.slice_size
                w_start = j * (W - self.overlap) // self.slice_size
                h_end = min(h_start + slice_h, H)
                w_end = min(w_start + slice_w, W)
                
                # Extract slice
                slice_input = z[:, :, h_start:h_end, w_start:w_end]
                
                # Pad if necessary
                pad_h = slice_h - (h_end - h_start)
                pad_w = slice_w - (w_end - w_start)
                if pad_h > 0 or pad_w > 0:
                    slice_input = F.pad(slice_input, (0, pad_w, 0, pad_h), mode='reflect')
                
                # Decode slice
                slice_output = self.vae.decode(slice_input).sample
                
                # Calculate output position
                out_h_start = h_start * vae_factor
                out_w_start = w_start * vae_factor
                out_h_end = min(out_h_start + out_slice_h, out_h)
                out_w_end = min(out_w_start + out_slice_w, out_w)
                
                # Create weight for blending
                weight = self._create_blend_weight(
                    out_h_end - out_h_start,
                    out_w_end - out_w_start,
                    i > 0,
                    j > 0,
                    i < self.slice_size - 1,
                    j < self.slice_size - 1
                ).to(z.device)
                
                # Add to output with weighting
                output[:, :, out_h_start:out_h_end, out_w_start:out_w_end] += \
                    slice_output[:, :, :out_h_end-out_h_start, :out_w_end-out_w_start] * weight
                weight_map[:, :, out_h_start:out_h_end, out_w_start:out_w_end] += weight
                
                # Clear cache after each slice
                torch.cuda.empty_cache()
        
        # Normalize by weights
        output = output / (weight_map + 1e-8)
        
        return output
    
    def _create_blend_weight(
        self,
        h: int,
        w: int,
        blend_top: bool,
        blend_left: bool,
        blend_bottom: bool,
        blend_right: bool
    ) -> torch.Tensor:
        """Create blending weight for smooth transitions."""
        weight = torch.ones((1, 1, h, w))
        blend_size = self.overlap * 8  # Convert to pixel space
        
        if blend_top:
            for i in range(blend_size):
                weight[:, :, i, :] *= i / blend_size
                
        if blend_bottom:
            for i in range(blend_size):
                weight[:, :, -(i+1), :] *= i / blend_size
                
        if blend_left:
            for i in range(blend_size):
                weight[:, :, :, i] *= i / blend_size
                
        if blend_right:
            for i in range(blend_size):
                weight[:, :, :, -(i+1)] *= i / blend_size
                
        return weight


def enable_vae_slicing(vae: nn.Module, slice_size: int = 4) -> Tuple[SlicedVAEEncoder, SlicedVAEDecoder]:
    """
    Enable VAE slicing for memory-efficient processing.
    
    Args:
        vae: VAE model
        slice_size: Number of slices per dimension (4 = 16 total slices)
        
    Returns:
        Tuple of (encoder, decoder) with slicing enabled
    """
    encoder = SlicedVAEEncoder(vae, slice_size=slice_size)
    decoder = SlicedVAEDecoder(vae, slice_size=slice_size)
    
    return encoder, decoder