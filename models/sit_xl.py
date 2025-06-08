"""
SiT-XL (Scalable interpolant Transformer) model implementation for SDXL-compatible image generation.
Based on https://github.com/willisma/SiT but adapted for SDXL pipeline with 2-3B parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint

# Try to import flash attention
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Flash Attention not available, using standard attention")


class TimestepEmbedding(nn.Module):
    """Timestep embedding with sinusoidal encoding."""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        # MLP for timestep embedding
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor of shape (batch_size,)
        Returns:
            Timestep embeddings of shape (batch_size, dim * 4)
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Apply MLP
        embedding = self.mlp(embedding)
        return embedding


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization with timestep and condition modulation."""
    
    def __init__(self, dim: int, time_embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        
        # Modulation parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 6 * dim, bias=True)
        )
        
        # Initialize modulation to identity
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (B, N, D)
            emb: Embedding tensor of shape (B, time_embed_dim)
        Returns:
            Tuple of (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(emb).chunk(6, dim=-1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


class Attention(nn.Module):
    """Multi-head self-attention with RoPE (Rotary Position Embedding)."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # RoPE parameters
        self.register_buffer("freqs_cos", None)
        self.register_buffer("freqs_sin", None)
        
        if self.use_flash_attn and FLASH_ATTN_AVAILABLE:
            print("Flash Attention available and will be used during training")
    
    def init_rope(self, max_seq_len: int):
        """Initialize RoPE frequencies."""
        dim = self.head_dim
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(max_seq_len, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        
        self.freqs_cos = torch.cos(freqs)
        self.freqs_sin = torch.sin(freqs)
    
    def apply_rope(self, xq: torch.Tensor, xk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to queries and keys."""
        # xq, xk shape: (B, num_heads, seq_len, head_dim)
        B, H, N, D = xq.shape
        
        # If RoPE not initialized or sequence length changed, reinitialize
        if self.freqs_cos is None or self.freqs_cos.shape[0] < N:
            device = xq.device
            dim = D
            freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
            t = torch.arange(N, device=device)
            freqs = torch.outer(t, freqs.to(device)).float()
            self.freqs_cos = torch.cos(freqs).to(device)
            self.freqs_sin = torch.sin(freqs).to(device)
        
        # Get the required portion of frequencies
        freqs_cos = self.freqs_cos[:N].unsqueeze(0).unsqueeze(0)  # (1, 1, N, D/2)
        freqs_sin = self.freqs_sin[:N].unsqueeze(0).unsqueeze(0)  # (1, 1, N, D/2)
        
        # Split queries and keys into real and imaginary parts
        xq_r, xq_i = xq[..., ::2], xq[..., 1::2]
        xk_r, xk_i = xk[..., ::2], xk[..., 1::2]
        
        # Apply rotation
        xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
        xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
        xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
        xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
        
        # Interleave real and imaginary parts
        xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(-2)
        xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(-2)
        
        return xq_out, xk_out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Apply RoPE if initialized
        if self.freqs_cos is not None:
            q, k = self.apply_rope(q, k)
        
        if self.use_flash_attn and self.training:
            # Use Flash Attention during training
            # Ensure correct dtype for flash attention
            orig_dtype = q.dtype
            if orig_dtype not in [torch.float16, torch.bfloat16]:
                q = q.half()
                k = k.half()
                v = v.half()
            
            # Reshape for flash attention: (B, N, H, D)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Flash attention with dropout
            # Handle gradient checkpointing wrapper
            attn_drop = self.attn_drop
            if hasattr(attn_drop, 'module'):  # If wrapped
                dropout_p = attn_drop.module.p if self.training else 0.0
            else:
                dropout_p = attn_drop.p if self.training else 0.0
            x = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=self.scale)
            x = x.reshape(B, N, C)
            
            # Convert back to original dtype if needed
            if orig_dtype not in [torch.float16, torch.bfloat16]:
                x = x.to(orig_dtype)
        else:
            # Standard attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SiTBlock(nn.Module):
    """SiT Transformer block with adaptive layer normalization."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        time_embed_dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, use_flash_attn=use_flash_attn)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), dropout=drop)
        
        # Adaptive layer norm modulation
        self.adaLN = AdaptiveLayerNorm(dim, time_embed_dim)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, D)
            time_emb: Time embedding of shape (B, time_embed_dim)
        Returns:
            Output tensor of shape (B, N, D)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(x, time_emb)
        
        # Self-attention with modulation
        x_norm = self.norm1(x)
        x_mod = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_mod)
        
        # FFN with modulation
        x_norm = self.norm2(x)
        x_mod = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mod)
        
        return x


class SiTXL(nn.Module):
    """
    SiT-XL model for SDXL-compatible image generation.
    Target: 2-3B parameters for efficiency while maintaining quality.
    """
    
    def __init__(
        self,
        in_channels: int = 4,  # VAE latent channels
        out_channels: int = 4,  # VAE latent channels
        patch_size: int = 2,  # Patch size for latent space
        dim: int = 1536,  # Hidden dimension (reduced from typical XL for 2-3B params)
        depth: int = 48,  # Number of transformer blocks
        num_heads: int = 24,  # Number of attention heads
        mlp_ratio: float = 4.0,
        class_embed_dim: int = 1280,  # CLIP text embedding dimension
        num_classes: Optional[int] = None,  # For class conditioning
        learn_sigma: bool = False,  # Whether to learn variance
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_flash_attn: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_features = self.dim = dim
        self.learn_sigma = learn_sigma
        self.use_checkpoint = use_checkpoint
        
        # Maximum number of patches (for 256x256 latents with patch_size=2, we get 128x128=16384 patches)
        # We use a larger value to accommodate various aspect ratios
        self.max_patches = 16384
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_patches, dim))
        
        # Time embedding
        time_embed_dim = dim * 4
        self.time_embed = TimestepEmbedding(dim, max_period=10000)
        
        # Text conditioning
        # SDXL uses CLIP-L (768 dim) and CLIP-G (1280 dim) = 2048 dim total
        text_embed_dim = 768 + 1280  # CLIP-L + CLIP-G pooled dimensions
        self.text_embed = nn.Sequential(
            nn.Linear(text_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Additional conditioning embedding
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, time_embed_dim)
        else:
            self.class_embed = None
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SiTBlock(
                dim=dim,
                num_heads=num_heads,
                time_embed_dim=time_embed_dim,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                use_flash_attn=use_flash_attn,
            )
            for _ in range(depth)
        ])
        
        # Output layers
        self.norm_out = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, patch_size ** 2 * out_channels * (2 if learn_sigma else 1))
        
        # Initialize weights
        self._init_weights()
        
        # Initialize RoPE for all attention layers
        max_seq_len = self.max_patches
        for block in self.blocks:
            block.attn.init_rope(max_seq_len)
        
        # Print model size
        self._print_model_size()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize position embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize patch embedding
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.size(0), -1))
        
        # Initialize output projection to zero
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        
        # Initialize other layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.elementwise_affine and module.weight is not None:
                    nn.init.ones_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        self.apply(_basic_init)
    
    def _print_model_size(self):
        """Print model size in billions of parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"SiT-XL Model Size: {total_params / 1e9:.2f}B parameters")
    
    def unpatchify(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Reconstruct images from patches.
        Args:
            x: (B, N, patch_size^2 * C)
            H, W: Height and width of latent
        Returns:
            imgs: (B, C, H, W)
        """
        B = x.shape[0]
        C = self.out_channels * (2 if self.learn_sigma else 1)
        P = self.patch_size
        
        h = H // P
        w = W // P
        
        x = x.reshape(B, h, w, P, P, C)
        x = torch.einsum('bhwpqc->bchpwq', x)
        imgs = x.reshape(B, C, H, W)
        
        return imgs
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_embeddings_pool: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of SiT-XL.
        Args:
            x: Latent input tensor of shape (B, C, H, W)
            t: Timestep tensor of shape (B,)
            text_embeddings: Text embeddings from CLIP of shape (B, seq_len, embed_dim)
            text_embeddings_pool: Pooled text embeddings of shape (B, embed_dim)
            class_labels: Optional class labels of shape (B,)
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, dim, H/P, W/P)
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Add position embedding
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Time embedding
        time_emb = self.time_embed(t)  # (B, dim * 4)
        
        # Text conditioning
        # Use pooled embeddings from both CLIP models
        if text_embeddings_pool is not None:
            # Concatenate pooled embeddings from CLIP-L and CLIP-G
            text_cond = self.text_embed(text_embeddings_pool)  # (B, dim * 4)
            time_emb = time_emb + text_cond
        
        # Class conditioning
        if self.class_embed is not None and class_labels is not None:
            class_emb = self.class_embed(class_labels)
            time_emb = time_emb + class_emb
        
        # Apply transformer blocks (gradient checkpointing disabled for stability)
        for block in self.blocks:
            x = block(x, time_emb)
        
        # Output projection
        x = self.norm_out(x)
        x = self.proj_out(x)
        
        # Unpatchify
        x = self.unpatchify(x, H, W)
        
        if self.learn_sigma:
            x, _ = torch.chunk(x, 2, dim=1)
        
        return x


# Model configurations
def sit_xl_1(**kwargs) -> SiTXL:
    """SiT-XL configuration targeting ~1B parameters for memory-constrained setups."""
    return SiTXL(
        dim=1024,
        depth=28,
        num_heads=16,
        patch_size=2,
        use_flash_attn=True,
        use_checkpoint=False,  # Disable gradient checkpointing for stability
        **kwargs
    )


def sit_xl_512m(**kwargs) -> SiTXL:
    """SiT-XL configuration targeting ~512M parameters for extreme memory constraints."""
    return SiTXL(
        dim=768,
        depth=20,
        num_heads=12,
        patch_size=2,
        use_flash_attn=True,
        use_checkpoint=False,
        **kwargs
    )


def sit_xl_2(**kwargs) -> SiTXL:
    """SiT-XL configuration targeting ~2B parameters."""
    return SiTXL(
        dim=1536,
        depth=42,
        num_heads=24,
        patch_size=2,
        use_flash_attn=True,
        use_checkpoint=True,
        **kwargs
    )


def sit_xl_3(**kwargs) -> SiTXL:
    """SiT-XL configuration targeting ~3B parameters."""
    return SiTXL(
        dim=1792,
        depth=48,
        num_heads=28,
        patch_size=2,
        use_flash_attn=True,
        use_checkpoint=True,
        **kwargs
    )


if __name__ == "__main__":
    # Test model instantiation
    model = sit_xl_2()
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 4, 128, 128)  # Latent input
    t = torch.randint(0, 1000, (batch_size,))  # Timesteps
    text_emb = torch.randn(batch_size, 77, 1280)  # CLIP text embeddings
    text_pool = torch.randn(batch_size, 1280 * 2)  # Pooled embeddings (CLIP-L + CLIP-G)
    
    with torch.no_grad():
        out = model(x, t, text_emb, text_pool)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
