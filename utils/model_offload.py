"""
Model offloading utilities for memory-efficient training.
<<<<<<< HEAD
Based on techniques from kohya-ss/sd-scripts and OneTrainer.
=======
Independent implementation for CPU/GPU model management.
>>>>>>> fd5c51c (fixes)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
import gc


class ModelOffloader:
    """Handles offloading models between GPU and CPU/RAM."""
    
    def __init__(self, device: torch.device = torch.device("cuda")):
        self.device = device
        self.cpu_device = torch.device("cpu")
        self.offloaded_modules = {}
        
    def offload_module(self, module: nn.Module, name: str):
        """Offload a module to CPU."""
        module.to(self.cpu_device)
        self.offloaded_modules[name] = module
        torch.cuda.empty_cache()
        
    def load_module(self, name: str) -> nn.Module:
        """Load a module back to GPU."""
        if name in self.offloaded_modules:
            module = self.offloaded_modules[name]
            module.to(self.device)
            return module
        else:
            raise ValueError(f"Module {name} not found in offloaded modules")
    
    def clear_cache(self):
        """Clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()


class SequentialOffloader:
    """
    Sequential model offloading for encoder-decoder architectures.
    Keeps only the active model on GPU.
    """
    
    def __init__(self, device: torch.device = torch.device("cuda")):
        self.device = device
        self.cpu_device = torch.device("cpu")
        self.active_module = None
        
    def offload_all_except(self, modules: Dict[str, nn.Module], keep: Optional[str] = None):
        """Offload all modules except the specified one."""
        for name, module in modules.items():
            if name != keep:
                # Only offload if not already on CPU
                if next(module.parameters()).device != self.cpu_device:
                    module.to(self.cpu_device)
            else:
                # Only move to GPU if not already there
                if next(module.parameters()).device != self.device:
                    module.to(self.device)
                self.active_module = name
        # Clear cache less frequently to reduce overhead
        if keep is None:  # Only clear when offloading everything
            torch.cuda.empty_cache()
    
    def activate_module(self, modules: Dict[str, nn.Module], name: str):
        """Activate a specific module and offload others."""
        self.offload_all_except(modules, keep=name)


class GradientCheckpointingWrapper(nn.Module):
    """
    Wrapper to apply gradient checkpointing to any module.
    """
    
    def __init__(self, module: nn.Module, use_reentrant: bool = False):
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant
    
    def __getattr__(self, name):
        """Forward attribute access to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
    def forward(self, *args, **kwargs):
        if self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self.module, *args, use_reentrant=self.use_reentrant, **kwargs)
        else:
            return self.module(*args, **kwargs)


def enable_memory_efficient_attention(model: nn.Module):
    """
    Enable memory efficient attention for all applicable layers.
    Tries xformers first, then falls back to flash attention or torch SDPA.
    """
    try:
        import xformers
        import xformers.ops
        
        def replace_attention(module):
            for name, child in module.named_children():
                if hasattr(child, 'set_use_memory_efficient_attention_xformers'):
                    child.set_use_memory_efficient_attention_xformers(True)
                else:
                    replace_attention(child)
        
        replace_attention(model)
        print("Enabled xformers memory efficient attention")
        return True
    except ImportError:
        pass
    
    # Try torch's native SDPA
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        print("Using PyTorch's native scaled_dot_product_attention")
        return True
    
    return False


def calculate_model_memory(model: nn.Module) -> float:
    """Calculate approximate memory usage of a model in GB."""
    total_params = 0
    total_buffers = 0
    
    for param in model.parameters():
        total_params += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        total_buffers += buffer.numel() * buffer.element_size()
    
    total_bytes = total_params + total_buffers
    return total_bytes / (1024 ** 3)  # Convert to GB


def optimize_model_memory(model: nn.Module, optimization_level: str = "balanced"):
    """
    Apply memory optimizations based on level:
    - 'minimal': Basic optimizations
    - 'balanced': Gradient checkpointing + some offloading
    - 'aggressive': Maximum memory savings (slower)
    """
    
    if optimization_level == "minimal":
        # Just enable memory efficient attention
        enable_memory_efficient_attention(model)
        
    elif optimization_level == "balanced":
        # Enable gradient checkpointing for transformer blocks
        enable_memory_efficient_attention(model)
        
        # Find and wrap transformer blocks
        for name, module in model.named_modules():
            if "block" in name.lower() or "layer" in name.lower():
                if hasattr(module, 'forward'):
                    # Wrap with gradient checkpointing
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    if parent_name:
                        parent = model
                        for part in parent_name.split('.'):
                            parent = getattr(parent, part)
                        setattr(parent, child_name, GradientCheckpointingWrapper(module))
                        
    elif optimization_level == "aggressive":
        # Maximum memory savings
        enable_memory_efficient_attention(model)
        
        # Enable gradient checkpointing for all layers
        def wrap_all_layers(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if len(list(child.children())) == 0:  # Leaf module
                    if hasattr(child, 'forward') and child.forward.__module__ != 'torch.nn.modules.module':
                        setattr(module, name, GradientCheckpointingWrapper(child))
                else:
                    wrap_all_layers(child, full_name)
        
        wrap_all_layers(model)
        
        # Convert to lower precision where possible
        for param in model.parameters():
            if param.dtype == torch.float32:
                param.data = param.data.half()
                
    torch.cuda.empty_cache()
    return model


class ActivationCheckpointing:
    """
    Custom activation checkpointing that's more memory efficient than PyTorch's default.
    """
    
    @staticmethod
    def checkpoint_sequential(functions, segments, *args):
        """
        Checkpoint a sequential model by dividing it into segments.
        """
        if segments <= 0:
            raise ValueError("Number of segments must be positive")
            
        def run_function(start, end, *args):
            for i in range(start, end):
                args = functions[i](*args) if isinstance(args, tuple) else (functions[i](args),)
            return args
        
        if not torch.is_grad_enabled():
            return run_function(0, len(functions), *args)
            
        # Divide functions into segments
        segment_size = len(functions) // segments
        end = 0
        
        for i in range(segments):
            start = end
            end = start + segment_size if i < segments - 1 else len(functions)
            
            if i < segments - 1:
                args = torch.utils.checkpoint.checkpoint(
                    run_function, start, end, *args,
                    use_reentrant=False
                )
            else:
                args = run_function(start, end, *args)
                
        return args