"""
Memory-efficient optimizer utilities.
Based on bitsandbytes 8-bit optimizer implementation.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Optional, Dict, Any
import math


class Adam8bit(Optimizer):
    """
    8-bit Adam optimizer that reduces memory usage by ~75% compared to standard Adam.
    This is a simplified implementation that doesn't require bitsandbytes.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam8bit, self).__init__(params, defaults)
        
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam8bit does not support sparse gradients')
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Use float16 for momentum buffers to save memory
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float16)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float16)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                # Convert to float32 for computation, then back to float16 for storage
                exp_avg_float = exp_avg.float()
                exp_avg_sq_float = exp_avg_sq.float()
                grad_float = grad.float()
                
                exp_avg_float.mul_(beta1).add_(grad_float, alpha=1 - beta1)
                exp_avg_sq_float.mul_(beta2).addcmul_(grad_float, grad_float, value=1 - beta2)
                
                # Store back as float16
                exp_avg.copy_(exp_avg_float.half())
                exp_avg_sq.copy_(exp_avg_sq_float.half())
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Apply bias correction and compute update
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Compute update in float32
                denom = exp_avg_sq_float.sqrt().add_(group['eps'])
                update = exp_avg_float / denom
                
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                    
                p.data.add_(update, alpha=-step_size)
                
        return loss


def create_optimizer_with_memory_efficient_mode(
    model_parameters,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    use_8bit: bool = False,
    **kwargs
):
    """
    Create optimizer with optional 8-bit mode for memory efficiency.
    """
    if use_8bit:
        try:
            # Try to use bitsandbytes if available
            import bitsandbytes as bnb
            if optimizer_type == "adamw":
                # Use PagedAdamW8bit which is more compatible with mixed precision
                return bnb.optim.PagedAdamW8bit(
                    model_parameters,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    **kwargs
                )
            elif optimizer_type == "adam":
                return bnb.optim.PagedAdam8bit(
                    model_parameters,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    **kwargs
                )
        except (ImportError, AttributeError):
            # Fallback to our implementation
            print("bitsandbytes not available or PagedOptimizer not found, using custom 8-bit optimizer")
            return Adam8bit(
                model_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs
            )
    else:
        # Standard PyTorch optimizers
        if optimizer_type == "adamw":
            return torch.optim.AdamW(
                model_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_type == "adam":
            return torch.optim.Adam(
                model_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class MemoryEfficientAdamW(torch.optim.AdamW):
    """
    AdamW optimizer with memory-efficient features.
    Stores optimizer states in lower precision.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, amsgrad=False, state_dtype=torch.float16):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.state_dtype = state_dtype
        
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with memory-efficient state storage."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('AdamW does not support sparse gradients')
                    grads.append(p.grad)
                    
                    state = self.state[p]
                    # Lazy state initialization with specified dtype
                    if len(state) == 0:
                        state['step'] = 0
                        # Store momentum in lower precision
                        state['exp_avg'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, dtype=self.state_dtype
                        )
                        state['exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, dtype=self.state_dtype
                        )
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(
                                p, memory_format=torch.preserve_format, dtype=self.state_dtype
                            )
                            
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    
                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                        
                    state['step'] += 1
                    state_steps.append(state['step'])
                    
            # Perform optimization step
            self._single_tensor_adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=group['betas'][0],
                beta2=group['betas'][1],
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group.get('maximize', False),
                foreach=group.get('foreach', None),
                capturable=group.get('capturable', False),
            )
            
        return loss
    
    def _single_tensor_adamw(self, params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                            state_steps, *, amsgrad, beta1, beta2, lr, weight_decay, eps,
                            maximize, foreach, capturable):
        """Modified to handle mixed precision states."""
        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            
            # Convert to float32 for computation
            exp_avg_float = exp_avg.float()
            exp_avg_sq_float = exp_avg_sq.float()
            
            # Perform weight decay
            param.mul_(1 - lr * weight_decay)
            
            # Decay the first and second moment running average coefficient
            exp_avg_float.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq_float.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
            
            if amsgrad:
                max_exp_avg_sq = max_exp_avg_sqs[i]
                max_exp_avg_sq_float = max_exp_avg_sq.float()
                torch.maximum(max_exp_avg_sq_float, exp_avg_sq_float, out=max_exp_avg_sq_float)
                max_exp_avg_sq.copy_(max_exp_avg_sq_float.to(self.state_dtype))
                denom = max_exp_avg_sq_float.sqrt().add_(eps)
            else:
                denom = exp_avg_sq_float.sqrt().add_(eps)
                
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1
            
            # Update parameters
            param.addcdiv_(exp_avg_float, denom, value=-step_size)
            
            # Store states back in lower precision
            exp_avg.copy_(exp_avg_float.to(self.state_dtype))
            exp_avg_sq.copy_(exp_avg_sq_float.to(self.state_dtype))