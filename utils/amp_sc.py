import math
from typing import List, Optional, Tuple, Union

import torch


class NullCtx:
    """A no-op context manager for when mixed precision is disabled."""
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AmpOptimizer:
    """A wrapper for PyTorch optimizers that handles mixed precision training, gradient scaling, and clipping."""
    def __init__(
        self,
        mixed_precision: int,
        optimizer: torch.optim.Optimizer,
        names: List[str],
        paras: List[torch.nn.Parameter],
        grad_clip: float,
        n_gradient_accumulation: int = 1,
    ):
        """
        Args:
            mixed_precision (int): Whether to use mixed precision training.
                                  0: Disabled, 1: FP16, 2: BF16.
            optimizer (torch.optim.Optimizer): The base optimizer.
            names (List[str]): Names of the parameters.
            paras (List[torch.nn.Parameter]): Parameters to optimize (must require gradients).
            grad_clip (float): Gradient clipping value. If <= 0, no clipping is applied.
            n_gradient_accumulation (int): Number of gradient accumulation steps.
        """
        self.enable_amp = mixed_precision > 0
        self.using_fp16_rather_bf16 = mixed_precision == 1
        
        # Set up mixed precision context and gradient scaler
        if self.enable_amp:
            self.amp_ctx = torch.autocast(
                'cuda',
                enabled=True,
                dtype=torch.float16 if self.using_fp16_rather_bf16 else torch.bfloat16,
                cache_enabled=True,
            )
            # Only FP16 needs a gradient scaler
            self.scaler = torch.cuda.amp.GradScaler(init_scale=2. ** 11, growth_interval=1000) if self.using_fp16_rather_bf16 else None
        else:
            self.amp_ctx = NullCtx()
            self.scaler = None
        
        self.optimizer = optimizer
        self.names = names
        self.paras = paras  # These parameters have already been filtered to require gradients
        self.grad_clip = grad_clip
        self.early_clipping = self.grad_clip > 0 and not hasattr(optimizer, 'global_grad_norm')
        self.late_clipping = self.grad_clip > 0 and hasattr(optimizer, 'global_grad_norm')
        
        self.r_accu = 1 / n_gradient_accumulation  # Scaling factor for gradient accumulation
    
    def backward_clip_step(
        self, stepping: bool, loss: torch.Tensor,
    ) -> Tuple[Optional[Union[torch.Tensor, float]], Optional[float]]:
        """
        Perform backward pass, gradient clipping, and optimizer step.

        Args:
            stepping (bool): Whether to perform an optimizer step.
            loss (torch.Tensor): The loss tensor to backpropagate.

        Returns:
            Tuple[Optional[Union[torch.Tensor, float]], Optional[float]]:
                - The gradient norm if clipping was applied, otherwise None.
                - The log2 of the gradient scaler scale if using FP16, otherwise None.
        """
        # Scale the loss for gradient accumulation
        loss = loss.mul(self.r_accu)
        
        # Perform backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward(retain_graph=False, create_graph=False)
        else:
            loss.backward(retain_graph=False, create_graph=False)
        
        orig_norm = scaler_sc = None
        
        if stepping:
            # Unscale gradients if using a scaler
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            
            # Apply early gradient clipping if needed
            if self.early_clipping:
                orig_norm = torch.nn.utils.clip_grad_norm_(self.paras, self.grad_clip)
            
            # Perform optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                scaler_sc = self.scaler.get_scale()
                
                # Prevent scaler from overflowing
                if scaler_sc > 32768.0:  # FP16 will overflow when >65536
                    self.scaler.update(new_scale=32768.0)
                else:
                    self.scaler.update()
                
                # Log the scaler scale
                try:
                    scaler_sc = float(math.log2(scaler_sc))
                except Exception as e:
                    print(f'[scaler_sc = {scaler_sc}]\n' * 15, flush=True)
                    raise e
            else:
                self.optimizer.step()
            
            # Apply late gradient clipping if needed
            if self.late_clipping:
                orig_norm = self.optimizer.global_grad_norm
            
            # Reset gradients
            self.optimizer.zero_grad(set_to_none=True)
        
        return orig_norm, scaler_sc
    
    def state_dict(self) -> dict:
        """Return the state of the optimizer and scaler (if applicable)."""
        state = {'optimizer': self.optimizer.state_dict()}
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        return state
    
    def load_state_dict(self, state: dict, strict: bool = True):
        """Load the state of the optimizer and scaler (if applicable)."""
        if self.scaler is not None and 'scaler' in state:
            try:
                self.scaler.load_state_dict(state['scaler'])
            except Exception as e:
                print(f'[FP16 load_state_dict error] {e}')
        self.optimizer.load_state_dict(state['optimizer'])
