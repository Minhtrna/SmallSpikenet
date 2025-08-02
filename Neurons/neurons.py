import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate

class SurrogateGradient(torch.autograd.Function):
    
    
    @staticmethod
    def forward(ctx, input, threshold, num_thresholds):
        ctx.save_for_backward(input, threshold)
        ctx.num_thresholds = num_thresholds
        
        # Forward: discrete computation
        theta_base = threshold / num_thresholds
        highest_crossed = torch.floor(input / theta_base)
        # Mathematical clamp
        highest_crossed = torch.max(torch.zeros_like(highest_crossed), 
                                  torch.min(highest_crossed, 
                                          torch.full_like(highest_crossed, num_thresholds)))
        return highest_crossed
    
    @staticmethod
    def backward(ctx, grad_output):
        input_, threshold = ctx.saved_tensors
        num_thresholds = ctx.num_thresholds
        alpha = 2.0  # alpha for sigmoid derivative
        
        # Multi-threshold gradient: sum of sigmoid derivatives for each level
        theta_base = threshold / num_thresholds
        
        # Vectorized computation for all levels at once
        levels = torch.arange(1, num_thresholds + 1, device=input_.device, dtype=input_.dtype)
        
        # Broadcast: [batch_dims..., 1] vs [num_levels]
        input_expanded = input_.unsqueeze(-1)  # [..., 1]
        level_thresholds = levels * theta_base  # [num_levels]
        
        # Compute sigmoid derivative for all levels simultaneously
        x_shifted = input_expanded - level_thresholds  # [..., num_levels]
        scaled_x = x_shifted * alpha / theta_base
        
        # Sigmoid derivatives for all levels
        sigmoid_vals = torch.sigmoid(scaled_x)
        sigmoid_derivatives = sigmoid_vals * (1.0 - sigmoid_vals) * alpha / theta_base
        
        #  weighting 
        proximity_weights = torch.exp(-0.5 * (x_shifted / theta_base) ** 2)  # Gaussian proximity
        level_weights = (levels / num_thresholds) ** 0.5  # Moderate level bias
        combined_weights = proximity_weights * level_weights.unsqueeze(0)
        
        # Apply weighting
        weighted_derivatives = sigmoid_derivatives * combined_weights
        
        # Normalize by total weight 
        total_weights = combined_weights.sum(dim=-1, keepdim=True) + 1e-8
        grad_input = (weighted_derivatives * grad_output.unsqueeze(-1)).sum(dim=-1) / total_weights.squeeze(-1)
        
        # Enhanced threshold gradient với level-specific contributions
        current_levels = torch.clamp(input_ / theta_base, 0, num_thresholds)
        level_contributions = (weighted_derivatives * levels.unsqueeze(0)).sum(dim=-1)
        grad_threshold = -(level_contributions * grad_output / threshold).sum()
        
        # Clamp instead of tanh for better stability
        grad_input = torch.clamp(grad_input, -1.0, 1.0)
        grad_threshold = torch.clamp(grad_threshold, -0.1, 0.1)
        
        return grad_input, grad_threshold, None

class LM_HT_LIF(neuron.BaseNode):
    """
    Learnable Multi-hierarchical Threshold Leaky Integrate-and-Fire Neuron
    with configurable reset mechanism and Leaky Memory
    
    Reset modes:
    - 'hybrid': Full reset for max threshold, partial reset for intermediate thresholds
    - 'soft': Subtract threshold from membrane potential (v = v - θ)
    - 'hard': Reset to zero (v = 0)
    """
    def __init__(self, init_threshold=1.0, init_leakage=0.2, num_thresholds=4, 
                 memory_factor=0.1, detach_reset=False, reset_mode='hybrid'):
        super().__init__(v_threshold=init_threshold, v_reset=None, detach_reset=detach_reset)
        
        # Initialize parameters
        self.threshold = nn.Parameter(torch.tensor(init_threshold))
        self.leak = nn.Parameter(torch.tensor(init_leakage))
        self.num_thresholds = num_thresholds
        self.memory_factor = memory_factor
        self.memory = None
        
        # Reset mode - validate and set
        valid_modes = ['hybrid', 'soft', 'hard']
        if reset_mode not in valid_modes:
            raise ValueError(f"Reset mode must be one of {valid_modes}, got {reset_mode}")
        self.reset_mode = reset_mode
        
        # For monitoring spike activity
        self.spike_counter = None
        
        # Flag for first forward pass
        self.first_forward = True
    
    def neuronal_charge(self, x):
        """Update membrane potential with proper handling of spikingjelly state types"""
        # Initialize state variables if needed (following spikingjelly pattern)
        if self.v is None or not isinstance(self.v, torch.Tensor) or self.v.shape != x.shape:
            # For first initialization or if wrong type/shape
            if self.v is None or not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros_like(x)
            else:
                # Keep values but match shape (unlikely case)
                self.v = torch.zeros_like(x).fill_(self.v.item() if isinstance(self.v, torch.Tensor) else float(self.v))
            self.first_forward = True
    
        if self.memory is None or not isinstance(self.memory, torch.Tensor) or self.memory.shape != x.shape:
            # Similar handling for memory
            if self.memory is None or not isinstance(self.memory, torch.Tensor):
                self.memory = torch.zeros_like(x)
            else:
                self.memory = torch.zeros_like(x).fill_(self.memory.item() if isinstance(self.memory, torch.Tensor) else float(self.memory))
    
        # Handle first forward pass with proper detach behavior
        if self.first_forward:
            v_for_update = self.v.detach()
            memory_for_update = self.memory.detach()
            self.first_forward = False
        else:
            v_for_update = self.v
            memory_for_update = self.memory
    
        # Use sigmoid instead of clamp for better gradient properties
        leakage_factor = torch.sigmoid(self.leak)
        
        # Update memory with input (no detach)
        memory_new = memory_for_update + self.memory_factor * (x - memory_for_update)
        
        # Apply leakage formula
        v_new = leakage_factor * v_for_update + (1 - leakage_factor) * x
        

        self.memory = memory_new  # No detach
        self.v = v_new  # No detach
        
        return v_new
    
    def neuronal_fire(self):
        """
        Generate spikes với smooth threshold
        """
        # Smooth positive threshold: sqrt(x^2 + ε) thay vì abs(x) + ε
        threshold_value = torch.sqrt(self.threshold * self.threshold + 1e-8)
            
        spike = SurrogateGradient.apply(self.v, threshold_value, self.num_thresholds)
        
        # Monitoring unchanged
        if self.training:
            with torch.no_grad():
                if self.spike_counter is None:
                    self.spike_counter = (spike > 0).float()
                else:
                    self.spike_counter = 0.9 * self.spike_counter + 0.1 * (spike > 0).float()
        
        return spike
    
    def neuronal_reset(self, spike):
        """
        Apply reset mechanism with improved gradient flow using multiplication approach
        
        Args:
            spike: Spike tensor
                
        Returns:
            Updated membrane potential after reset
        """
        # Only detach spike if explicitly requested via detach_reset
        spike_for_reset = spike.detach() if self.detach_reset else spike
        
        # Base threshold unit
        threshold_value = torch.abs(self.threshold) + 1e-6
        threshold_unit = threshold_value / self.num_thresholds
        
        # Apply reset based on selected mode
        if self.reset_mode == 'hybrid':
            # Hybrid reset: full reset for max threshold, partial for intermediate
            # Convert mask to float tensor for multiplication approach
            max_spike_float = (spike_for_reset == self.num_thresholds).float()
            
            # Calculate reset amount for partial reset
            reset_amount = spike_for_reset * threshold_unit
            
            # Apply soft reset for non-max spikes: v = v - reset_amount
            soft_reset = self.v - reset_amount * (1.0 - max_spike_float)
            
            # For max spikes, apply hard reset with small residual (5%)
            # to maintain gradient flow
            residual = 0.05  # Small residual for gradient flow
            
            # Combine using multiplication approach:
            # v = (1-max_spike)*soft_reset + max_spike*(residual*v)
            self.v = (1.0 - max_spike_float) * soft_reset + max_spike_float * (residual * self.v)
            
        elif self.reset_mode == 'soft':
            # Soft reset using SpikinJelly-style multiplication
            # v = v - spike * threshold_unit
            reset_amount = spike_for_reset * threshold_unit
            self.v = self.v - reset_amount
            
        elif self.reset_mode == 'hard':
            # Hard reset using SpikinJelly-style multiplication with small residual
            # v = (1-spike)*v + spike*residual*v
            binary_spike = (spike_for_reset > 0).float()
            residual = 0.05  # Small residual for gradient flow
            self.v = (1.0 - binary_spike) * self.v + binary_spike * (residual * self.v)
        
        return self.v
    
    # This is the key method required by SpikinJelly's BaseNode
    def single_step_forward(self, x):
        """
        Single step forward for compatibility with spikingjelly
        """
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike
    
    def multi_step_forward(self, x_seq):
        """
        Process a sequence of inputs through the neuron.
        
        Args:
            x_seq: Input sequence tensor with shape [T, B, ...] 
                  where T is sequence length, B is batch size
                   
        Returns:
            Sequence of spike outputs with same shape as input
        """
        return super().multi_step_forward(x_seq)
    
    def extra_repr(self):
        leak_val = torch.sigmoid(self.leak).item() if isinstance(self.leak, torch.Tensor) else self.leak
        return f'threshold={self.threshold.item():.4f}, leakage={leak_val:.4f}, ' \
               f'num_thresholds={self.num_thresholds}, memory_factor={self.memory_factor}, ' \
               f'reset_mode={self.reset_mode}'
                
    def reset(self):
        """Reset all state variables in spikingjelly-compatible way"""
        super().reset()
        self.v = None
        self.memory = None
        self.spike_counter = None
        self.first_forward = True

