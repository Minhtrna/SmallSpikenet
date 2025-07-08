import torch
import torch.nn as nn
from Neurons import LM_HT_LIF
from Neurons.neurons import LM_HT_LIF

class ConvBNLIF(nn.Module):
    """
    Conv + BatchNorm + LM-HT-LIF block using v1 neuron
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, 
                 init_threshold=0.5, leakage=0.2, num_thresholds=4, memory_factor=0.1, reset_mode='hybrid'):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # LM-HT-LIF neuron with configurable parameters
        self.neuron = LM_HT_LIF(
            init_threshold=init_threshold,
            init_leakage=leakage,
            num_thresholds=num_thresholds,
            memory_factor=memory_factor,
            reset_mode=reset_mode
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.neuron(x)
        return x
    
    def reset_neurons(self):
        """Reset all LIF neurons in this block"""
        self.neuron.reset()


class SimpleInvertedResidual(nn.Module):
    """
    Simplified Inverted Residual Block for smaller networks
    """
    def __init__(self, inp, oup, stride=1, expand_ratio=6, 
                 init_threshold=0.5, leakage=0.2, num_thresholds=4, memory_factor=0.1, reset_mode='hybrid'):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup

        # Reduced complexity: just one expansion + depthwise + projection
        hidden_dim = int(round(inp * expand_ratio))
        
        # Simplified structure with consistent threshold initialization
        self.conv = nn.Sequential(
            # Expansion
            ConvBNLIF(inp, hidden_dim, kernel_size=1, 
                     init_threshold=init_threshold, leakage=leakage, 
                     num_thresholds=num_thresholds, memory_factor=memory_factor, reset_mode=reset_mode),
            # Depthwise
            ConvBNLIF(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim,
                     init_threshold=init_threshold, leakage=leakage,
                     num_thresholds=num_thresholds, memory_factor=memory_factor, reset_mode=reset_mode),
            # Projection - no activation
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
    
    def reset_neurons(self):
        """Reset all LIF neurons in this block"""
        for module in self.conv:
            if hasattr(module, 'reset_neurons'):
                module.reset_neurons()