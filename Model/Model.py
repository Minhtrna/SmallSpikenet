import torch
import torch.nn as nn
from Blocks import ConvBNLIF, SimpleInvertedResidual

# ========================================
# MODEL DEFINITION
# ========================================

class MiniMobileNetV2LIF(nn.Module):
    """
    Scaled-up MobileNetV2 with LM-HT-LIF v1 neurons, targeting ~1M parameters
    """
    def __init__(self, config):
        super().__init__()
        self.num_timesteps = config['NUM_TIMESTEPS']
        self.input_scale = config['INPUT_SCALE']
        
        # Scaled up parameters
        input_channel = int(48 * config['WIDTH_MULT'])    # 72 channels
        mid_channel = int(96 * config['WIDTH_MULT'])      # 144 channels  
        mid_channel2 = int(128 * config['WIDTH_MULT'])    # 192 channels
        last_channel = int(1280 * config['WIDTH_MULT'])   # 1920 channels
        
        # Store neuron parameters for passing to blocks
        neuron_params = {
            'init_threshold': config['INIT_THRESHOLD'],
            'leakage': config['LEAKAGE'],
            'num_thresholds': config['NUM_THRESHOLDS'],
            'memory_factor': config['MEMORY_FACTOR'],
            'reset_mode': config['RESET_MODE']
        }
        
        # Build network layers
        self.layer1 = ConvBNLIF(3, input_channel, stride=2, **neuron_params)
        
        self.layer2 = SimpleInvertedResidual(input_channel, mid_channel, stride=2, expand_ratio=6, **neuron_params)
        self.layer3 = SimpleInvertedResidual(mid_channel, mid_channel, stride=1, expand_ratio=6, **neuron_params)
        self.layer4 = SimpleInvertedResidual(mid_channel, mid_channel2, stride=2, expand_ratio=6, **neuron_params)
        self.layer5 = SimpleInvertedResidual(mid_channel2, mid_channel2, stride=1, expand_ratio=6, **neuron_params)
        
        self.layer6 = ConvBNLIF(mid_channel2, last_channel, kernel_size=1, **neuron_params)
        
        # Classifier with configurable dropout
        self.dropout = nn.Dropout(config['DROPOUT_RATE'])
        self.classifier = nn.Linear(last_channel, config['NUM_CLASSES'])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                # Smaller momentum for more stable statistics
                m.momentum = 0.05
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Reset all neurons before forward pass
        self.reset_neurons()
        
        # Apply input scaling
        x = x / self.input_scale
        
        # Process for multiple timesteps
        outputs = []
        for t in range(self.num_timesteps):
            # Pass through layers
            out = self.layer1(x)            # Conv+BN+LIF
            out = self.layer2(out)          # Inverted Residual Block 1
            out = self.layer3(out)          # Inverted Residual Block 2  
            out = self.layer4(out)          # Inverted Residual Block 3
            out = self.layer5(out)          # Inverted Residual Block 4
            out = self.layer6(out)          # Final Conv+BN+LIF
            
            # Global average pooling
            out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = self.dropout(out)
            
            # Classification
            out = self.classifier(out)
            
            outputs.append(out)
        
        # Average outputs across timesteps
        return torch.stack(outputs).mean(0)
    
    def reset_neurons(self):
        """Reset all LIF neurons in the network"""
        self.layer1.reset_neurons()
        self.layer2.reset_neurons()
        self.layer3.reset_neurons()
        self.layer4.reset_neurons()
        self.layer5.reset_neurons()
        self.layer6.reset_neurons()

def create_model(config):
    """Create model with configuration"""
    return MiniMobileNetV2LIF(config)

