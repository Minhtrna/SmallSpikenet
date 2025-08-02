"""
This file contains utilities for training and benchmarking SmallSpikenet models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
import numpy as np
import psutil
import pynvml
pynvml.nvmlInit()

# Add wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ========================================
# ENERGY MONITORING
# ========================================

class EnergyMonitor:
    """
    Monitor GPU energy consumption during training and inference with improved accuracy
    """
    def __init__(self, device):
        self.device = device
        self.gpu_available = device.type == 'cuda'
        self.gpu_handle = None
        
        if self.gpu_available:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_max_power = pynvml.nvmlDeviceGetPowerManagementLimit(self.gpu_handle) / 1000.0
                print(f"GPU max power: {self.gpu_max_power:.2f}W")
            except Exception as e:
                print(f"Warning: Could not initialize GPU power monitoring: {e}")
                self.gpu_available = False
        else:
            print("Warning: GPU not available for power monitoring")
        
        # Energy tracking variables
        self.reset_measurements()
    
    def reset_measurements(self):
        """Reset all energy measurements"""
        self.total_gpu_energy = 0.0  # mJ
        self.measurement_count = 0
        self.start_time = None
        self.end_time = None
        self.baseline_power = 0.0  # NEW: Track baseline power
        self.power_samples = []    # NEW: Store power samples
    
    def start_measurement(self):
        """Start energy measurement with baseline power measurement"""
        self.reset_measurements()
        # Measure baseline power before starting
        self.measure_baseline_power()
        self.start_time = time.time()
    
    def measure_gpu_power(self):
        """Measure instantaneous GPU power consumption"""
        gpu_power = 0.0
        
        if self.gpu_available and self.gpu_handle:
            try:
                gpu_power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                gpu_power = gpu_power_mw / 1000.0  # Convert mW to W
            except Exception as e:
                # Fallback: estimate based on GPU utilization
                try:
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    gpu_power = self.gpu_max_power * (gpu_util.gpu / 100.0)
                except:
                    print(f"Warning: Could not measure GPU power: {e}")
                    gpu_power = 0.0
        
        return gpu_power
    
    def measure_baseline_power(self, num_samples=5):
        """Measure baseline GPU power consumption when idle"""
        if not self.gpu_available:
            self.baseline_power = 0.0
            return 0.0
        
        baseline_samples = []
        for _ in range(num_samples):
            power = self.measure_gpu_power()
            baseline_samples.append(power)
            time.sleep(0.01)  # 10ms between samples
        
        self.baseline_power = np.mean(baseline_samples) if baseline_samples else 0.0
        print(f"Baseline GPU power: {self.baseline_power:.1f}W")
        return self.baseline_power
    
    def add_measurement_improved(self, duration_ms=100):
        """
        Improved energy measurement with multiple power samples and baseline correction
        """
        if not self.gpu_available:
            return 0.0
        
        duration_s = duration_ms / 1000.0
        
        # Take multiple power samples right after inference
        power_samples = []
        sample_start = time.time()
        
        # Sample for a short period after inference (max 10ms to avoid overhead)
        sample_duration = min(0.01, duration_s * 0.1)  # 10ms or 10% of inference time
        
        while (time.time() - sample_start) < sample_duration:
            power = self.measure_gpu_power()
            power_samples.append(power)
            time.sleep(0.001)  # 1ms between samples
        
        if len(power_samples) > 0:
            # Calculate statistics
            avg_power = np.mean(power_samples)
            max_power = np.max(power_samples)
            min_power = np.min(power_samples)
            
            # Store samples for analysis
            self.power_samples.extend(power_samples)
            
            # Calculate net inference power (subtract baseline)
            net_inference_power = max(0, avg_power - self.baseline_power)
            
            # Calculate energy using net inference power
            gpu_energy_mj = net_inference_power * duration_s * 1000  # W * s * 1000 = mJ
            
            print(f"Power: Baseline={self.baseline_power:.1f}W, "
                  f"Avg={avg_power:.1f}W, Range={min_power:.1f}-{max_power:.1f}W, "
                  f"Net={net_inference_power:.1f}W, Samples={len(power_samples)}")
        else:
            gpu_energy_mj = 0.0
        
        self.total_gpu_energy += gpu_energy_mj
        self.measurement_count += 1
        
        return gpu_energy_mj
    
    def add_measurement(self, duration_ms=100):
        """Add energy measurement - now uses improved method"""
        return self.add_measurement_improved(duration_ms)
    
    def end_measurement(self):
        """End energy measurement and return results with detailed statistics"""
        self.end_time = time.time()
        total_time = self.end_time - self.start_time if self.start_time else 0.0
        
        # Calculate power statistics
        if self.power_samples:
            avg_measured_power = np.mean(self.power_samples)
            max_measured_power = np.max(self.power_samples)
            min_measured_power = np.min(self.power_samples)
            std_measured_power = np.std(self.power_samples)
            net_avg_power = max(0, avg_measured_power - self.baseline_power)
        else:
            avg_measured_power = max_measured_power = min_measured_power = std_measured_power = net_avg_power = 0.0
        
        results = {
            'gpu_energy_mj': self.total_gpu_energy,
            'total_energy_mj': self.total_gpu_energy,  # Only GPU energy
            'measurement_time_s': total_time,
            'baseline_power_w': self.baseline_power,
            'avg_measured_power_w': avg_measured_power,
            'max_measured_power_w': max_measured_power,
            'min_measured_power_w': min_measured_power,
            'std_measured_power_w': std_measured_power,
            'net_inference_power_w': net_avg_power,
            'power_samples_count': len(self.power_samples),
            'measurement_count': self.measurement_count
        }
        
        return results

# ========================================
# FLOPS COUNTER
# ========================================

class FLOPsCounter:
    """
    Count FLOPs (Floating Point Operations) for the model based on known architecture
    """
    def __init__(self, model):
        self.model = model
        self.total_flops = 0
        self.total_params = 0
        # Get model configuration
        self.num_timesteps = getattr(model, 'num_timesteps', 2)
        self.width_mult = getattr(model, 'width_mult', 1.0)  # Default from config
    
    def count_conv2d_flops(self, input_shape, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        """Count FLOPs for Conv2d layer"""
        batch_size, in_channels, input_height, input_width = input_shape
        
        # Calculate output dimensions
        output_height = (input_height + 2 * padding - kernel_size) // stride + 1
        output_width = (input_width + 2 * padding - kernel_size) // stride + 1
        
        # Calculate FLOPs
        kernel_flops = (in_channels // groups) * kernel_size * kernel_size
        output_elements = batch_size * output_height * output_width * out_channels
        flops = output_elements * kernel_flops
        
        return flops, (batch_size, out_channels, output_height, output_width)
    
    def count_linear_flops(self, input_shape, out_features):
        """Count FLOPs for Linear layer"""
        batch_size = input_shape[0]
        in_features = input_shape[1]
        flops = batch_size * in_features * out_features
        return flops
    
    def count_batchnorm_flops(self, input_shape):
        """Count FLOPs for BatchNorm layer (normalization + scale + shift)"""
        batch_size, channels, height, width = input_shape
        # 4 operations per element: (x-mean)/std, scale, shift, variance calc
        flops = batch_size * channels * height * width * 4
        return flops
    
    def count_snn_neuron_flops(self, input_shape):
        """Count FLOPs for LM-HT-LIF neuron operations per timestep"""
        batch_size, channels, height, width = input_shape
        elements = batch_size * channels * height * width
        # LM-HT-LIF operations per timestep per element:
        # - Membrane potential update: 2 ops (leak + input)
        # - Multi-threshold comparison: 1 op
        # - Spike generation: 1 op  
        # - Reset mechanism: 1 op
        ops_per_element = 5
        flops = elements * ops_per_element
        return flops
    
    def count_convbnlif_flops(self, input_shape, out_channels, kernel_size=3, stride=1, groups=1):
        """Count FLOPs for ConvBNLIF block"""
        # Conv2d FLOPs
        conv_flops, output_shape = self.count_conv2d_flops(
            input_shape, out_channels, kernel_size, stride, kernel_size//2, groups
        )
        
        # BatchNorm FLOPs
        bn_flops = self.count_batchnorm_flops(output_shape)
        
        # LM-HT-LIF neuron FLOPs (per timestep)
        neuron_flops = self.count_snn_neuron_flops(output_shape)
        
        total_flops = conv_flops + bn_flops + neuron_flops
        return total_flops, output_shape
    
    def count_inverted_residual_flops(self, input_shape, inp, oup, stride=1, expand_ratio=6):
        """Count FLOPs for SimpleInvertedResidual block"""
        hidden_dim = int(round(inp * expand_ratio))
        total_flops = 0
        current_shape = input_shape
        
        # 1. Expansion: 1x1 ConvBNLIF inp -> hidden_dim
        expansion_flops, current_shape = self.count_convbnlif_flops(
            current_shape, hidden_dim, kernel_size=1, stride=1, groups=1
        )
        total_flops += expansion_flops
        
        # 2. Depthwise: 3x3 ConvBNLIF hidden_dim -> hidden_dim (groups=hidden_dim)
        depthwise_flops, current_shape = self.count_convbnlif_flops(
            current_shape, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim
        )
        total_flops += depthwise_flops
        
        # 3. Projection: 1x1 Conv + BN hidden_dim -> oup (no LIF)
        proj_conv_flops, current_shape = self.count_conv2d_flops(
            current_shape, oup, kernel_size=1, stride=1, padding=0, groups=1
        )
        proj_bn_flops = self.count_batchnorm_flops(current_shape)
        total_flops += proj_conv_flops + proj_bn_flops
        
        # 4. Residual connection (if applicable)
        if stride == 1 and inp == oup:
            # Element-wise addition
            batch_size, channels, height, width = current_shape
            total_flops += batch_size * channels * height * width
        
        return total_flops, current_shape
    
    def estimate_model_flops(self, input_shape=(1, 3, 32, 32)):
        """Calculate FLOPs based on known MiniMobileNetV2LIF architecture"""
        self.total_flops = 0
        current_shape = input_shape
        
        # Calculate channel dimensions based on width_mult
        input_channel = int(48 * self.width_mult)    # 72 channels
        mid_channel = int(96 * self.width_mult)      # 144 channels  
        mid_channel2 = int(128 * self.width_mult)    # 192 channels
        last_channel = int(1280 * self.width_mult)   # 1920 channels
        
        print(f"Model architecture FLOPs calculation:")
        print(f"Input shape: {current_shape}")
        print(f"Channels: {input_channel}, {mid_channel}, {mid_channel2}, {last_channel}")
        
        # Layer 1: ConvBNLIF(3, 72, stride=2)
        layer1_flops, current_shape = self.count_convbnlif_flops(
            current_shape, input_channel, kernel_size=3, stride=2
        )
        self.total_flops += layer1_flops
        print(f"Layer1 ConvBNLIF: {layer1_flops:,} FLOPs, shape: {current_shape}")
        
        # Layer 2: SimpleInvertedResidual(72, 144, stride=2, expand_ratio=6)
        layer2_flops, current_shape = self.count_inverted_residual_flops(
            current_shape, input_channel, mid_channel, stride=2, expand_ratio=6
        )
        self.total_flops += layer2_flops
        print(f"Layer2 InvRes: {layer2_flops:,} FLOPs, shape: {current_shape}")
        
        # Layer 3: SimpleInvertedResidual(144, 144, stride=1, expand_ratio=6)
        layer3_flops, current_shape = self.count_inverted_residual_flops(
            current_shape, mid_channel, mid_channel, stride=1, expand_ratio=6
        )
        self.total_flops += layer3_flops
        print(f"Layer3 InvRes: {layer3_flops:,} FLOPs, shape: {current_shape}")
        
        # Layer 4: SimpleInvertedResidual(144, 192, stride=2, expand_ratio=6)
        layer4_flops, current_shape = self.count_inverted_residual_flops(
            current_shape, mid_channel, mid_channel2, stride=2, expand_ratio=6
        )
        self.total_flops += layer4_flops
        print(f"Layer4 InvRes: {layer4_flops:,} FLOPs, shape: {current_shape}")
        
        # Layer 5: SimpleInvertedResidual(192, 192, stride=1, expand_ratio=6)
        layer5_flops, current_shape = self.count_inverted_residual_flops(
            current_shape, mid_channel2, mid_channel2, stride=1, expand_ratio=6
        )
        self.total_flops += layer5_flops
        print(f"Layer5 InvRes: {layer5_flops:,} FLOPs, shape: {current_shape}")
        
        # Layer 6: ConvBNLIF(192, 1920, kernel_size=1)
        layer6_flops, current_shape = self.count_convbnlif_flops(
            current_shape, last_channel, kernel_size=1, stride=1
        )
        self.total_flops += layer6_flops
        print(f"Layer6 ConvBNLIF: {layer6_flops:,} FLOPs, shape: {current_shape}")
        
        # Global Average Pooling (negligible FLOPs)
        gap_flops = current_shape[1] * current_shape[2] * current_shape[3]  # Sum operations
        self.total_flops += gap_flops
        
        # Classifier: Linear(1920, num_classes)
        num_classes = 10  # CIFAR-10
        classifier_flops = self.count_linear_flops((input_shape[0], last_channel), num_classes)
        self.total_flops += classifier_flops
        print(f"Classifier Linear: {classifier_flops:,} FLOPs")
        
        # Multiply by number of timesteps
        single_timestep_flops = self.total_flops
        self.total_flops *= self.num_timesteps
        
        print(f"\nSingle timestep FLOPs: {single_timestep_flops:,}")
        print(f"Total FLOPs (×{self.num_timesteps} timesteps): {self.total_flops:,}")
        
        return self.total_flops

# ========================================
# INFERENCE BENCHMARK
# ========================================

class InferenceBenchmark:
    """
    Benchmark inference time and energy consumption with operations counting
    """
    def __init__(self, model, device, energy_monitor):
        self.model = model
        self.device = device
        self.energy_monitor = energy_monitor
        self.flops_counter = FLOPsCounter(model)
        
        # Calculate FLOPs per forward pass
        self.flops_per_inference = self.flops_counter.estimate_model_flops()
        print(f"Estimated FLOPs per inference: {self.flops_per_inference:,}")
    
    def benchmark_inference(self, test_loader, num_batches=10):
        """
        Benchmark inference performance on test data with improved energy monitoring
        """
        print(f"\n=== Starting Inference Benchmark ({num_batches} batches) ===")
        
        self.model.eval()
        
        # Prepare benchmark data
        benchmark_batches = []
        batch_count = 0
        for images, labels in test_loader:
            if batch_count >= num_batches:
                break
            benchmark_batches.append((images.to(self.device), labels.to(self.device)))
            batch_count += 1
        
        # Warmup runs
        print("Performing warmup runs...")
        with torch.no_grad():
            for i in range(3):
                images, _ = benchmark_batches[0]
                _ = self.model(images)
        
        # Actual benchmark
        print("Starting actual benchmark...")
        inference_times = []
        energy_measurements = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(benchmark_batches):
                batch_size = images.size(0)
                
                # Start energy measurement (now includes baseline measurement)
                self.energy_monitor.start_measurement()
                
                # Measure inference time
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                start_time = time.perf_counter()
                
                # Forward pass
                outputs = self.model(images)
                
                # End timing
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                end_time = time.perf_counter()
                
                # Calculate inference time
                inference_time_ms = (end_time - start_time) * 1000
                
                # Add energy measurement for the actual inference duration
                gpu_energy = self.energy_monitor.add_measurement(inference_time_ms)
                energy_results = self.energy_monitor.end_measurement()
                
                # Calculate operations for this batch
                total_ops = self.flops_per_inference * batch_size
                
                # Store results
                inference_times.append(inference_time_ms)
                energy_measurements.append({
                    **energy_results,
                    'total_ops': total_ops,
                    'batch_size': batch_size
                })
                
                # Calculate accuracy for this batch
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                batch_acc = 100.0 * correct / labels.size(0)
                
                # Calculate energy per operation
                energy_per_op = energy_results['total_energy_mj'] / total_ops * 1e6  # Convert to nJ/Op
                
                # Enhanced batch output with power details
                print(f"Batch {batch_idx+1}/{num_batches}: "
                      f"Time: {inference_time_ms:.2f}ms, "
                      f"Energy: {energy_results['total_energy_mj']:.2f}mJ, "
                      f"Baseline: {energy_results.get('baseline_power_w', 0):.1f}W, "
                      f"Net Power: {energy_results.get('net_inference_power_w', 0):.1f}W, "
                      f"Energy/Op: {energy_per_op:.3f}nJ/Op, "
                      f"Acc: {batch_acc:.1f}%")

        # Calculate statistics with enhanced energy metrics
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        avg_total_energy = np.mean([e['total_energy_mj'] for e in energy_measurements])
        avg_gpu_energy = avg_total_energy  # Only GPU energy now
        total_ops_per_batch = np.mean([e['total_ops'] for e in energy_measurements])
        
        # Enhanced power statistics
        avg_baseline_power = np.mean([e.get('baseline_power_w', 0) for e in energy_measurements])
        avg_measured_power = np.mean([e.get('avg_measured_power_w', 0) for e in energy_measurements])
        avg_net_power = np.mean([e.get('net_inference_power_w', 0) for e in energy_measurements])
        avg_power_samples = np.mean([e.get('power_samples_count', 0) for e in energy_measurements])
        
        # Calculate per-sample metrics
        samples_per_batch = benchmark_batches[0][0].size(0)
        avg_time_per_sample = avg_inference_time / samples_per_batch
        avg_energy_per_sample = avg_total_energy / samples_per_batch
        
        # Calculate per-operation metrics
        avg_energy_per_op_mj = avg_total_energy / total_ops_per_batch
        avg_energy_per_op_nj = avg_energy_per_op_mj * 1e6  # Convert mJ to nJ
        ops_per_second = total_ops_per_batch * 1000 / avg_inference_time  # ops/sec
        
        benchmark_results = {
            # Timing metrics
            'avg_inference_time_ms': avg_inference_time,
            'std_inference_time_ms': std_inference_time,
            'avg_time_per_sample_ms': avg_time_per_sample,
            
            # Energy metrics
            'avg_total_energy_mj': avg_total_energy,
            'avg_gpu_energy_mj': avg_gpu_energy,
            'avg_energy_per_sample_mj': avg_energy_per_sample,
            'avg_energy_per_op_mj': avg_energy_per_op_mj,
            'avg_energy_per_op_nj': avg_energy_per_op_nj,
            
            # Enhanced power metrics
            'avg_baseline_power_w': avg_baseline_power,
            'avg_measured_power_w': avg_measured_power,
            'avg_net_inference_power_w': avg_net_power,
            'avg_power_samples_count': avg_power_samples,
            
            # Operations metrics
            'flops_per_inference': self.flops_per_inference,
            'total_ops_per_batch': total_ops_per_batch,
            'ops_per_second': ops_per_second,
            
            # Efficiency metrics
            'throughput_samples_per_sec': (samples_per_batch * 1000) / avg_inference_time,
            'energy_efficiency_samples_per_mj': samples_per_batch / avg_total_energy,
            'energy_efficiency_ops_per_mj': total_ops_per_batch / avg_total_energy,
            
            # Metadata
            'num_batches': num_batches,
            'batch_size': samples_per_batch
        }
        
        return benchmark_results

    def print_benchmark_results(self, results):
        """Print detailed benchmark results with enhanced energy metrics"""
        print(f"\n=== Inference Benchmark Results ===")
        print(f"Number of batches:           {results['num_batches']}")
        print(f"Batch size:                  {results['batch_size']}")
        print(f"FLOPs per inference:         {results['flops_per_inference']:,}")
        print(f"")
        print(f"Timing Performance:")
        print(f"  Avg inference time:        {results['avg_inference_time_ms']:.2f} ± {results['std_inference_time_ms']:.2f} ms")
        print(f"  Avg time per sample:       {results['avg_time_per_sample_ms']:.3f} ms")
        print(f"  Throughput:                {results['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"  Operations throughput:     {results['ops_per_second']:.0f} ops/sec")
        print(f"")
        print(f"Power Analysis:")
        print(f"  Avg baseline power:        {results.get('avg_baseline_power_w', 0):.1f}W")
        print(f"  Avg measured power:        {results.get('avg_measured_power_w', 0):.1f}W")
        print(f"  Avg net inference power:   {results.get('avg_net_inference_power_w', 0):.1f}W")
        print(f"  Avg power samples/batch:   {results.get('avg_power_samples_count', 0):.1f}")
        print(f"")
        print(f"Energy Consumption:")
        print(f"  Avg total energy:          {results['avg_total_energy_mj']:.2f} mJ")
        print(f"  Avg energy per sample:     {results['avg_energy_per_sample_mj']:.3f} mJ")
        print(f"  Avg energy per operation:  {results['avg_energy_per_op_nj']:.3f} nJ/Op")
        print(f"")
        print(f"Efficiency Metrics:")
        print(f"  Energy efficiency:         {results['energy_efficiency_samples_per_mj']:.2f} samples/mJ")
        print(f"  Operations efficiency:     {results['energy_efficiency_ops_per_mj']:.0f} ops/mJ")
        print(f"  Computational intensity:   {results['flops_per_inference'] / (results['avg_inference_time_ms'] / 1000):.0f} ops/sec")
        print(f"=====================================\n")

# ========================================
# TRAINING AND TESTING FUNCTIONS
# ========================================

def test_model(model, test_loader, device):
    """Test model accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def setup_optimizer(model, config):
    """Setup optimizer with different learning rates for neuron parameters"""
    # Create parameter groups for different learning rates
    neuron_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'neuron.threshold' in name or 'neuron.leak' in name:
            neuron_params.append(param)
        else:
            other_params.append(param)
    
    # Setup optimizer with different learning rates
    optimizer = optim.Adam([
        {'params': other_params},
        {'params': neuron_params, 'lr': config['LEARNING_RATE'] * config['NEURON_LR_MULTIPLIER']}
    ], lr=config['LEARNING_RATE'], weight_decay=config['WEIGHT_DECAY'])
    
    return optimizer

def setup_scheduler(optimizer, config):
    """Setup learning rate scheduler"""
    if config['LR_SCHEDULER'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['NUM_EPOCHS'])
    elif config['LR_SCHEDULER'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['LR_STEP_SIZE'], gamma=config['LR_GAMMA'])
    elif config['LR_SCHEDULER'] == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['LR_GAMMA'])
    else:
        scheduler = None
    
    return scheduler

def train_model(model, train_loader, test_loader, config, device):
    """
    Complete training pipeline
    
    Args:
        model: The model to train
        train_loader: Training data loader
        test_loader: Test data loader
        config: Configuration dictionary
        device: Training device
        
    Returns:
        dict: Training results and metrics
    """
    print("Setting up training...")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer and scheduler
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config)
    
    # Initialize energy monitor
    energy_monitor = EnergyMonitor(device)
    
    # Training tracking
    train_losses = []
    train_accs = []
    best_acc = 0
    
    # Create results file
    results_file = config['RESULTS_FILE']
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(config['NUM_EPOCHS']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['NUM_EPOCHS']}", leave=True)
        
        for i, (images, labels) in enumerate(progress_bar):
            # Move to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['GRAD_CLIP_NORM'])
            
            # Update weights
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{running_loss/(i+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        
        # Store statistics
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{config['NUM_EPOCHS']} - "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Train Acc: {epoch_acc:.2f}%")
        
        # Save model checkpoint
        if (epoch + 1) % config['SAVE_CHECKPOINT_EVERY'] == 0 or epoch == config['NUM_EPOCHS'] - 1:
            checkpoint_path = f"{config['SAVE_DIR']}/{config['MODEL_NAME']}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint at epoch {epoch+1}")
        
        # Save results to CSV
        with open(results_file, 'a') as f:
            if epoch == 0:
                f.write('epoch,train_loss,train_acc,test_acc,inference_time_ms,total_energy_mj,energy_per_sample_mj,energy_per_op_nj,flops_per_inference\n')
            f.write(f'{epoch+1},{epoch_loss:.6f},{epoch_acc:.6f},,,,,\n')
    
    # Training completed
    elapsed_time = time.time() - start_time
    print(f"\n=== Training Completed in {elapsed_time/60:.2f} minutes ===")
    
    # Final test
    print("Performing final test...")
    final_test_acc = test_model(model, test_loader, device)
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    
    # Final benchmark
    print("Performing final inference benchmark...")
    inference_benchmark = InferenceBenchmark(model, device, energy_monitor)
    benchmark_results = inference_benchmark.benchmark_inference(test_loader, num_batches=config['BENCHMARK_BATCHES'])
    inference_benchmark.print_benchmark_results(benchmark_results)
    
    # Save final model
    if config['SAVE_FINAL_MODEL']:
        final_path = f"{config['SAVE_DIR']}/{config['MODEL_NAME']}_final.pth"
        torch.save(model.state_dict(), final_path)
        print(f"Saved final model to: {final_path}")
    
    # Update CSV with final results
    with open(results_file, 'a') as f:
        f.write(f'final_test,,,{final_test_acc:.6f},{benchmark_results["avg_inference_time_ms"]:.3f},{benchmark_results["avg_total_energy_mj"]:.3f},{benchmark_results["avg_energy_per_sample_mj"]:.6f},{benchmark_results["avg_energy_per_op_nj"]:.3f},{benchmark_results["flops_per_inference"]}\n')
    
    # Create training results summary
    training_results = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'final_test_acc': final_test_acc,
        'training_time_minutes': elapsed_time / 60,
        'benchmark_results': benchmark_results,
        'results_file': results_file
    }
    
    # Print training summary
    print("\n=== Training Summary ===")
    print(f"Total training time: {training_results['training_time_minutes']:.2f} minutes")
    print(f"Final test accuracy: {final_test_acc:.2f}%")
    print(f"Avg inference time: {benchmark_results['avg_inference_time_ms']:.2f}ms")
    print(f"Avg energy per sample: {benchmark_results['avg_energy_per_sample_mj']:.3f}mJ")
    print(f"Avg energy per operation: {benchmark_results['avg_energy_per_op_nj']:.2f}nJ/Op")
    print(f"Energy efficiency: {benchmark_results['energy_efficiency_ops_per_mj']:.0f} ops/mJ")
    print(f"FLOPs per inference: {benchmark_results['flops_per_inference']:,}")
    print(f"Results saved to: {results_file}")
    print(f"Model checkpoints saved to: {config['SAVE_DIR']}")
    print("========================\n")
    
    return training_results

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ========================================
# WANDB LOGGING UTILITIES
# ========================================

class WandbLogger:
    """Lightweight wandb logging wrapper that can be enabled/disabled"""
    
    def __init__(self, enabled=False, config=None):
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None
        
        if enabled and not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Logging disabled.")
            self.enabled = False
        
        if self.enabled and config:
            self.init_wandb(config)
    
    def init_wandb(self, config):
        """Initialize wandb if enabled"""
        if not self.enabled:
            return
        
        run_name = config.get('WANDB_RUN_NAME') or f"SmallSpikenet_{config['DATASET']}_T{config['NUM_TIMESTEPS']}_W{config['WIDTH_MULT']}"
        
        self.run = wandb.init(
            project=config.get('WANDB_PROJECT', 'SmallSpikenet'),
            entity=config.get('WANDB_ENTITY', None),
            name=run_name,
            config=config,
            tags=[f"dataset_{config['DATASET']}", f"timesteps_{config['NUM_TIMESTEPS']}", "SNN"]
        )
        
        if self.run:
            print(f"Wandb initialized: {self.run.url}")
    
    def watch_model(self, model):
        """Watch model for gradients"""
        if self.enabled and self.run:
            wandb.watch(model, log="all", log_freq=100)
    
    def log(self, data):
        """Log data to wandb"""
        if self.enabled and self.run:
            wandb.log(data)
    
    def log_artifact(self, filepath, name, type="model", description=""):
        """Log artifact to wandb"""
        if self.enabled and self.run:
            try:
                artifact = wandb.Artifact(name=name, type=type, description=description)
                artifact.add_file(filepath)
                wandb.log_artifact(artifact)
            except Exception as e:
                print(f"Warning: Could not log artifact {name}: {e}")
    
    def finish(self):
        """Finish wandb run"""
        if self.enabled and self.run:
            wandb.finish()

# ========================================
# WANDB HELPER FUNCTIONS
# ========================================

def create_wandb_config(
    dataset, num_epochs, batch_size, learning_rate, weight_decay,
    num_timesteps, width_mult, init_threshold, dropout_rate, 
    num_thresholds, leakage, memory_factor, reset_mode,
    lr_scheduler, grad_clip_norm, use_autoaugment, device,
    wandb_project='SmallSpikenet', wandb_entity=None, wandb_run_name=None
):
    """Create wandb configuration dictionary"""
    return {
        'DATASET': dataset,
        'NUM_EPOCHS': num_epochs,
        'BATCH_SIZE': batch_size,
        'LEARNING_RATE': learning_rate,
        'WEIGHT_DECAY': weight_decay,
        'NUM_TIMESTEPS': num_timesteps,
        'WIDTH_MULT': width_mult,
        'INIT_THRESHOLD': init_threshold,
        'DROPOUT_RATE': dropout_rate,
        'NUM_THRESHOLDS': num_thresholds,
        'LEAKAGE': leakage,
        'MEMORY_FACTOR': memory_factor,
        'RESET_MODE': reset_mode,
        'LR_SCHEDULER': lr_scheduler,
        'GRAD_CLIP_NORM': grad_clip_norm,
        'USE_AUTOAUGMENT': use_autoaugment,
        'DEVICE': str(device),
        'ARCHITECTURE': 'MiniMobileNetV2LIF',
        'WANDB_PROJECT': wandb_project,
        'WANDB_ENTITY': wandb_entity,
        'WANDB_RUN_NAME': wandb_run_name
    }

def train_epoch_with_logging(model, train_loader, optimizer, criterion, epoch, num_epochs, 
                           grad_clip_norm, device, wandb_logger=None):
    """Train one epoch with optional wandb logging"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        
        # Update weights
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Log batch metrics every 100 batches
        if wandb_logger and batch_idx % 100 == 0:
            wandb_logger.log({
                'batch_loss': loss.item(),
                'batch_acc': 100.0 * correct / total,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch_progress': epoch + batch_idx / len(train_loader)
            })
        
        # Update progress bar
        if batch_idx % 10 == 0:
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc

def save_checkpoint_with_logging(model, epoch, loss, acc, model_config, save_dir, wandb_logger=None):
    """Save checkpoint with optional wandb logging"""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'accuracy': acc,
        'config': model_config
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Log to wandb if enabled
    if wandb_logger:
        wandb_logger.log_artifact(
            checkpoint_path, 
            name=f"model_epoch_{epoch+1}",
            type="model",
            description=f"Model checkpoint at epoch {epoch+1}"
        )