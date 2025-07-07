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
# Add energy measurement imports

# ========================================
# ENERGY MONITORING
# ========================================

class EnergyMonitor:
    """
    Monitor energy consumption during training and inference
    """
    def __init__(self, device):
        self.device = device
        self.gpu_available = device.type == 'cuda' and NVML_AVAILABLE
        self.cpu_tdp = self._get_cpu_tdp()  # Thermal Design Power
        self.gpu_handle = None
        
        if self.gpu_available:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                # Fix: use the correct function name
                self.gpu_max_power = pynvml.nvmlDeviceGetPowerManagementLimit(self.gpu_handle) / 1000.0  # Convert to watts
                print(f"GPU max power: {self.gpu_max_power:.2f}W")
            except Exception as e:
                print(f"Warning: Could not initialize GPU power monitoring: {e}")
                self.gpu_available = False
        
        print(f"CPU TDP: {self.cpu_tdp}W")
        
        # Energy tracking variables
        self.reset_measurements()
    
    def _get_cpu_tdp(self):
        """Estimate CPU TDP (Thermal Design Power) in watts"""
        try:
            # Try to get CPU info from Windows
            if os.name == 'nt':
                import wmi
                c = wmi.WMI()
                for processor in c.Win32_Processor():
                    # Rough estimate based on CPU cores and frequency
                    cores = processor.NumberOfCores
                    freq = float(processor.MaxClockSpeed) / 1000.0  # Convert MHz to GHz
                    # Rough TDP estimation: cores * frequency * scaling factor
                    estimated_tdp = cores * freq * 15  # Rough scaling factor
                    return min(estimated_tdp, 150)  # Cap at 150W
        except:
            pass
        
        # Fallback estimation based on CPU count
        cpu_count = psutil.cpu_count()
        if cpu_count <= 4:
            return 65.0  # Low-power CPU
        elif cpu_count <= 8:
            return 95.0  # Mid-range CPU
        else:
            return 125.0  # High-end CPU
    
    def reset_measurements(self):
        """Reset all energy measurements"""
        self.total_cpu_energy = 0.0  # mJ
        self.total_gpu_energy = 0.0  # mJ
        self.measurement_count = 0
        self.start_time = None
        self.end_time = None
    
    def start_measurement(self):
        """Start energy measurement"""
        self.reset_measurements()
        self.start_time = time.time()
    
    def measure_instantaneous_power(self):
        """Measure instantaneous power consumption"""
        cpu_power = 0.0
        gpu_power = 0.0
        
        # CPU power estimation based on utilization
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_power = self.cpu_tdp * (cpu_percent / 100.0)
        
        # GPU power measurement
        if self.gpu_available and self.gpu_handle:
            try:
                gpu_power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                gpu_power = gpu_power_mw / 1000.0  # Convert mW to W
            except:
                # Fallback: estimate based on GPU utilization
                try:
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    gpu_power = self.gpu_max_power * (gpu_util.gpu / 100.0)
                except:
                    gpu_power = 0.0
        
        return cpu_power, gpu_power
    
    def add_measurement(self, duration_ms=100):
        """Add a power measurement over specified duration"""
        cpu_power, gpu_power = self.measure_instantaneous_power()
        
        # Convert power (W) to energy (mJ) over the measurement duration
        duration_s = duration_ms / 1000.0
        cpu_energy_mj = cpu_power * duration_s * 1000  # W * s * 1000 = mJ
        gpu_energy_mj = gpu_power * duration_s * 1000  # W * s * 1000 = mJ
        
        self.total_cpu_energy += cpu_energy_mj
        self.total_gpu_energy += gpu_energy_mj
        self.measurement_count += 1
        
        return cpu_energy_mj, gpu_energy_mj
    
    def end_measurement(self):
        """End energy measurement and return results"""
        self.end_time = time.time()
        total_time = self.end_time - self.start_time if self.start_time else 0.0
        
        results = {
            'cpu_energy_mj': self.total_cpu_energy,
            'gpu_energy_mj': self.total_gpu_energy,
            'total_energy_mj': self.total_cpu_energy + self.total_gpu_energy,
            'measurement_time_s': total_time,
            'avg_cpu_power_w': self.total_cpu_energy / (total_time * 1000) if total_time > 0 else 0.0,
            'avg_gpu_power_w': self.total_gpu_energy / (total_time * 1000) if total_time > 0 else 0.0,
            'measurement_count': self.measurement_count
        }
        
        return results

# ========================================
# FLOPS COUNTER
# ========================================

class FLOPsCounter:
    """
    Count FLOPs (Floating Point Operations) for the model
    """
    def __init__(self, model):
        self.model = model
        self.total_flops = 0
        self.total_params = 0
    
    def count_conv2d_flops(self, input_shape, weight_shape, stride=1, padding=0, groups=1):
        """Count FLOPs for Conv2d layer"""
        batch_size, in_channels, input_height, input_width = input_shape
        out_channels, in_channels_per_group, kernel_height, kernel_width = weight_shape
        
        output_height = (input_height + 2 * padding - kernel_height) // stride + 1
        output_width = (input_width + 2 * padding - kernel_width) // stride + 1
        
        # FLOPs = batch_size * output_height * output_width * out_channels * 
        #         (in_channels_per_group * kernel_height * kernel_width + 1) # +1 for bias if present
        kernel_flops = in_channels_per_group * kernel_height * kernel_width
        output_elements = batch_size * output_height * output_width * out_channels
        flops = output_elements * kernel_flops
        
        return flops, (batch_size, out_channels, output_height, output_width)
    
    def count_linear_flops(self, input_shape, weight_shape):
        """Count FLOPs for Linear layer"""
        batch_size = input_shape[0]
        in_features, out_features = weight_shape
        flops = batch_size * in_features * out_features
        return flops, (batch_size, out_features)
    
    def estimate_model_flops(self, input_shape=(1, 3, 32, 32)):
        """Estimate total FLOPs for one forward pass"""
        self.total_flops = 0
        current_shape = input_shape
        
        # Track through model layers
        # Layer 1: ConvBNLIF(3, 72, stride=2)
        flops, current_shape = self.count_conv2d_flops(current_shape, (72, 3, 3, 3), stride=2, padding=1)
        self.total_flops += flops
        
        # Layer 2: SimpleInvertedResidual(72, 144, stride=2, expand_ratio=6)
        # Expansion: 1x1 conv 72 -> 432
        flops, temp_shape = self.count_conv2d_flops(current_shape, (432, 72, 1, 1))
        self.total_flops += flops
        # Depthwise: 3x3 depthwise conv 432 -> 432 with stride=2
        flops, temp_shape = self.count_conv2d_flops(temp_shape, (432, 1, 3, 3), stride=2, padding=1, groups=432)
        self.total_flops += flops
        # Projection: 1x1 conv 432 -> 144
        flops, current_shape = self.count_conv2d_flops(temp_shape, (144, 432, 1, 1))
        self.total_flops += flops
        
        # Layer 3: SimpleInvertedResidual(144, 144, stride=1, expand_ratio=6)
        # Expansion: 1x1 conv 144 -> 864
        flops, temp_shape = self.count_conv2d_flops(current_shape, (864, 144, 1, 1))
        self.total_flops += flops
        # Depthwise: 3x3 depthwise conv 864 -> 864
        flops, temp_shape = self.count_conv2d_flops(temp_shape, (864, 1, 3, 3), padding=1, groups=864)
        self.total_flops += flops
        # Projection: 1x1 conv 864 -> 144
        flops, current_shape = self.count_conv2d_flops(temp_shape, (144, 864, 1, 1))
        self.total_flops += flops
        
        # Layer 4: SimpleInvertedResidual(144, 192, stride=2, expand_ratio=6)
        # Expansion: 1x1 conv 144 -> 864
        flops, temp_shape = self.count_conv2d_flops(current_shape, (864, 144, 1, 1))
        self.total_flops += flops
        # Depthwise: 3x3 depthwise conv 864 -> 864 with stride=2
        flops, temp_shape = self.count_conv2d_flops(temp_shape, (864, 1, 3, 3), stride=2, padding=1, groups=864)
        self.total_flops += flops
        # Projection: 1x1 conv 864 -> 192
        flops, current_shape = self.count_conv2d_flops(temp_shape, (192, 864, 1, 1))
        self.total_flops += flops
        
        # Layer 5: SimpleInvertedResidual(192, 192, stride=1, expand_ratio=6)
        # Expansion: 1x1 conv 192 -> 1152
        flops, temp_shape = self.count_conv2d_flops(current_shape, (1152, 192, 1, 1))
        self.total_flops += flops
        # Depthwise: 3x3 depthwise conv 1152 -> 1152
        flops, temp_shape = self.count_conv2d_flops(temp_shape, (1152, 1, 3, 3), padding=1, groups=1152)
        self.total_flops += flops
        # Projection: 1x1 conv 1152 -> 192
        flops, current_shape = self.count_conv2d_flops(temp_shape, (192, 1152, 1, 1))
        self.total_flops += flops
        
        # Layer 6: ConvBNLIF(192, 1920, kernel_size=1)
        flops, current_shape = self.count_conv2d_flops(current_shape, (1920, 192, 1, 1))
        self.total_flops += flops
        
        # Global Average Pooling (minimal FLOPs)
        # Classifier: Linear(1920, 10)
        # Flatten to (batch_size, 1920)
        flat_shape = (input_shape[0], 1920)
        flops, _ = self.count_linear_flops(flat_shape, (1920, 10))
        self.total_flops += flops
        
        # Multiply by number of timesteps
        self.total_flops *= getattr(self.model, 'num_timesteps', 2)
        
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
        Benchmark inference performance on test data with operations counting
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
                
                # Start energy measurement
                self.energy_monitor.start_measurement()
                
                # Measure inference time
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                start_time = time.perf_counter()
                
                # Forward pass
                outputs = self.model(images)
                
                # End timing
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                end_time = time.perf_counter()
                
                # End energy measurement
                inference_time_ms = (end_time - start_time) * 1000
                
                # Add energy measurement for the actual inference duration
                cpu_energy, gpu_energy = self.energy_monitor.add_measurement(inference_time_ms)
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
                
                print(f"Batch {batch_idx+1}/{num_batches}: "
                      f"Time: {inference_time_ms:.2f}ms, "
                      f"Energy: {energy_results['total_energy_mj']:.2f}mJ, "
                      f"Energy/Op: {energy_per_op:.2f}nJ/Op, "
                      f"Acc: {batch_acc:.1f}%")
        
        # Calculate statistics
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        avg_total_energy = np.mean([e['total_energy_mj'] for e in energy_measurements])
        avg_cpu_energy = np.mean([e['cpu_energy_mj'] for e in energy_measurements])
        avg_gpu_energy = np.mean([e['gpu_energy_mj'] for e in energy_measurements])
        total_ops_per_batch = np.mean([e['total_ops'] for e in energy_measurements])
        
        # Calculate per-sample metrics
        samples_per_batch = benchmark_batches[0][0].size(0)
        avg_time_per_sample = avg_inference_time / samples_per_batch
        avg_energy_per_sample = avg_total_energy / samples_per_batch
        
        # Calculate per-operation metrics
        avg_energy_per_op_mj = avg_total_energy / total_ops_per_batch
        avg_energy_per_op_nj = avg_energy_per_op_mj * 1e6  # Convert mJ to nJ
        ops_per_second = total_ops_per_batch * 1000 / avg_inference_time  # ops/sec
        
        benchmark_results = {
            'avg_inference_time_ms': avg_inference_time,
            'std_inference_time_ms': std_inference_time,
            'avg_time_per_sample_ms': avg_time_per_sample,
            'avg_total_energy_mj': avg_total_energy,
            'avg_cpu_energy_mj': avg_cpu_energy,
            'avg_gpu_energy_mj': avg_gpu_energy,
            'avg_energy_per_sample_mj': avg_energy_per_sample,
            'avg_energy_per_op_mj': avg_energy_per_op_mj,
            'avg_energy_per_op_nj': avg_energy_per_op_nj,
            'flops_per_inference': self.flops_per_inference,
            'total_ops_per_batch': total_ops_per_batch,
            'ops_per_second': ops_per_second,
            'throughput_samples_per_sec': (samples_per_batch * 1000) / avg_inference_time,
            'energy_efficiency_samples_per_mj': samples_per_batch / avg_total_energy,
            'energy_efficiency_ops_per_mj': total_ops_per_batch / avg_total_energy,
            'num_batches': num_batches,
            'batch_size': samples_per_batch
        }
        
        return benchmark_results
    
    def print_benchmark_results(self, results):
        """Print detailed benchmark results with operations metrics"""
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
        print(f"Energy Consumption:")
        print(f"  Avg total energy:          {results['avg_total_energy_mj']:.2f} mJ")
        print(f"  Avg CPU energy:            {results['avg_cpu_energy_mj']:.2f} mJ")
        print(f"  Avg GPU energy:            {results['avg_gpu_energy_mj']:.2f} mJ")
        print(f"  Avg energy per sample:     {results['avg_energy_per_sample_mj']:.3f} mJ")
        print(f"  Avg energy per operation:  {results['avg_energy_per_op_nj']:.2f} nJ/Op")
        print(f"")
        print(f"Efficiency Metrics:")
        print(f"  Energy efficiency:         {results['energy_efficiency_samples_per_mj']:.2f} samples/mJ")
        print(f"  Operations efficiency:     {results['energy_efficiency_ops_per_mj']:.0f} ops/mJ")
        print(f"  Energy per operation:      {results['avg_energy_per_op_nj']:.2f} nJ/Op")
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