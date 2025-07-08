"""
Main execution script for training SmallSpikenet on CIFAR-10 or CIFAR-100 datasets.
This script handles dataset loading, model creation, training, and testing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys

# Add paths
from Model.Model import create_model
from utils.Tools import count_parameters, EnergyMonitor, InferenceBenchmark
from utils.datasets import create_dataloaders, validate_dataset_config

# ========================================
# CONFIGURATION - EDIT THESE VALUES
# ========================================

# === BASIC SETTINGS ===
DATASET = 'cifar10'         # 'cifar10' or 'cifar100'
NUM_EPOCHS = 150
BATCH_SIZE = 256
LEARNING_RATE = 0.001
DEVICE = 'auto'             # 'auto', 'cpu', 'cuda'

# === MODEL SETTINGS ===
NUM_TIMESTEPS = 2           # SNN timesteps 
WIDTH_MULT = 1.0           # Model size 
INIT_THRESHOLD = 1.0       # Neuron threshold 
DROPOUT_RATE = 0.2         # Dropout 

# === SNN NEURON SETTINGS ===
NUM_THRESHOLDS = 4         # Multi-threshold levels (2-8)
LEAKAGE = 0.2             # Membrane leakage (0.1-0.3)
MEMORY_FACTOR = 0.1       # Memory retention (0.05-0.2)
RESET_MODE = 'hybrid'     # 'soft', 'hard', 'hybrid'

# === TRAINING SETTINGS ===
WEIGHT_DECAY = 4e-5       # L2 regularization
GRAD_CLIP_NORM = 1.0      # Gradient clipping
LR_SCHEDULER = 'cosine'   # 'cosine', 'step', 'none'

# === DATA SETTINGS ===
USE_AUTOAUGMENT = False   # Enable AutoAugment
DATA_DIR = './data'       # Data directory
NUM_WORKERS = 2           # DataLoader workers
PIN_MEMORY = True         # Pin memory for faster GPU transfer

# === SAVE SETTINGS ===
SAVE_DIR = './checkpoints'
SAVE_EVERY = 10           # Save checkpoint every N epochs
SAVE_FINAL = True         # Save final model

# === DEBUG/TESTING ===
DEBUG_MODE = False        # Quick testing (5 epochs, small batch)

# ========================================
# AUTO CONFIGURATION
# ========================================

if DEBUG_MODE:
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    print("DEBUG MODE: Reduced epochs and batch size")

# Device setup
if DEVICE == 'auto':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(DEVICE)

print(f"SmallSpikenet Training")
print(f"Dataset: {DATASET.upper()}")
print(f"Device: {device}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Timesteps: {NUM_TIMESTEPS}")
print(f"Threshold: {INIT_THRESHOLD}")
print(f"AutoAugment: {USE_AUTOAUGMENT}")
print("=" * 50)

# ========================================
# CONFIGURATION CREATION
# ========================================

def create_dataset_config():
    """Create configuration for dataset factory"""
    config = {
        'DATASET': DATASET,
        'DATA_DIR': DATA_DIR,
        'BATCH_SIZE': BATCH_SIZE,
        'NUM_WORKERS': NUM_WORKERS,
        'PIN_MEMORY': PIN_MEMORY,
        'USE_AUTOAUGMENT': USE_AUTOAUGMENT,
        'DROP_LAST': False
    }
    return config

def create_model_config():
    """Create configuration for model"""
    config = {
        'NUM_TIMESTEPS': NUM_TIMESTEPS,
        'WIDTH_MULT': WIDTH_MULT,
        'INIT_THRESHOLD': INIT_THRESHOLD,
        'NUM_THRESHOLDS': NUM_THRESHOLDS,
        'LEAKAGE': LEAKAGE,
        'MEMORY_FACTOR': MEMORY_FACTOR,
        'RESET_MODE': RESET_MODE,
        'DROPOUT_RATE': DROPOUT_RATE,
        'INPUT_SCALE': 1.2
    }
    return config

# ========================================
# DATA LOADING
# ========================================

def create_data_loaders():
    """Create train and test dataloaders using dataset factory"""
    print("Creating data loaders...")
    
    # Create dataset configuration
    dataset_config = create_dataset_config()
    
    # Validate configuration
    validate_dataset_config(dataset_config)
    
    # Create dataloaders using dataset factory
    train_loader, test_loader, dataset_info = create_dataloaders(dataset_config)
    
    print(f"Dataset: {dataset_info['name'].upper()}")
    print(f"Classes: {dataset_info['num_classes']}")
    print(f"AutoAugment: {'Enabled' if dataset_info['use_autoaugment'] else 'Disabled'}")
    print(f"Mean: {dataset_info['mean']}")
    print(f"Std: {dataset_info['std']}")
    
    return train_loader, test_loader, dataset_info

# ========================================
# MODEL CREATION
# ========================================

def create_model_and_optimizer(dataset_info):
    """Create model and optimizer"""
    print("\nCreating model...")
    
    # Create model config with dataset info
    model_config = create_model_config()
    model_config['NUM_CLASSES'] = dataset_info['num_classes']
    
    # Create model
    model = create_model(model_config).to(device)
    
    param_count = count_parameters(model)
    print(f"Model created: {param_count:,} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Create scheduler
    if LR_SCHEDULER == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    elif LR_SCHEDULER == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None
    
    return model, optimizer, scheduler, model_config

# ========================================
# TRAINING FUNCTIONS
# ========================================

def test_model(model, test_loader):
    """Test model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy

def train_epoch(model, train_loader, optimizer, criterion, epoch):
    """Train one epoch"""
    from tqdm import tqdm
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        
        # Update weights
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        if batch_idx % 10 == 0:
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc

def save_checkpoint(model, epoch, loss, acc, model_config):
    """Save model checkpoint"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    checkpoint_path = os.path.join(SAVE_DIR, f'model_epoch_{epoch+1}.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'accuracy': acc,
        'config': model_config
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")

# ========================================
# MAIN TRAINING LOOP
# ========================================

def main():
    """Main training function"""
    
    # Create data loaders
    train_loader, test_loader, dataset_info = create_data_loaders()
    
    # Create model
    model, optimizer, scheduler, model_config = create_model_and_optimizer(dataset_info)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training tracking
    train_losses = []
    train_accs = []
    best_acc = 0.0
    
    print(f"\nStarting training...")
    print("=" * 50)
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        
        # Train one epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, epoch)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Store results
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Print results
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        # Save periodic checkpoint
        if (epoch + 1) % SAVE_EVERY == 0:
            save_checkpoint(model, epoch, train_loss, train_acc, model_config)
        
        print("-" * 30)
    
    # Training completed
    training_time = time.time() - start_time
    
    print(f"\nTraining completed!")
    print(f"Training time: {training_time/60:.2f} minutes")
    
    # After training completes, run comprehensive inference testing
    print("\nPerforming inference benchmark testing...")
    
    # Create energy monitor
    energy_monitor = EnergyMonitor(device)
    
    # Create inference benchmark
    inference_benchmark = InferenceBenchmark(model, device, energy_monitor)
    
    # Run benchmark (use 50 batches for more comprehensive results)
    benchmark_results = inference_benchmark.benchmark_inference(test_loader, num_batches=50)
    
    # Print benchmark results
    inference_benchmark.print_benchmark_results(benchmark_results)
    
    # Calculate final test accuracy
    test_acc = test_model(model, test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    best_acc = test_acc
    
    # Save final model
    if SAVE_FINAL:
        final_path = os.path.join(SAVE_DIR, 'model_final.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'final_accuracy': test_acc,
            'benchmark_results': benchmark_results,
            'model_config': model_config,
            'dataset_info': dataset_info
        }, final_path)
        print(f"Final model saved: {final_path}")
    
    print(f"Models saved in: {SAVE_DIR}")
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_acc': test_acc,
        'best_accuracy': best_acc,
        'training_time': training_time,
        'benchmark_results': benchmark_results,
        'dataset_info': dataset_info
    }

# ========================================
# RUN TRAINING
# ========================================

if __name__ == "__main__":
    results = main()