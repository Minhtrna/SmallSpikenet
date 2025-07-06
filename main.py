"""
Simple Training Script for SmallSpikenet
Easy to understand and customize all parameters
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import os
import sys

# Add paths
sys.path.append('Model')
sys.path.append('utils')

from Model.Model import create_model
from utils.training import count_parameters

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

# === DATA AUGMENTATION ===
USE_AUGMENTATION = True    # Enable data augmentation
RANDOM_CROP_PADDING = 4   # Crop padding
RANDOM_HORIZONTAL_FLIP = True
RANDOM_ROTATION = 0       # Rotation degrees (0 = disabled)
COLOR_JITTER = False      # Color augmentation

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

# Dataset configuration
if DATASET == 'cifar10':
    NUM_CLASSES = 10
    DATASET_MEAN = (0.4914, 0.4822, 0.4465)
    DATASET_STD = (0.2023, 0.1994, 0.2010)
elif DATASET == 'cifar100':
    NUM_CLASSES = 100
    DATASET_MEAN = (0.5071, 0.4867, 0.4408)
    DATASET_STD = (0.2675, 0.2565, 0.2761)
else:
    raise ValueError(f"Unsupported dataset: {DATASET}")

# Device setup
if DEVICE == 'auto':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(DEVICE)

print(f" SmallSpikenet Training")
print(f"Dataset: {DATASET.upper()} ({NUM_CLASSES} classes)")
print(f"Device: {device}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Timesteps: {NUM_TIMESTEPS}")
print(f"Threshold: {INIT_THRESHOLD}")
print("=" * 50)

# ========================================
# DATA LOADING
# ========================================

def create_data_transforms():
    """Create train and test transforms"""
    
    # Test transforms (no augmentation)
    test_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD)
    ]
    
    # Train transforms
    train_transforms = []
    
    if USE_AUGMENTATION:
        if RANDOM_CROP_PADDING > 0:
            train_transforms.append(transforms.RandomCrop(32, padding=RANDOM_CROP_PADDING))
        
        if RANDOM_HORIZONTAL_FLIP:
            train_transforms.append(transforms.RandomHorizontalFlip())
        
        if RANDOM_ROTATION > 0:
            train_transforms.append(transforms.RandomRotation(RANDOM_ROTATION))
        
        if COLOR_JITTER:
            train_transforms.append(transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ))
    
    # Add common transforms
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD)
    ])
    
    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)

def create_dataloaders():
    """Create train and test dataloaders"""
    train_transform, test_transform = create_data_transforms()
    
    # Select dataset class
    if DATASET == 'cifar10':
        dataset_class = torchvision.datasets.CIFAR10
    else:
        dataset_class = torchvision.datasets.CIFAR100
    
    print("Loading datasets...")
    
    # Create datasets
    train_dataset = dataset_class(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = dataset_class(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=2, pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, test_loader

# ========================================
# MODEL CREATION
# ========================================

def create_config():
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
        'NUM_CLASSES': NUM_CLASSES,
        'INPUT_SCALE': 1.2
    }
    return config

def create_model_and_optimizer():
    """Create model and optimizer"""
    print("\nCreating model...")
    
    config = create_config()
    model = create_model(config).to(device)
    
    param_count = count_parameters(model)
    print(f" Model created: {param_count:,} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Create scheduler
    if LR_SCHEDULER == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    elif LR_SCHEDULER == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None
    
    return model, optimizer, scheduler

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

def save_checkpoint(model, epoch, loss, acc):
    """Save model checkpoint"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    checkpoint_path = os.path.join(SAVE_DIR, f'model_epoch_{epoch+1}.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'accuracy': acc,
        'config': {
            'NUM_TIMESTEPS': NUM_TIMESTEPS,
            'WIDTH_MULT': WIDTH_MULT,
            'INIT_THRESHOLD': INIT_THRESHOLD,
            'NUM_CLASSES': NUM_CLASSES
        }
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")

# ========================================
# MAIN TRAINING LOOP
# ========================================

def main():
    """Main training function"""
    
    # Create data loaders
    train_loader, test_loader = create_dataloaders()
    
    # Create model
    model, optimizer, scheduler = create_model_and_optimizer()
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training tracking
    train_losses = []
    train_accs = []
    test_accs = []
    best_acc = 0.0
    
    print(f"\nStarting training...")
    print("=" * 50)
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        
        # Train one epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, epoch)
        
        # Test accuracy
        test_acc = test_model(model, test_loader)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Store results
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # Print results
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            if SAVE_FINAL:
                save_checkpoint(model, epoch, train_loss, test_acc)
                print(f"New best accuracy: {best_acc:.2f}%")
        
        # Save periodic checkpoint
        if (epoch + 1) % SAVE_EVERY == 0:
            save_checkpoint(model, epoch, train_loss, test_acc)
        
        print("-" * 30)
    
    # Training completed
    training_time = time.time() - start_time
    
    print(f"\nTraining completed!")
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Models saved in: {SAVE_DIR}")
    
    # Save final model
    if SAVE_FINAL:
        final_path = os.path.join(SAVE_DIR, 'model_final.pth')
        torch.save(model.state_dict(), final_path)
        print(f"Final model saved: {final_path}")
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_accuracy': best_acc,
        'training_time': training_time
    }

# ========================================
# RUN TRAINING
# ========================================

if __name__ == "__main__":
    results = main()