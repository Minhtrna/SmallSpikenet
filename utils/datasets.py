"""
Dataset management and data augmentation for SmallSpikenet
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# ========================================
# DATASET CONFIGURATIONS
# ========================================

# Dataset normalization values
DATASET_CONFIGS = {
    'cifar10': {
        'num_classes': 10,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010),
        'dataset_class': torchvision.datasets.CIFAR10
    },
    'cifar100': {
        'num_classes': 100,
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
        'dataset_class': torchvision.datasets.CIFAR100
    }
}

# ========================================
# DATA AUGMENTATION
# ========================================

def get_transforms(dataset_name, use_autoaugment=False):
    """
    Get train and test transforms with fixed augmentation
    
    Args:
        dataset_name (str): Name of the dataset
        use_autoaugment (bool): Whether to use AutoAugment
        
    Returns:
        tuple: (train_transform, test_transform)
    """
    dataset_config = DATASET_CONFIGS[dataset_name]
    
    # Training transforms with fixed augmentation
    train_transforms = []
    
    # AutoAugment (optional)
    if use_autoaugment:
        if dataset_name == 'cifar10':
            train_transforms.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
        elif dataset_name == 'cifar100':
            train_transforms.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))  # Use CIFAR10 policy for CIFAR100
    
    # Fixed augmentations
    train_transforms.extend([
        transforms.RandomCrop(32, padding=4),  # Fixed padding of 4
        transforms.RandomHorizontalFlip(),     # Fixed probability of 0.5
        transforms.ToTensor(),
        transforms.Normalize(dataset_config['mean'], dataset_config['std'])
    ])
    
    # Test transforms (no augmentation)
    test_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(dataset_config['mean'], dataset_config['std'])
    ]
    
    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)

# ========================================
# DATASET FACTORY
# ========================================

class DatasetFactory:
    """
    Factory class for creating datasets and dataloaders
    """
    
    def __init__(self, config):
        self.config = config
        self.dataset_name = config.get('DATASET', 'cifar10').lower()
        
        if self.dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. Supported: {list(DATASET_CONFIGS.keys())}")
        
        self.dataset_config = DATASET_CONFIGS[self.dataset_name]
        
        # Update config with dataset-specific values
        self.config['NUM_CLASSES'] = self.dataset_config['num_classes']
        self.config['DATASET_MEAN'] = self.dataset_config['mean']
        self.config['DATASET_STD'] = self.dataset_config['std']
    
    def create_datasets(self):
        """Create train and test datasets"""
        use_autoaugment = self.config.get('USE_AUTOAUGMENT', False)
        transform_train, transform_test = get_transforms(self.dataset_name, use_autoaugment)
        
        print(f"Loading {self.dataset_name.upper()} dataset...")
        print(f"Dataset config: {self.dataset_config['num_classes']} classes")
        print(f"AutoAugment: {'Enabled' if use_autoaugment else 'Disabled'}")
        print(f"Fixed augmentations: RandomCrop(32, padding=4), RandomHorizontalFlip()")
        
        # Create datasets
        dataset_class = self.dataset_config['dataset_class']
        
        train_dataset = dataset_class(
            root=self.config['DATA_DIR'],
            train=True,
            download=True,
            transform=transform_train
        )
        
        test_dataset = dataset_class(
            root=self.config['DATA_DIR'],
            train=False,
            download=True,
            transform=transform_test
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        
        return train_dataset, test_dataset
    
    def create_dataloaders(self):
        """Create train and test dataloaders"""
        train_dataset, test_dataset = self.create_datasets()
        
        # Create dataloaders
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config['BATCH_SIZE'],
            shuffle=True,
            num_workers=self.config['NUM_WORKERS'],
            pin_memory=self.config['PIN_MEMORY'],
            drop_last=self.config.get('DROP_LAST', False)
        )
        
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config['BATCH_SIZE'],
            shuffle=False,
            num_workers=self.config['NUM_WORKERS'],
            pin_memory=self.config['PIN_MEMORY'],
            drop_last=False
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print(f"Batch size: {self.config['BATCH_SIZE']}")
        
        return train_loader, test_loader
    
    def get_dataset_info(self):
        """Get dataset information"""
        return {
            'name': self.dataset_name,
            'num_classes': self.dataset_config['num_classes'],
            'mean': self.dataset_config['mean'],
            'std': self.dataset_config['std'],
            'use_autoaugment': self.config.get('USE_AUTOAUGMENT', False)
        }

# ========================================
# CONVENIENCE FUNCTIONS
# ========================================

def create_dataloaders(config):
    """
    Convenience function to create dataloaders
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (train_loader, test_loader, dataset_info)
    """
    factory = DatasetFactory(config)
    train_loader, test_loader = factory.create_dataloaders()
    dataset_info = factory.get_dataset_info()
    
    return train_loader, test_loader, dataset_info

def get_dataset_classes(dataset_name):
    """
    Get number of classes for a dataset
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        int: Number of classes
    """
    dataset_name = dataset_name.lower()
    if dataset_name in DATASET_CONFIGS:
        return DATASET_CONFIGS[dataset_name]['num_classes']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_dataset_stats(dataset_name):
    """
    Get normalization statistics for a dataset
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        tuple: (mean, std)
    """
    dataset_name = dataset_name.lower()
    if dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
        return config['mean'], config['std']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# ========================================
# DATASET VALIDATION
# ========================================

def validate_dataset_config(config):
    """
    Validate dataset configuration
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if valid
    """
    dataset_name = config.get('DATASET', 'cifar10').lower()
    
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return True

# ========================================
# SAMPLE VISUALIZATION (OPTIONAL)
# ========================================

def visualize_samples(dataloader, dataset_info, num_samples=8):
    """
    Visualize dataset samples (optional function for debugging)
    
    Args:
        dataloader: DataLoader to sample from
        dataset_info: Dataset information
        num_samples: Number of samples to visualize
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Get class names
        if dataset_info['name'] == 'cifar10':
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck']
        elif dataset_info['name'] == 'cifar100':
            # CIFAR-100 has 100 classes - just use numbers
            class_names = [f'Class_{i}' for i in range(100)]
        else:
            class_names = [f'Class_{i}' for i in range(dataset_info['num_classes'])]
        
        # Get a batch
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
        
        # Denormalize images for visualization
        mean = torch.tensor(dataset_info['mean']).view(3, 1, 1)
        std = torch.tensor(dataset_info['std']).view(3, 1, 1)
        images = images * std + mean
        images = torch.clamp(images, 0, 1)
        
        # Create subplot
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            img = images[i].permute(1, 2, 0).numpy()
            axes[i].imshow(img)
            axes[i].set_title(f'{class_names[labels[i]]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")
    except Exception as e:
        print(f"Error during visualization: {e}")

# ========================================
# TESTING FUNCTION
# ========================================

def test_dataset_factory():
    """Test the dataset factory with sample configurations"""
    
    # Test CIFAR-10 without AutoAugment
    config_cifar10 = {
        'DATASET': 'cifar10',
        'DATA_DIR': './data',
        'BATCH_SIZE': 32,
        'NUM_WORKERS': 2,
        'PIN_MEMORY': True,
        'USE_AUTOAUGMENT': False
    }
    
    print("Testing CIFAR-10 (without AutoAugment)...")
    train_loader, test_loader, dataset_info = create_dataloaders(config_cifar10)
    print(f"Dataset info: {dataset_info}")
    print()
    
    # Test CIFAR-100 with AutoAugment
    config_cifar100 = {
        'DATASET': 'cifar100',
        'DATA_DIR': './data',
        'BATCH_SIZE': 64,
        'NUM_WORKERS': 2,
        'PIN_MEMORY': True,
        'USE_AUTOAUGMENT': True
    }
    
    print("Testing CIFAR-100 (with AutoAugment)...")
    train_loader, test_loader, dataset_info = create_dataloaders(config_cifar100)
    print(f"Dataset info: {dataset_info}")

if __name__ == "__main__":
    test_dataset_factory()