"""
Dataset management and data augmentation for SmallSpikenet
Supports CIFAR-10 and CIFAR-100 datasets with configurable augmentation
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
# DATA AUGMENTATION POLICIES
# ========================================

class DataAugmentationPolicy:
    """
    Configurable data augmentation policies for different training strategies
    """
    
    @staticmethod
    def get_basic_policy(dataset_name, config):
        """Basic augmentation policy - standard transforms"""
        dataset_config = DATASET_CONFIGS[dataset_name]
        
        train_transforms = []
        test_transforms = []
        
        # Training augmentations
        if config.get('RANDOM_CROP_PADDING', 0) > 0:
            train_transforms.append(
                transforms.RandomCrop(32, padding=config['RANDOM_CROP_PADDING'])
            )
        
        if config.get('RANDOM_HORIZONTAL_FLIP', False):
            train_transforms.append(transforms.RandomHorizontalFlip())
        
        # Common transforms
        common_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(dataset_config['mean'], dataset_config['std'])
        ]
        
        train_transforms.extend(common_transforms)
        test_transforms.extend(common_transforms)
        
        return transforms.Compose(train_transforms), transforms.Compose(test_transforms)
    
    @staticmethod
    def get_enhanced_policy(dataset_name, config):
        """Enhanced augmentation policy - more aggressive transforms"""
        dataset_config = DATASET_CONFIGS[dataset_name]
        
        train_transforms = []
        
        # Enhanced training augmentations
        if config.get('RANDOM_CROP_PADDING', 0) > 0:
            train_transforms.append(
                transforms.RandomCrop(32, padding=config['RANDOM_CROP_PADDING'])
            )
        
        if config.get('RANDOM_HORIZONTAL_FLIP', False):
            train_transforms.append(transforms.RandomHorizontalFlip())
        
        # Additional augmentations for enhanced policy
        if config.get('RANDOM_ROTATION', 0) > 0:
            train_transforms.append(
                transforms.RandomRotation(degrees=config['RANDOM_ROTATION'])
            )
        
        if config.get('COLOR_JITTER', False):
            train_transforms.append(
                transforms.ColorJitter(
                    brightness=config.get('COLOR_JITTER_BRIGHTNESS', 0.2),
                    contrast=config.get('COLOR_JITTER_CONTRAST', 0.2),
                    saturation=config.get('COLOR_JITTER_SATURATION', 0.2),
                    hue=config.get('COLOR_JITTER_HUE', 0.1)
                )
            )
        
        if config.get('RANDOM_ERASING', False):
            train_transforms.extend([
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=config.get('RANDOM_ERASING_PROB', 0.5),
                    scale=config.get('RANDOM_ERASING_SCALE', (0.02, 0.33)),
                    ratio=config.get('RANDOM_ERASING_RATIO', (0.3, 3.3))
                ),
                transforms.Normalize(dataset_config['mean'], dataset_config['std'])
            ])
        else:
            train_transforms.extend([
                transforms.ToTensor(),
                transforms.Normalize(dataset_config['mean'], dataset_config['std'])
            ])
        
        # Test transforms (no augmentation)
        test_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(dataset_config['mean'], dataset_config['std'])
        ]
        
        return transforms.Compose(train_transforms), transforms.Compose(test_transforms)
    
    @staticmethod
    def get_light_policy(dataset_name, config):
        """Light augmentation policy - minimal transforms for energy efficiency"""
        dataset_config = DATASET_CONFIGS[dataset_name]
        
        train_transforms = []
        
        # Minimal training augmentations
        if config.get('RANDOM_HORIZONTAL_FLIP', False):
            train_transforms.append(transforms.RandomHorizontalFlip())
        
        # Common transforms
        common_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(dataset_config['mean'], dataset_config['std'])
        ]
        
        train_transforms.extend(common_transforms)
        test_transforms = common_transforms.copy()
        
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
    
    def get_transforms(self):
        """Get train and test transforms based on augmentation policy"""
        augmentation_policy = self.config.get('AUGMENTATION_POLICY', 'basic')
        
        if augmentation_policy == 'enhanced':
            return DataAugmentationPolicy.get_enhanced_policy(self.dataset_name, self.config)
        elif augmentation_policy == 'light':
            return DataAugmentationPolicy.get_light_policy(self.dataset_name, self.config)
        else:  # basic
            return DataAugmentationPolicy.get_basic_policy(self.dataset_name, self.config)
    
    def create_datasets(self):
        """Create train and test datasets"""
        transform_train, transform_test = self.get_transforms()
        
        print(f"Loading {self.dataset_name.upper()} dataset...")
        print(f"Dataset config: {self.dataset_config['num_classes']} classes")
        print(f"Augmentation policy: {self.config.get('AUGMENTATION_POLICY', 'basic')}")
        
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
            'augmentation_policy': self.config.get('AUGMENTATION_POLICY', 'basic')
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
    
    augmentation_policy = config.get('AUGMENTATION_POLICY', 'basic')
    valid_policies = ['basic', 'enhanced', 'light']
    
    if augmentation_policy not in valid_policies:
        raise ValueError(f"Invalid augmentation policy: {augmentation_policy}. Valid options: {valid_policies}")
    
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
    
    # Test CIFAR-10
    config_cifar10 = {
        'DATASET': 'cifar10',
        'DATA_DIR': './data',
        'BATCH_SIZE': 32,
        'NUM_WORKERS': 2,
        'PIN_MEMORY': True,
        'AUGMENTATION_POLICY': 'basic',
        'RANDOM_CROP_PADDING': 4,
        'RANDOM_HORIZONTAL_FLIP': True
    }
    
    print("Testing CIFAR-10...")
    train_loader, test_loader, dataset_info = create_dataloaders(config_cifar10)
    print(f"Dataset info: {dataset_info}")
    print()
    
    # Test CIFAR-100
    config_cifar100 = {
        'DATASET': 'cifar100',
        'DATA_DIR': './data',
        'BATCH_SIZE': 64,
        'NUM_WORKERS': 2,
        'PIN_MEMORY': True,
        'AUGMENTATION_POLICY': 'enhanced',
        'RANDOM_CROP_PADDING': 4,
        'RANDOM_HORIZONTAL_FLIP': True,
        'RANDOM_ROTATION': 15,
        'COLOR_JITTER': True,
        'RANDOM_ERASING': True
    }
    
    print("Testing CIFAR-100...")
    train_loader, test_loader, dataset_info = create_dataloaders(config_cifar100)
    print(f"Dataset info: {dataset_info}")

if __name__ == "__main__":
    test_dataset_factory()