"""
Configuration file for SmallSpikenet training
All hyperparameters are centralized here for easy tuning
"""

# ========================================
# DATASET CONFIGURATION
# ========================================

# Dataset Selection
DATASET = 'cifar10'  # Options: 'cifar10', 'cifar100'

# Data Augmentation
AUGMENTATION_POLICY = 'basic'  # Options: 'basic', 'enhanced', 'light'
RANDOM_CROP_PADDING = 4
RANDOM_HORIZONTAL_FLIP = True

# Enhanced Augmentation Options (used when AUGMENTATION_POLICY = 'enhanced')
RANDOM_ROTATION = 15  # degrees
COLOR_JITTER = True
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.2
COLOR_JITTER_HUE = 0.1
RANDOM_ERASING = True
RANDOM_ERASING_PROB = 0.5
RANDOM_ERASING_SCALE = (0.02, 0.33)
RANDOM_ERASING_RATIO = (0.3, 3.3)

# Data Loading
NUM_WORKERS = 4
PIN_MEMORY = True
DROP_LAST = False

# ========================================
# TRAINING CONFIGURATION
# ========================================

# Training Parameters
NUM_EPOCHS = 150
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 4e-5

# Learning Rate Scheduling
LR_SCHEDULER = 'cosine'  # Options: 'cosine', 'step', 'exponential', 'none'
LR_STEP_SIZE = 30        # For step scheduler
LR_GAMMA = 0.1           # For step/exponential scheduler

# Gradient Clipping
GRAD_CLIP_NORM = 1.0     # Maximum gradient norm

# ========================================
# SNN PARAMETERS
# ========================================

# Temporal Parameters
NUM_TIMESTEPS = 2        # Number of timesteps (1-4 recommended)

# LM-HT-LIF Neuron Parameters
NUM_THRESHOLDS = 4       # Multi-threshold levels (2-8 range)
LEAKAGE = 0.2           # Membrane leakage (0.1-0.3 range)
MEMORY_FACTOR = 0.1     # Memory retention (0.05-0.2 range)
RESET_MODE = 'hybrid'   # Reset mechanism: 'soft', 'hard', 'hybrid'
INIT_THRESHOLD = 0.5    # Initial threshold (0.2-0.8 range)

# Neuron Learning Rates
NEURON_LR_MULTIPLIER = 5.0  # Multiplier for neuron parameter learning rates

# ========================================
# MODEL ARCHITECTURE
# ========================================

# Model Scaling
WIDTH_MULT = 1.0        # Model width multiplier (0.5-2.0 range)
NUM_CLASSES = 10        # Will be auto-set based on dataset

# Input Preprocessing
INPUT_SCALE = 1.2       # Input scaling factor (x = x / INPUT_SCALE)

# Dropout
DROPOUT_RATE = 0.2      # Dropout rate for classifier

# ========================================
# PATHS AND SAVING
# ========================================

# Directories
DATA_DIR = './data'
SAVE_DIR = './checkpoints'
RESULTS_DIR = './results'

# File Names
RESULTS_FILE = 'training_results.csv'
MODEL_NAME = 'mini_mobilenetv2_lif'

# Checkpoint Saving
SAVE_CHECKPOINT_EVERY = 10  # Save checkpoint every N epochs
SAVE_FINAL_MODEL = True     # Save final trained model

# ========================================
# INFERENCE AND BENCHMARKING
# ========================================

# Benchmark Parameters
BENCHMARK_BATCHES = 20      # Number of batches for inference benchmark
WARMUP_RUNS = 3            # Number of warmup runs before benchmark

# Energy Monitoring
ENABLE_ENERGY_MONITORING = True
ENERGY_MEASUREMENT_INTERVAL = 100  # ms

# ========================================
# DEVICE CONFIGURATION
# ========================================

# Device Selection
USE_CUDA = True             # Use CUDA if available
DEVICE = 'auto'             # 'auto', 'cpu', 'cuda', or specific device like 'cuda:0'

# Performance Optimizations
CUDNN_BENCHMARK = True      # Enable cudnn benchmark for faster training ?
MIXED_PRECISION = False     # Enable mixed precision training (experimental)

# ========================================
# EXPERIMENT CONFIGURATION PRESETS
# ========================================

# Predefined configurations for different experiments
EXPERIMENT_CONFIGS = {
    'default': {
        'description': 'Default configuration for CIFAR-10',
        # Uses all default values above
    },
    
    'cifar100': {
        'description': 'Configuration for CIFAR-100',
        'DATASET': 'cifar100',
        'NUM_EPOCHS': 200,
        'LEARNING_RATE': 0.0008,
        'AUGMENTATION_POLICY': 'enhanced',
        'MODEL_NAME': 'mini_mobilenetv2_lif_cifar100',
        'RESULTS_FILE': 'cifar100_results.csv'
    },
    
    'high_accuracy': {
        'description': 'Configuration optimized for highest accuracy',
        'NUM_EPOCHS': 200,
        'LEARNING_RATE': 0.0005,
        'INIT_THRESHOLD': 0.4,
        'NUM_TIMESTEPS': 3,
        'GRAD_CLIP_NORM': 0.5,
        'AUGMENTATION_POLICY': 'enhanced'
    },
    
    'energy_efficient': {
        'description': 'Configuration optimized for energy efficiency',
        'NUM_TIMESTEPS': 1,
        'INIT_THRESHOLD': 0.7,
        'WIDTH_MULT': 0.75,
        'BATCH_SIZE': 512,
        'AUGMENTATION_POLICY': 'light'
    },
    
    'fast_training': {
        'description': 'Configuration for faster training',
        'NUM_EPOCHS': 100,
        'LEARNING_RATE': 0.002,
        'BATCH_SIZE': 512,
        'NUM_TIMESTEPS': 1,
        'AUGMENTATION_POLICY': 'basic'
    },
    
    'large_model': {
        'description': 'Larger model configuration',
        'WIDTH_MULT': 1.5,
        'NUM_TIMESTEPS': 4,
        'LEARNING_RATE': 0.0008,
        'GRAD_CLIP_NORM': 2.0,
        'AUGMENTATION_POLICY': 'enhanced'
    }
}

# ========================================
# HELPER FUNCTIONS
# ========================================

def get_config(experiment='default'):
    """
    Get configuration for specific experiment
    
    Args:
        experiment (str): Name of experiment configuration
        
    Returns:
        dict: Configuration dictionary
    """
    import copy
    
    # Start with default config (all module-level variables)
    config = {}
    current_module = globals()
    
    for key, value in current_module.items():
        if key.isupper() and not key.startswith('EXPERIMENT_'):
            config[key] = value
    
    # Apply experiment-specific overrides
    if experiment in EXPERIMENT_CONFIGS:
        experiment_config = EXPERIMENT_CONFIGS[experiment]
        for key, value in experiment_config.items():
            if key != 'description':
                config[key] = value
    
    return config

def print_config(config=None, experiment='default'):
    """
    Print current configuration
    
    Args:
        config (dict, optional): Configuration dictionary
        experiment (str): Name of experiment
    """
    if config is None:
        config = get_config(experiment)
    
    print(f"\n=== Configuration: {experiment} ===")
    if experiment in EXPERIMENT_CONFIGS:
        print(f"Description: {EXPERIMENT_CONFIGS[experiment].get('description', 'No description')}")
    
    print("\n--- Training Parameters ---")
    print(f"Epochs:           {config['NUM_EPOCHS']}")
    print(f"Batch size:       {config['BATCH_SIZE']}")
    print(f"Learning rate:    {config['LEARNING_RATE']}")
    print(f"Weight decay:     {config['WEIGHT_DECAY']}")
    print(f"LR scheduler:     {config['LR_SCHEDULER']}")
    print(f"Gradient clip:    {config['GRAD_CLIP_NORM']}")
    
    print("\n--- SNN Parameters ---")
    print(f"Timesteps:        {config['NUM_TIMESTEPS']}")
    print(f"Thresholds:       {config['NUM_THRESHOLDS']}")
    print(f"Leakage:          {config['LEAKAGE']}")
    print(f"Memory factor:    {config['MEMORY_FACTOR']}")
    print(f"Reset mode:       {config['RESET_MODE']}")
    print(f"Init threshold:   {config['INIT_THRESHOLD']}")
    print(f"Neuron LR mult:   {config['NEURON_LR_MULTIPLIER']}")
    
    print("\n--- Model Architecture ---")
    print(f"Width multiplier: {config['WIDTH_MULT']}")
    print(f"Number of classes:{config['NUM_CLASSES']}")
    print(f"Input scaling:    /{config['INPUT_SCALE']}")
    print(f"Dropout rate:     {config['DROPOUT_RATE']}")
    
    print("=============================\n")

def validate_config(config):
    """
    Validate configuration parameters
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if valid, raises ValueError if invalid
    """
    # Validate ranges
    if not (1 <= config['NUM_TIMESTEPS'] <= 8):
        raise ValueError(f"NUM_TIMESTEPS must be between 1-8, got {config['NUM_TIMESTEPS']}")
    
    if not (0.1 <= config['INIT_THRESHOLD'] <= 1.0):
        raise ValueError(f"INIT_THRESHOLD must be between 0.1-1.0, got {config['INIT_THRESHOLD']}")
    
    if not (0.0 <= config['LEAKAGE'] <= 1.0):
        raise ValueError(f"LEAKAGE must be between 0.0-1.0, got {config['LEAKAGE']}")
    
    if not (0.1 <= config['WIDTH_MULT'] <= 3.0):
        raise ValueError(f"WIDTH_MULT must be between 0.1-3.0, got {config['WIDTH_MULT']}")
    
    if config['RESET_MODE'] not in ['soft', 'hard', 'hybrid']:
        raise ValueError(f"RESET_MODE must be 'soft', 'hard', or 'hybrid', got {config['RESET_MODE']}")
    
    if config['LR_SCHEDULER'] not in ['cosine', 'step', 'exponential', 'none']:
        raise ValueError(f"LR_SCHEDULER must be 'cosine', 'step', 'exponential', or 'none', got {config['LR_SCHEDULER']}")
    
    return True

# ========================================
# QUICK ACCESS FUNCTIONS
# ========================================

def quick_config(timesteps=None, threshold=None, width_mult=None, learning_rate=None):
    """
    Quick configuration override for common parameters
    
    Args:
        timesteps (int, optional): Number of timesteps
        threshold (float, optional): Initial threshold
        width_mult (float, optional): Width multiplier
        learning_rate (float, optional): Learning rate
        
    Returns:
        dict: Updated configuration
    """
    config = get_config('default')
    
    if timesteps is not None:
        config['NUM_TIMESTEPS'] = timesteps
    if threshold is not None:
        config['INIT_THRESHOLD'] = threshold
    if width_mult is not None:
        config['WIDTH_MULT'] = width_mult
    if learning_rate is not None:
        config['LEARNING_RATE'] = learning_rate
    
    validate_config(config)
    return config