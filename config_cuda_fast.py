"""
Optimized training configuration for CUDA acceleration.
Target: < 30 minutes training time
"""

# Training Configuration
TRAINING_CONFIG = {
    # Device settings
    'device': 'cuda',  # Will auto-fallback to CPU if CUDA unavailable
    'use_amp': True,   # Mixed precision training (FP16) - 2-3x speedup
    
    # Batch settings - optimized for GPU memory
    'batch_size': 32,              # Base batch size
    'gradient_accumulation_steps': 2,  # Effective batch = 64
    'num_workers': 4,              # Parallel data loading
    
    # Training duration
    'num_epochs': 10,
    'patience': 5,  # Early stopping patience
    
    # Optimizer settings
    'learning_rate': 2e-4,  # Slightly higher for larger batch
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    
    # Loss settings
    'evidential_coef': 0.1,
    
    # Model architecture
    'drug_model': 'seyonec/PubChem10M_SMILES_BPE_450k',
    'protein_model': 'Rostlab/prot_bert',
    'attention_dim': 512,
    'hidden_dims': [512, 256],
    'freeze_encoders': True,  # Freeze pretrained encoders for speed
}

# Performance tuning tips
PERFORMANCE_TIPS = """
🚀 CUDA OPTIMIZATION TIPS:

1. BATCH SIZE:
   - Increase if you have more GPU memory (16GB+ → try 64)
   - Decrease if you get OOM errors (8GB → try 16)
   
2. MIXED PRECISION (AMP):
   - Enabled by default for 2-3x speedup
   - Uses FP16 instead of FP32
   - Minimal accuracy loss
   
3. GRADIENT ACCUMULATION:
   - Simulates larger batch sizes
   - Current: 32 * 2 = 64 effective batch size
   
4. DATA LOADING:
   - num_workers=4 for parallel loading
   - pin_memory=True for faster GPU transfer
   
5. ENCODER FREEZING:
   - Pretrained encoders are frozen
   - Only trains attention + prediction head
   - Much faster convergence
   
6. LEARNING RATE SCHEDULE:
   - OneCycleLR with 10% warmup
   - Cosine annealing for smooth convergence
   
Expected Performance:
- GPU: ~15-25 minutes for 10 epochs
- CPU: ~2-3 hours for 10 epochs
- Speedup: 6-10x with CUDA + AMP
"""

if __name__ == '__main__':
    print(PERFORMANCE_TIPS)
    print("\nCurrent Configuration:")
    print("-" * 60)
    for key, value in TRAINING_CONFIG.items():
        print(f"{key:30s}: {value}")
