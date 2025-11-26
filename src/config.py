from dataclasses import dataclass
import os

@dataclass
class Config:
    # Model configuration
    base_model_name: str = "facebook/wav2vec2-base"
    dataset_repo_id: str = "DigitalUmuganda/ASR_Fellowship_Challenge_Dataset"
    dataset_local_dir: str = "data/raw"
    output_dir: str = "checkpoints/adapter"
    device: str = "cpu"  # Force CPU for compatibility
    
    # Training configuration (CPU-optimized)
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_scheduler_type: str = "linear"
    seed: int = 42
    
    # Adapter configuration
    adapter_dim: int = 64
    
    # Audio processing
    target_sampling_rate: int = 16000
    max_audio_length_s: int = 15
    
    # Evaluation - ADD THE MISSING ATTRIBUTES
    base_transcriptions_path: str = "base_transcriptions.txt"
    finetuned_transcriptions_path: str = "finetuned_transcriptions.txt"
    references_path: str = "references.txt"  # Optional: for WER calculation

cfg = Config()

# Create necessary directories
os.makedirs(cfg.dataset_local_dir, exist_ok=True)
os.makedirs(cfg.output_dir, exist_ok=True)
os.makedirs("outputs", exist_ok=True)