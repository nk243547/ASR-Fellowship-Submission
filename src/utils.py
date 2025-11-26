import re
from jiwer import wer
import torch

def normalize_text(s: str):
    """Normalize text for WER computation"""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[\.,!?;:\"'()\[\]{}]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def count_trainable_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_wer_from_files(ref_file, hyp_file):
    """Compute WER between reference and hypothesis files"""
    with open(ref_file, 'r', encoding='utf-8') as f:
        refs = [line.strip() for line in f if line.strip()]
    
    with open(hyp_file, 'r', encoding='utf-8') as f:
        hyps = [line.strip() for line in f if line.strip()]
    
    # Ensure same number of lines
    min_len = min(len(refs), len(hyps))
    refs = refs[:min_len]
    hyps = hyps[:min_len]
    
    # Normalize texts
    refs_norm = [normalize_text(r) for r in refs]
    hyps_norm = [normalize_text(h) for h in hyps]
    
    return wer('\n'.join(refs_norm), '\n'.join(hyps_norm))

def compute_wer(refs, hyps):
    """Compute WER between reference and hypothesis lists"""
    refs = [normalize_text(r) for r in refs]
    hyps = [normalize_text(h) for h in hyps]
    return wer('\n'.join(refs), '\n'.join(hyps))

def save_model_weights(model, path):
    """Save base model weights"""
    torch.save(model.state_dict(), path)

def load_model_weights(model, path):
    """Load model weights"""
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model