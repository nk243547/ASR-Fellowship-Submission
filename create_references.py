#!/usr/bin/env python3
"""
Standalone script to create references.txt from the dataset
"""

import os
import sys
from datasets import load_dataset

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now import from src
from config import cfg
from utils import normalize_text

def create_references_file():
    """Create references.txt file from dataset ground truth"""
    
    # Load dataset
    try:
        ds = load_dataset(cfg.dataset_repo_id, split=None)
        print("Dataset loaded successfully")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Pick correct split (same as used in evaluation)
    if "test" in ds:
        test = ds["test"]
        print("Using test split")
    elif "validation" in ds:
        test = ds["validation"]
        print("Using validation split")
    elif "train" in ds:
        test = ds["train"]
        print("Using train split")
    else:
        print(f"No suitable split found. Available: {list(ds.keys())}")
        return

    print(f"Creating references for {len(test)} examples...")

    references_path = 'references.txt'
    
    count = 0
    with open(references_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(test):
            # Extract ground truth text - try different possible keys
            text = None
            for key in ["text", "raw_text", "transcription", "sentence"]:
                if key in ex and ex[key]:
                    text = ex[key]
                    break
            
            if text is None:
                print(f"Warning: No ground truth text found for example {i}")
                text = ""  # Write empty line to maintain alignment
            
            # Normalize and write
            normalized_text = normalize_text(str(text))
            f.write(normalized_text + "\n")
            count += 1
            
            if i % 10 == 0:  # Progress indicator
                print(f"Processed {i}/{len(test)} examples")

    print(f"Created {references_path} with {count} references")
    
    # Show first few references for verification
    print("\nFirst 5 references for verification:")
    with open(references_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"{i+1}: {line.strip()}")
            else:
                break

if __name__ == "__main__":
    create_references_file()