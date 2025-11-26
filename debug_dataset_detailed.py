#!/usr/bin/env python3
"""
Detailed debug script to understand the exact dataset structure
"""

from datasets import load_dataset
import json

def main():
    dataset_repo_id = "DigitalUmuganda/ASR_Fellowship_Challenge_Dataset"
    
    try:
        ds = load_dataset(dataset_repo_id, split=None)
        print("✓ Dataset loaded successfully")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return

    print(f"\nDataset splits: {list(ds.keys())}")
    
    for split_name in ds.keys():
        print(f"\n{'='*60}")
        print(f"{split_name.upper()} SPLIT")
        print(f"{'='*60}")
        split_data = ds[split_name]
        print(f"Number of examples: {len(split_data)}")
        
        if len(split_data) > 0:
            first_ex = split_data[0]
            print("\nALL KEYS AND VALUES for first example:")
            print("-" * 40)
            
            for key, value in first_ex.items():
                print(f"\n{key}:")
                if isinstance(value, (str, int, float, bool)) or value is None:
                    print(f"  Type: {type(value).__name__}")
                    print(f"  Value: {repr(value)}")
                else:
                    print(f"  Type: {type(value).__name__}")
                    if hasattr(value, '__len__'):
                        print(f"  Length: {len(value)}")
                    if hasattr(value, 'shape'):
                        print(f"  Shape: {value.shape}")
                    # Try to show a sample if it's a list/array
                    if hasattr(value, '__getitem__') and len(value) > 0:
                        try:
                            sample = value[0] if len(value) > 1 else value
                            print(f"  Sample: {repr(sample)}")
                        except:
                            pass
            
            # Special attention to audio field
            if 'audio' in first_ex:
                audio_data = first_ex['audio']
                print(f"\nAUDIO FIELD DETAILS:")
                if isinstance(audio_data, dict):
                    for k, v in audio_data.items():
                        print(f"  audio['{k}']: {type(v).__name__} = {repr(v) if isinstance(v, (str, int, float, bool)) or v is None else '...'}")
                else:
                    print(f"  audio: {type(audio_data).__name__}")

if __name__ == "__main__":
    main()