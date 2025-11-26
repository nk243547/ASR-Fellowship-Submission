#!/usr/bin/env python3
"""
Check if this is a WebDataset format dataset
"""

from datasets import load_dataset

def main():
    dataset_repo_id = "DigitalUmuganda/ASR_Fellowship_Challenge_Dataset"
    
    ds = load_dataset(dataset_repo_id, split='train')
    
    # Check the first example
    example = ds[0]
    print("First example keys:", list(example.keys()))
    
    # WebDataset often has these fields
    webdataset_fields = ['__key__', '__url__', 'webm', 'txt', 'json']
    found_fields = [field for field in webdataset_fields if field in example]
    print(f"WebDataset fields found: {found_fields}")
    
    # Check for text in various possible locations
    if 'txt' in example:
        print(f"'txt' field content: {example['txt']}")
    
    if 'json' in example:
        print(f"'json' field type: {type(example['json'])}")
        try:
            if hasattr(example['json'], 'decode'):
                json_text = example['json'].decode('utf-8')
                print(f"'json' field content: {json_text}")
        except:
            pass
    
    # Check if text is in the audio dictionary
    if 'audio' in example and isinstance(example['audio'], dict):
        audio = example['audio']
        print("Audio dict keys:", list(audio.keys()))
        if 'text' in audio:
            print(f"Audio text: {audio['text']}")

if __name__ == "__main__":
    main()