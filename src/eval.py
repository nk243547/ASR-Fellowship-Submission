import os
import io
import tempfile
import warnings
import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
from src.config import cfg
from src.adapters import insert_adapters_wav2vec2
from src.utils import normalize_text, compute_wer_from_files

# Suppress librosa warnings to clean up output
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", message="PySoundFile failed")

def extract_audio_and_sr(example):
    """
    Detects audio for all dataset formats:
    - HF Audio objects
    - Path-based datasets
    - WebDataset (.webm / .opus) binary blobs
    """
    keys = example.keys()

    # 1. HF Audio object
    if "audio" in keys:
        audio = example["audio"]
        if isinstance(audio, dict):
            return audio["array"], audio["sampling_rate"]
        else:
            try:
                arr, sr = sf.read(audio)
                return arr, sr
            except Exception as e:
                print(f"Warning: Could not read audio file {audio}: {e}")
                return None, None

    # 2. Standard file paths
    for key in ["path", "file", "audio_filepath"]:
        if key in keys:
            try:
                arr, sr = sf.read(example[key])
                return arr, sr
            except Exception as e:
                print(f"Warning: Could not read audio file {example[key]}: {e}")
                continue

    # 3. WebDataset (.webm) — YOUR CASE
    if "webm" in keys:
        data = example["webm"]

        if data is None:
            return None, None

        # Convert dict-with-bytes → bytes
        if isinstance(data, bytes):
            blob = data
        elif isinstance(data, dict) and "bytes" in data:
            blob = data["bytes"]
        else:
            print(f"Warning: Unknown 'webm' format: {type(data)}, skipping example")
            return None, None

        # WebM/Opus cannot be decoded by soundfile → we use librosa
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp:
            tmp.write(blob)
            tmp.flush()
            try:
                arr, sr = librosa.load(tmp.name, sr=None, res_type='kaiser_fast')
                return arr, sr
            except Exception as e:
                print(f"Warning: Could not decode webm audio: {e}")
                return None, None

    # 4. No usable audio key found
    print(f"Warning: No audio field found in example. Keys = {keys}")
    return None, None

def load_model(adapter_path=None):
    """Load model with optional adapter weights"""
    model = Wav2Vec2ForCTC.from_pretrained(cfg.base_model_name)

    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False

    # Insert adapters
    model = insert_adapters_wav2vec2(model, adapter_dim=cfg.adapter_dim)

    # Load adapter weights if available
    if adapter_path and os.path.exists(adapter_path):
        try:
            adapter_state = torch.load(adapter_path, map_location="cpu")
            # Filter out keys that don't exist in the model
            model_state = model.state_dict()
            filtered_adapter_state = {k: v for k, v in adapter_state.items() if k in model_state}
            model_state.update(filtered_adapter_state)
            model.load_state_dict(model_state)
            print("Loaded adapter weights from:", adapter_path)
        except Exception as e:
            print(f"Warning: Could not load adapter from {adapter_path}: {e}")
            print("Continuing with untrained adapters...")

    model.to(cfg.device)
    model.eval()
    return model

def decode_array(model, processor, array, sr):
    """Decode audio array to text transcription"""
    if sr != cfg.target_sampling_rate:
        array = librosa.resample(
            y=array.astype("float32"), 
            orig_sr=sr, 
            target_sr=cfg.target_sampling_rate
        )

    inputs = processor(
        array,
        sampling_rate=cfg.target_sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.input_values.to(cfg.device)

    with torch.no_grad():
        logits = model(inputs).logits

    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]

def generate_transcriptions():
    """Generate base and fine-tuned transcriptions"""
    os.makedirs("outputs", exist_ok=True)

    # Use safe attribute access with fallback values
    base_output = getattr(cfg, 'base_transcriptions_path', 'base_transcriptions.txt')
    finetuned_output = getattr(cfg, 'finetuned_transcriptions_path', 'finetuned_transcriptions.txt')
    references_path = getattr(cfg, 'references_path', 'references.txt')

    print(f"Output files:")
    print(f"  Base: {base_output}")
    print(f"  Fine-tuned: {finetuned_output}")
    print(f"  References: {references_path}")

    # Load dataset
    try:
        ds = load_dataset(cfg.dataset_repo_id, split=None)
        print("Dataset loaded successfully")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Pick correct split
    if "test" in ds:
        test = ds["test"]
        print("Using test split")
    elif "validation" in ds:
        test = ds["validation"]
        print("Using validation split")
    elif "train" in ds:
        test = ds["train"]
        print("Using train split (no test/validation found)")
    else:
        print(f"No suitable split found. Available: {list(ds.keys())}")
        return

    print(f"Dataset size: {len(test)} examples")

    processor = Wav2Vec2Processor.from_pretrained(cfg.base_model_name)

    # BASE MODEL
    print("\n" + "="*50)
    print("Generating base transcriptions...")
    model_base = load_model(None)

    base_count = 0
    with open(base_output, "w", encoding="utf-8") as fbase:
        for i, ex in enumerate(test):
            if i % 10 == 0:  # Progress indicator
                print(f"Processing base example {i}/{len(test)}")
                
            try:
                arr, sr = extract_audio_and_sr(ex)
                if arr is None or sr is None:
                    print(f"Warning: Skipping example {i} due to missing/invalid audio")
                    fbase.write("\n")
                    continue
                hyp = decode_array(model_base, processor, arr, sr)
                fbase.write(normalize_text(hyp) + "\n")
                base_count += 1
            except Exception as e:
                print(f"Warning: Error processing example {i}: {e}")
                fbase.write("\n")
                continue

    print(f"Generated {base_count}/{len(test)} base transcriptions")

    # FINE-TUNED MODEL
    print("\n" + "="*50)
    print("Generating fine-tuned transcriptions...")

    # Find valid checkpoint files
    valid_ckpts = []
    if os.path.exists(cfg.output_dir):
        for p in os.listdir(cfg.output_dir):
            if p.endswith(".pt"):
                ckpt_path = os.path.join(cfg.output_dir, p)
                try:
                    torch.load(ckpt_path, map_location="cpu", weights_only=True)
                    valid_ckpts.append(ckpt_path)
                    print(f"Found valid checkpoint: {p}")
                except Exception as e:
                    print(f"Warning: Skipping corrupted checkpoint {p}: {e}")
    
    if not valid_ckpts:
        print("Warning: No valid adapter checkpoints found. Using base model for fine-tuned transcriptions.")
        adapter_ckpt = None
    else:
        adapter_ckpt = sorted(valid_ckpts)[-1]
        print(f"Using latest checkpoint: {adapter_ckpt}")

    model_ft = load_model(adapter_ckpt)

    ft_count = 0
    with open(finetuned_output, "w", encoding="utf-8") as fft:
        for i, ex in enumerate(test):
            if i % 10 == 0:  # Progress indicator
                print(f"Processing fine-tuned example {i}/{len(test)}")
                
            try:
                arr, sr = extract_audio_and_sr(ex)
                if arr is None or sr is None:
                    print(f"Warning: Skipping example {i} due to missing/invalid audio")
                    fft.write("\n")
                    continue
                hyp = decode_array(model_ft, processor, arr, sr)
                fft.write(normalize_text(hyp) + "\n")
                ft_count += 1
            except Exception as e:
                print(f"Warning: Error processing example {i}: {e}")
                fft.write("\n")
                continue

    print(f"Generated {ft_count}/{len(test)} fine-tuned transcriptions")
    
    print("\n" + "="*50)
    print("Evaluation complete.")
    print(f"Saved transcriptions:")
    print(f"  - Base: {base_output}")
    print(f"  - Fine-tuned: {finetuned_output}")

    # Calculate WER if reference file exists
    if os.path.exists(references_path):
        try:
            print(f"\nCalculating WER using references from: {references_path}")
            base_wer = compute_wer_from_files(references_path, base_output)
            ft_wer = compute_wer_from_files(references_path, finetuned_output)
            print(f"\n=== RESULTS ===")
            print(f"Base Model WER: {base_wer:.4f} ({base_wer*100:.2f}%)")
            print(f"Fine-tuned Model WER: {ft_wer:.4f} ({ft_wer*100:.2f}%)")
            print(f"Improvement: {(base_wer - ft_wer)*100:.2f}%")
            
            # Save results to file
            with open("results.txt", "w") as f:
                f.write(f"Base Model WER: {base_wer:.4f} ({base_wer*100:.2f}%)\n")
                f.write(f"Fine-tuned Model WER: {ft_wer:.4f} ({ft_wer*100:.2f}%)\n")
                f.write(f"Improvement: {(base_wer - ft_wer)*100:.2f}%\n")
            print("Results saved to results.txt")
            
        except Exception as e:
            print(f"Could not compute WER: {e}")
    else:
        print(f"\nReference file '{references_path}' not found.")
        print("To calculate WER, please create a references.txt file with ground truth transcriptions.")
        print("Each line should contain the reference text for the corresponding audio file.")

if __name__ == "__main__":
    generate_transcriptions()