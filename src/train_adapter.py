# src/train_adapter.py (CPU-optimized)
import os
import random
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from src.config import cfg
from src.adapters import insert_adapters_wav2vec2
from src.utils import count_trainable_parameters
import soundfile as sf
import librosa


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def prepare_processor():
    return Wav2Vec2Processor.from_pretrained(cfg.base_model_name)


def prepare_dataset(processor):
    ds = load_dataset(cfg.dataset_repo_id)
    train = ds['train']
    
    def prepare_batch(batch):
        audio = batch['audio']
        if isinstance(audio, dict):
            arr = audio['array']
            sr = audio['sampling_rate']
        else:
            arr, sr = sf.read(audio)
            # truncate long audio
            if len(arr) / sr > cfg.max_audio_length_s:
                arr = arr[: int(cfg.max_audio_length_s * sr)]
            if sr != cfg.target_sampling_rate:
                arr = librosa.resample(arr.astype('float32'), sr, cfg.target_sampling_rate)
            batch['input_values'] = processor(arr, sampling_rate=cfg.target_sampling_rate).input_values[0]
            batch['labels'] = processor.tokenizer(batch['text']).input_ids
            return batch

        train = train.map(prepare_batch, remove_columns=train.column_names)
        return train


def collate_fn(batch):
    import torch
    input_values = [b['input_values'] for b in batch]
    labels = [b['labels'] for b in batch]
    # pad using processor is better but for CPU simplicity we do naive padding
    max_len = max(len(x) for x in input_values)
    input_values_padded = [x + [0.0] * (max_len - len(x)) for x in input_values]
    return {'input_values': torch.tensor(input_values_padded), 'labels': torch.tensor(labels)}


def main():
    set_seed(cfg.seed)
    device = cfg.device
    processor = prepare_processor()
    model = Wav2Vec2ForCTC.from_pretrained(cfg.base_model_name)

    # freeze base
    for p in model.parameters():
        p.requires_grad = False

    # insert adapters (trainable)
    model = insert_adapters_wav2vec2(model, adapter_dim=cfg.adapter_dim)

    # move model to device (CPU)
    model.to(device)
    print('Trainable params:', count_trainable_parameters(model))

    train_ds = prepare_dataset(processor)
    loader = DataLoader(train_ds, batch_size=cfg.per_device_train_batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    num_update_steps = max(1, len(loader) // cfg.gradient_accumulation_steps) * cfg.num_train_epochs
    scheduler = get_scheduler(cfg.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=num_update_steps)

    model.train()
    global_step = 0
    os.makedirs(cfg.output_dir, exist_ok=True)
    for epoch in range(cfg.num_train_epochs):
        for i, batch in enumerate(loader):
            inputs = batch['input_values'].to(device).float()
            labels = batch['labels'].to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()

            if (i + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        # save adapter-only state
        state = {k: v.cpu() for k, v in model.state_dict().items() if 'adapter' in k}
        torch.save(state, os.path.join(cfg.output_dir, f'adapter_epoch_{epoch+1}.pt'))
        print('Saved adapter at epoch', epoch+1)

    print('Done')


if __name__ == '__main__':
    main()