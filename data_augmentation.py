#
# script applied only to Tamazight train portion of Mozilla common voice dataset
#

import os
import random
import torchaudio
import pandas as pd
import torchaudio.transforms as T
from tqdm import tqdm
import torch

random.seed(42)

def add_noise(audio, noise_level=0.005):
    return audio + noise_level * torch.randn_like(audio)

def adjust_volume(audio, gain_db):
    return audio * (10 ** (gain_db / 20))

def time_shift(audio, shift_limit=0.1):
    shift_amt = int(audio.shape[1] * shift_limit)
    shift = random.randint(-shift_amt, shift_amt)
    if shift > 0:
        return torch.cat((audio[:, shift:], torch.zeros(audio.shape[0], shift)), dim=1)
    else:
        return torch.cat((torch.zeros(audio.shape[0], -shift), audio[:, :shift]), dim=1)

def speed_change(audio, rate):
    transform = T.Resample(orig_freq=16000, new_freq=int(16000 * rate))
    stretched = transform(audio)
    return T.Resample(orig_freq=int(16000 * rate), new_freq=16000)(stretched)

def apply_random_augmentation(audio):
    aug_funcs = [
        lambda x: add_noise(x),
        lambda x: adjust_volume(x, random.uniform(-3, 3)),
        lambda x: time_shift(x),
        lambda x: speed_change(x, random.uniform(0.9, 1.1))
    ]
    aug = random.choice(aug_funcs)
    return aug(audio)

def prepare_dataset(
    csv_path, audio_dir, output_dir, output_csv,
    augmentations_per_clip=3
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path, sep=';')
    combined_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        orig_path = os.path.join(audio_dir, row['path'])
        sentence = row['sentence']
        base_filename = os.path.splitext(row['path'])[0]

        try:
            waveform, sr = torchaudio.load(orig_path)
            waveform = T.Resample(orig_freq=sr, new_freq=16000)(waveform)
            sr = 16000

            # Save original as WAV
            orig_wav_filename = f"{base_filename}.wav"
            orig_wav_path = os.path.join(output_dir, orig_wav_filename)
            torchaudio.save(orig_wav_path, waveform, sr)
            combined_data.append({'path': orig_wav_filename, 'sentence': sentence})

            # Generate augmentations
            for i in range(augmentations_per_clip):
                aug_wave = apply_random_augmentation(waveform)
                aug_filename = f"{base_filename}_aug{i}.wav"
                aug_path = os.path.join(output_dir, aug_filename)
                torchaudio.save(aug_path, aug_wave, sr)
                combined_data.append({'path': aug_filename, 'sentence': sentence})

        except Exception as e:
            print(f"Error processing {orig_path}: {e}")

    pd.DataFrame(combined_data).to_csv(output_csv, sep=';', index=False)
    print(f"Combined dataset saved: {output_csv}")
    print(f"Total clips: {len(combined_data)}")

