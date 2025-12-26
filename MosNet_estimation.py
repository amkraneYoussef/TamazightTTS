import os
import librosa
import numpy as np
import tensorflow as tf

# === Load MOSNet model ===
from tensorflow.keras.layers import LSTM

def load_mosnet_model(model_path='/kaggle/working/MOSNet/pre_trained/cnn_blstm.h5'):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"LSTM": LSTM},
        compile=False
    )
    return model
# === Extract log-mel spectrogram (MOSNet-compatible) ===
def extract_features(file_path, sr=16000, max_len=1000):
    y, _ = librosa.load(file_path, sr=sr)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=257,
        fmax=8000
    )
    log_S = librosa.power_to_db(S, ref=np.max).T  # (time, mel)

    if log_S.shape[0] > max_len:
        log_S = log_S[:max_len, :]
    else:
        pad = max_len - log_S.shape[0]
        log_S = np.pad(log_S, ((0, pad), (0, 0)), mode='constant')

    return log_S

# === Predict MOS score ===
def predict_mos(model, features):
    x = np.expand_dims(features, axis=0)  # (1, time, mel)
    return float(model.predict(x, verbose=0)[0][0])

# === Compute MOS on entire folder (no CSV) ===
def compute_folder_mos(audio_dir, sample_limit=None, exts=('.wav', '.flac', '.mp3')):
    model = load_mosnet_model()

    files = [
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.lower().endswith(exts)
    ]

    if sample_limit:
        files = files[:sample_limit]

    scores = []
    for fpath in files:
        try:
            features = extract_features(fpath)
            score = predict_mos(model, features)
            scores.append(score)
        except Exception as e:
            print(f"Error processing {os.path.basename(fpath)}: {e}")

    avg_mos = float(np.mean(scores)) if scores else float('nan')
    print(f"Average MOS for {audio_dir}: {avg_mos:.3f} (based on {len(scores)} samples)")
    return avg_mos

# === Path ===
aug_audio_dir = "/kaggle/input/mosoriginal/clips"

# === Compute MOS ===
mos_augmented = compute_folder_mos(aug_audio_dir)

print(f"\nN-MOS (Augmented Clips Folder): {mos_augmented:.3f}")
