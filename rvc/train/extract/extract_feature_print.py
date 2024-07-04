import os
import sys
import tqdm
import mlx.core as mx
import mlx.nn.functional as F
import soundfile as sf
import numpy as np

now_dir = os.getcwd()
sys.path.append(now_dir)
from rvc.lib.utils import load_embedding

device = sys.argv[1]  # Note: MLX doesn't use device specification like PyTorch
n_parts = int(sys.argv[2])
i_part = int(sys.argv[3])
i_gpu = sys.argv[4]  # Note: MLX doesn't use explicit GPU selection
exp_dir = sys.argv[5]
version = sys.argv[6]
is_half = bool(sys.argv[7])  # Note: MLX uses float16 by default on GPU
embedder_model = sys.argv[8]
try:
    embedder_model_custom = sys.argv[9]
except:
    embedder_model_custom = None

wav_path = f"{exp_dir}/sliced_audios_16k"
out_path = f"{exp_dir}/v1_extracted" if version == "v1" else f"{exp_dir}/v2_extracted"
os.makedirs(out_path, exist_ok=True)

def read_wave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = mx.array(wav)
    feats = mx.cast(feats, mx.float16) if is_half else mx.cast(feats, mx.float32)
    feats = mx.mean(feats, axis=-1) if feats.ndim == 2 else feats
    feats = feats.reshape(1, -1)
    if normalize:
        feats = F.layer_norm(feats, feats.shape)
    return feats

print("Starting feature extraction...")
models, saved_cfg, task = load_embedding(embedder_model, embedder_model_custom)
model = models[0]
# Note: MLX doesn't require explicit device moving or half-precision conversion

todo = sorted(os.listdir(wav_path))[i_part::n_parts]
n = max(1, len(todo) // 10)

if len(todo) == 0:
    print(
        "An error occurred in the feature extraction, make sure you have provided the audios correctly."
    )
else:
    with tqdm.tqdm(total=len(todo)) as pbar:
        for idx, file in enumerate(todo):
            try:
                if file.endswith(".wav"):
                    wav_file_path = os.path.join(wav_path, file)
                    out_file_path = os.path.join(out_path, file.replace("wav", "npy"))

                    if os.path.exists(out_file_path):
                        continue

                    feats = read_wave(wav_file_path, normalize=saved_cfg.task.normalize)
                    padding_mask = mx.zeros(feats.shape, dtype=mx.bool_)
                    inputs = {
                        "source": feats,
                        "padding_mask": padding_mask,
                        "output_layer": 9 if version == "v1" else 12,
                    }
                    
                    logits = model.extract_features(**inputs)
                    feats = (
                        model.final_proj(logits[0])
                        if version == "v1"
                        else logits[0]
                    )

                    feats = feats.squeeze(0).astype(mx.float32).numpy()
                    if np.isnan(feats).sum() == 0:
                        np.save(out_file_path, feats, allow_pickle=False)
                    else:
                        print(f"{file} is invalid")
                    pbar.set_description(f"Processing {file} {feats.shape}")
            except Exception as error:
                print(error)
            pbar.update(1)

    print("Feature extraction completed successfully!")
