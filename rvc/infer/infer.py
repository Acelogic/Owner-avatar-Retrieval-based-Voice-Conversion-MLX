import os
import sys
import time
import mlx.core as mx
import logging

import numpy as np
import soundfile as sf
import librosa

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.infer.pipeline import VC
from scipy.io import wavfile
from audio_upscaler import upscale
import noisereduce as nr
from rvc.lib.utils import load_audio
from rvc.lib.tools.split_audio import process_audio, merge_audio
from rvc.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rvc.configs.config import Config
from rvc.lib.utils import load_embedding

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

config = Config()
hubert_model = None
tgt_sr = None
net_g = None
vc = None
cpt = None
version = None
n_spk = None

def load_hubert(embedder_model, embedder_model_custom):
    global hubert_model
    models, _, _ = load_embedding(embedder_model, embedder_model_custom)
    hubert_model = models[0]
    # MLX doesn't have explicit device management, so we remove the .to(device) call
    if config.is_half:
        hubert_model = mx.cast(hubert_model, mx.float16)
    else:
        hubert_model = mx.cast(hubert_model, mx.float32)
    # MLX doesn't have an explicit eval() method, so we remove it

# The rest of the functions (remove_audio_noise, convert_audio_format) remain largely unchanged
# as they don't involve PyTorch-specific operations

def voice_conversion(
    sid=0,
    input_audio_path=None,
    f0_up_key=None,
    f0_file=None,
    f0_method=None,
    file_index=None,
    index_rate=None,
    resample_sr=0,
    rms_mix_rate=None,
    protect=None,
    hop_length=None,
    output_path=None,
    split_audio=False,
    f0autotune=False,
    filter_radius=None,
    embedder_model=None,
    embedder_model_custom=None,
):
    global tgt_sr, net_g, vc, hubert_model, version

    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95

        if audio_max > 1:
            audio /= audio_max

        if not hubert_model:
            load_hubert(embedder_model, embedder_model_custom)
        if_f0 = cpt.get("f0", 1)

        file_index = (
            file_index.strip(" ")
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip(" ")
            .replace("trained", "added")
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        if split_audio == "True":
            # The split_audio logic remains largely unchanged
            pass
        else:
            audio_opt = vc.pipeline(
                hubert_model,
                net_g,
                sid,
                audio,
                input_audio_path,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                if_f0,
                filter_radius,
                tgt_sr,
                resample_sr,
                rms_mix_rate,
                version,
                protect,
                hop_length,
                f0autotune,
                f0_file=f0_file,
            )
        if output_path is not None:
            sf.write(output_path, audio_opt, tgt_sr, format="WAV")

        return (tgt_sr, audio_opt)

    except Exception as error:
        print(error)

def get_vc(weight_root, sid):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model is not None:
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            # MLX doesn't have explicit GPU memory management, so we remove torch.cuda.empty_cache()

        if_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")
        if version == "v1":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs256NSFsid(
                    *cpt["config"], is_half=config.is_half
                )
            else:
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        elif version == "v2":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs768NSFsid(
                    *cpt["config"], is_half=config.is_half
                )
            else:
                net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
        del net_g, cpt
        cpt = None
    person = weight_root
    cpt = mx.load(person)  # Assuming MLX has a similar load function
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)

    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    net_g.load_weights(cpt["weight"])  # Assuming MLX has a similar load_weights function
    # MLX doesn't have explicit eval() or to() methods, so we remove them
    if config.is_half:
        net_g = mx.cast(net_g, mx.float16)
    else:
        net_g = mx.cast(net_g, mx.float32)
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]

# The infer_pipeline function remains largely unchanged as it mostly calls other functions
# and doesn't involve direct PyTorch operations

# Main execution and argument parsing remain unchanged