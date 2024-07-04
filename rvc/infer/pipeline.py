import numpy as np
import parselmouth
import mlx.core as mx
import mlx.nn as nn
import sys
import os
import time
from scipy import signal
import pyworld
import librosa
from functools import lru_cache
import random
import gc
import re
from annoy import AnnoyIndex

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.predictor.FCPE import FCPEF0Predictor

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

input_audio_path2wav = {}

@lru_cache
def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    audio = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0

def change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)

    rms1 = mx.array(rms1)
    rms1 = mx.broadcast_to(mx.expand_dims(rms1, 0), (1, data2.shape[0]))

    rms2 = mx.array(rms2)
    rms2 = mx.broadcast_to(mx.expand_dims(rms2, 0), (1, data2.shape[0]))
    rms2 = mx.maximum(rms2, mx.zeros_like(rms2) + 1e-6)

    data2 *= mx.power(rms1, 1 - rate) * mx.power(rms2, rate - 1)
    return data2.numpy()

class VC(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.sr = 16000
        self.window = 160
        self.t_pad = self.sr * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max
        self.device = config.device
        self.ref_freqs = [
            65.41, 82.41, 110.00, 146.83, 196.00, 246.94, 329.63, 440.00, 587.33, 783.99, 1046.50
        ]
        self.note_dict = self.generate_interpolated_frequencies()
        self.vector_index = None
        self.big_npy = None

    def generate_interpolated_frequencies(self):
        note_dict = []
        for i in range(len(self.ref_freqs) - 1):
            freq_low = self.ref_freqs[i]
            freq_high = self.ref_freqs[i + 1]
            interpolated_freqs = np.linspace(freq_low, freq_high, num=10, endpoint=False)
            note_dict.extend(interpolated_freqs)
        note_dict.append(self.ref_freqs[-1])
        return note_dict

    def autotune_f0(self, f0):
        autotuned_f0 = np.zeros_like(f0)
        for i, freq in enumerate(f0):
            closest_note = min(self.note_dict, key=lambda x: abs(x - freq))
            autotuned_f0[i] = closest_note
        return autotuned_f0

    def load_index(self, file_index, vector_size):
        self.vector_index = AnnoyIndex(vector_size, 'angular')
        self.vector_index.load(file_index)
        data_file = file_index.rsplit('.', 1)[0] + '.npy'
        if os.path.exists(data_file):
            self.big_npy = np.load(data_file)
        else:
            print(f"Warning: Data file {data_file} not found. Some functionalities may be limited.")

    def get_f0_crepe_computation(
        self,
        x,
        f0_min,
        f0_max,
        p_len,
        hop_length,
        model="full",
    ):
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        audio = mx.array(x)
        audio = mx.expand_dims(audio, 0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = mx.mean(audio, axis=0, keepdims=True)
        
        # Note: The following part needs to be adapted to use MLX instead of torchcrepe
        # You might need to implement a custom CREPE-like algorithm using MLX
        # or use a different f0 estimation method compatible with MLX
        # For now, I'll leave a placeholder
        
        print("CREPE computation not implemented in MLX. Using placeholder.")
        f0 = np.zeros(p_len)
        return f0

    def get_f0_hybrid_computation(
        self,
        methods_str,
        x,
        f0_min,
        f0_max,
        p_len,
        hop_length,
    ):
        methods_str = re.search("hybrid\[(.+)\]", methods_str)
        if methods_str:
            methods = [method.strip() for method in methods_str.group(1).split("+")]
        f0_computation_stack = []
        print(f"Calculating f0 pitch estimations for methods {str(methods)}")
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        for method in methods:
            f0 = None
            if method == "crepe":
                f0 = self.get_f0_crepe_computation(
                    x, f0_min, f0_max, p_len, int(hop_length)
                )
            elif method == "rmvpe":
                if not hasattr(self, "model_rmvpe"):
                    from rvc.lib.predictor.RMVPE import RMVPE
                    self.model_rmvpe = RMVPE(
                        "rmvpe.pt", is_half=self.is_half, device=self.device
                    )
                f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
                f0 = f0[1:]
            elif method == "fcpe":
                self.model_fcpe = FCPEF0Predictor(
                    "fcpe.pt",
                    f0_min=int(f0_min),
                    f0_max=int(f0_max),
                    dtype=mx.float32,
                    device=self.device,
                    sampling_rate=self.sr,
                    threshold=0.03,
                )
                f0 = self.model_fcpe.compute_f0(x, p_len=p_len)
                del self.model_fcpe
                gc.collect()
            f0_computation_stack.append(f0)

        print(f"Calculating hybrid median f0 from the stack of {str(methods)}")
        f0_computation_stack = [fc for fc in f0_computation_stack if fc is not None]
        f0_median_hybrid = None
        if len(f0_computation_stack) == 1:
            f0_median_hybrid = f0_computation_stack[0]
        else:
            f0_median_hybrid = np.nanmedian(f0_computation_stack, axis=0)
        return f0_median_hybrid

    def get_f0(
        self,
        input_audio_path,
        x,
        p_len,
        f0_up_key,
        f0_method,
        filter_radius,
        hop_length,
        f0autotune,
        inp_f0=None,
    ):
        global input_audio_path2wav
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        if f0_method == "pm":
            f0 = (
                parselmouth.Sound(x, self.sr)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = cache_harvest_f0(input_audio_path, self.sr, f0_max, f0_min, 10)
            if int(filter_radius) > 2:
                f0 = signal.medfilt(f0, 3)
        elif f0_method == "dio":
            f0, t = pyworld.dio(
                x.astype(np.double),
                fs=self.sr,
                f0_ceil=f0_max,
                f0_floor=f0_min,
                frame_period=10,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sr)
            f0 = signal.medfilt(f0, 3)
        elif f0_method == "crepe":
            f0 = self.get_f0_crepe_computation(
                x, f0_min, f0_max, p_len, int(hop_length)
            )
        elif f0_method == "crepe-tiny":
            f0 = self.get_f0_crepe_computation(
                x, f0_min, f0_max, p_len, int(hop_length), "tiny"
            )
        elif f0_method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                from rvc.lib.predictor.RMVPE import RMVPE
                self.model_rmvpe = RMVPE(
                    "rmvpe.pt", is_half=self.is_half, device=self.device
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        elif f0_method == "fcpe":
            self.model_fcpe = FCPEF0Predictor(
                "fcpe.pt",
                f0_min=int(f0_min),
                f0_max=int(f0_max),
                dtype=mx.float32,
                device=self.device,
                sampling_rate=self.sr,
                threshold=0.03,
            )
            f0 = self.model_fcpe.compute_f0(x, p_len=p_len)
            del self.model_fcpe
            gc.collect()
        elif "hybrid" in f0_method:
            input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = self.get_f0_hybrid_computation(
                f0_method,
                x,
                f0_min,
                f0_max,
                p_len,
                hop_length,
            )

        if f0autotune == "True":
            f0 = self.autotune_f0(f0)

        f0 *= pow(2, f0_up_key / 12)
        tf0 = self.sr // self.window
        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
                :shape
            ]
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int)

        return f0_coarse, f0bak

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        index_rate,
        version,
        protect,
    ):
        feats = mx.array(audio0)
        if self.is_half:
            feats = mx.cast(feats, mx.float16)
        else:
            feats = mx.cast(feats, mx.float32)
        if feats.ndim == 2:
            feats = mx.mean(feats, axis=-1)
        assert feats.ndim == 1, feats.ndim
        feats = mx.expand_dims(feats, 0)
        padding_mask = mx.zeros(feats.shape, dtype=mx.bool_)

        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        t0 = time.time()
        logits = model.extract_features(**inputs)
        feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.copy()
        
        if self.vector_index is not None and self.big_npy is not None and index_rate != 0:
            npy = feats[0].numpy()
            if self.is_half:
                npy = npy.astype("float32")
            
            indices, distances = self.vector_index.get_nns_by_vector(npy, 8, include_distances=True)
            weight = np.square(1 / np.array(distances))
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(self.big_npy[indices] * np.expand_dims(weight, axis=2), axis=1)
            
            if self.is_half:
                npy = npy.astype("float16")
            feats = mx.array(npy)[None, :, :] * index_rate + (1 - index_rate) * feats

        feats = mx.transpose(mx.reshape(mx.transpose(feats, (0, 2, 1)), (feats.shape[0], feats.shape[2], -1)), (0, 2, 1))
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = mx.transpose(mx.reshape(mx.transpose(feats0, (0, 2, 1)), (feats0.shape[0], feats0.shape[2], -1)), (0, 2, 1))
        
        t1 = time.time()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.copy()
            pitchff = mx.where(pitchf > 0, 1, protect)
            pitchff = mx.expand_dims(pitchff, -1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = mx.cast(feats, feats0.dtype)
        
        p_len = mx.array([p_len], dtype=mx.int32)
        
        if pitch is not None and pitchf is not None:
            audio1 = net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0].numpy()
        else:
            audio1 = net_g.infer(feats, p_len, sid)[0][0, 0].numpy()
        
        del feats, p_len, padding_mask
        t2 = time.time()
        return audio1

    def pipeline(
        self,
        model,
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
        f0_file=None,
    ):
        if file_index != "" and os.path.exists(file_index) and index_rate != 0:
            try:
                vector_size = model.final_proj.weight.shape[1] if hasattr(model, 'final_proj') else model.proj.shape[1]
                self.load_index(file_index, vector_size)
            except Exception as error:
                print(f"Error loading index: {error}")
                self.vector_index = None
                self.big_npy = None
        else:
            self.vector_index = None
            self.big_npy = None

        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )

        s = 0
        audio_opt = []
        t = None
        t1 = time.time()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name") == True:
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except Exception as error:
                print(error)
        sid = mx.array([sid], dtype=mx.int64)
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(
                input_audio_path,
                audio_pad,
                p_len,
                f0_up_key,
                f0_method,
                filter_radius,
                hop_length,
                f0autotune,
                inp_f0,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if self.device == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch = mx.array(pitch, dtype=mx.int64)[None, :]
            pitchf = mx.array(pitchf, dtype=mx.float32)[None, :]
        t2 = time.time()
        for t in opt_ts:
            t = t // self.window * self.window
            if if_f0 == 1:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        if if_f0 == 1:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid
        return audio_opt

# You may need to add any additional utility functions or classes here if they were present in the original file
