import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn

def init_weights(m, mean=0.0, std=0.01):
    # This function may need to be adapted based on MLX's initialization methods
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight = mx.random.normal(m.weight.shape, mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (mx.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * mx.exp(-2.0 * logs_q)
    return kl

def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = mx.random.uniform(shape) * 0.99998 + 0.00001
    return -mx.log(-mx.log(uniform_samples))

def rand_gumbel_like(x):
    return rand_gumbel(x.shape)

def slice_segments(x, ids_str, segment_size=4):
    ret = mx.zeros((x.shape[0], x.shape[1], segment_size))
    for i in range(x.shape[0]):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret = ret.at[i].set(x[i, :, idx_str:idx_end])
    return ret

def slice_segments2(x, ids_str, segment_size=4):
    ret = mx.zeros((x.shape[0], segment_size))
    for i in range(x.shape[0]):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret = ret.at[i].set(x[i, idx_str:idx_end])
    return ret

def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.shape
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (mx.random.uniform((b,)) * ids_str_max).astype(mx.int32)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = mx.arange(length, dtype=mx.float32)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1)
    inv_timescales = min_timescale * mx.exp(mx.arange(num_timescales, dtype=mx.float32) * -log_timescale_increment)
    scaled_time = mx.expand_dims(position, 0) * mx.expand_dims(inv_timescales, 1)
    signal = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=0)
    signal = mx.pad(signal, [(0, channels % 2), (0, 0)])
    signal = signal.reshape(1, channels, length)
    return signal

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    b, channels, length = x.shape
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal

def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    b, channels, length = x.shape
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return mx.concatenate([x, signal], axis=axis)

def subsequent_mask(length):
    mask = mx.tril(mx.ones((length, length)))
    return mx.expand_dims(mx.expand_dims(mask, 0), 0)

def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = mx.tanh(in_act[:, :n_channels_int, :])
    s_act = mx.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts

def shift_1d(x):
    x = mx.pad(x, [(0, 0), (0, 0), (1, 0)])[:, :, :-1]
    return x

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = mx.arange(max_length)
    return mx.expand_dims(x, 0) < mx.expand_dims(length, 1)

def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = mx.cumsum(duration, -1)
    cum_duration_flat = cum_duration.reshape(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).astype(mask.dtype)
    path = path.reshape(b, t_x, t_y)
    path = path - mx.pad(path, [(0, 0), (1, 0), (0, 0)])[:, :-1]
    path = mx.expand_dims(mx.transpose(mx.expand_dims(path, 1), (0, 1, 3, 2)), 1) * mask
    return path

def clip_grad_value_(parameters, clip_value, norm_type=2):
    # This function may need to be reimplemented based on MLX's gradient clipping methods
    # For now, we'll leave it as a placeholder
    pass

# Note: Some PyTorch-specific functions like F.pad and F.interpolate are not directly available in MLX.
# You may need to implement custom padding and interpolation functions using MLX operations.
