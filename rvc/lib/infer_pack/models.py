import math
import mlx.core as mx
import mlx.nn as nn
from . import modules
from . import attentions
from . import commons
from .commons import init_weights, get_padding
from typing import Optional

# MLX doesn't have direct equivalents for these PyTorch operations
# You may need to implement custom versions or find alternatives
# from torch.nn import Conv1d, ConvTranspose1d, Conv2d
# from torch.nn.utils import remove_weight_norm
# from torch.nn.utils.parametrizations import spectral_norm, weight_norm

def weight_norm(module):
    # Implement weight normalization for MLX
    # This is a placeholder and needs to be implemented
    return module

def remove_weight_norm(module):
    # Implement remove weight normalization for MLX
    # This is a placeholder and needs to be implemented
    pass

def spectral_norm(module):
    # Implement spectral normalization for MLX
    # This is a placeholder and needs to be implemented
    return module

class TextEncoder256(nn.Module):
    def __init__(self, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, f0=True):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.emb_phone = nn.Linear(256, hidden_channels)
        self.lrelu = nn.ReLU()  # MLX doesn't have LeakyReLU, using ReLU as an alternative
        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)
        self.encoder = attentions.Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout))
        self.proj = nn.Conv(hidden_channels, out_channels * 2, kernel_size=1)

    def __call__(self, phone: mx.array, pitch: Optional[mx.array], lengths: mx.array):
        if pitch is None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)
        x = self.lrelu(x)
        x = mx.transpose(x, (0, 2, 1))
        x_mask = mx.expand_dims(commons.sequence_mask(lengths, x.shape[2]), 1)
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = mx.split(stats, 2, axis=1)
        return m, logs, x_mask

class TextEncoder768(nn.Module):
    # Similar implementation as TextEncoder256, but with 768 input channels
    pass

class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = [modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True) for _ in range(n_flows)]
        for i in range(n_flows):
            self.flows.insert(2*i+1, modules.Flip())

    def __call__(self, x: mx.array, x_mask: mx.array, g: Optional[mx.array] = None, reverse: bool = False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

class PosteriorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv(in_channels, hidden_channels, kernel_size=1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv(hidden_channels, out_channels * 2, kernel_size=1)

    def __call__(self, x: mx.array, x_lengths: mx.array, g: Optional[mx.array] = None):
        x_mask = mx.expand_dims(commons.sequence_mask(x_lengths, x.shape[2]), 1)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = mx.split(stats, 2, axis=1)
        z = (m + mx.random.normal(m.shape) * mx.exp(logs)) * x_mask
        return z, m, logs, x_mask

class Generator(nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv(initial_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3)
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(nn.ConvTranspose(
                upsample_initial_channel // (2**i),
                upsample_initial_channel // (2**(i+1)),
                kernel_size=k,
                stride=u,
                padding=(k-u)//2
            )))

        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = nn.Conv(ch, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv(gin_channels, upsample_initial_channel, kernel_size=1)

    def __call__(self, x: mx.array, g: Optional[mx.array] = None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = nn.relu(x)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = nn.relu(x)
        x = self.conv_post(x)
        x = mx.tanh(x)
        return x

class SineGen(nn.Module):
    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0, flag_for_pulse=False):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        uv = mx.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def __call__(self, f0: mx.array, upp: int):
        # Implementation remains similar, but using MLX operations
        pass

class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0, is_half=True):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.is_half = is_half
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def __call__(self, x: mx.array, upp: int = 1):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_wavs = mx.astype(sine_wavs, self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None

class GeneratorNSF(nn.Module):
    # Similar implementation as Generator, but with NSF-specific components
    pass

class SynthesizerTrnMs256NSFsid(nn.Module):
    # Implement the synthesizer using MLX components
    pass

class SynthesizerTrnMs768NSFsid(nn.Module):
    # Similar to SynthesizerTrnMs256NSFsid, but for 768-dimensional input
    pass

class SynthesizerTrnMs256NSFsid_nono(nn.Module):
    # Implement the synthesizer without pitch information
    pass

class SynthesizerTrnMs768NSFsid_nono(nn.Module):
    # Similar to SynthesizerTrnMs256NSFsid_nono, but for 768-dimensional input
    pass

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        periods = [2, 3, 5, 7, 11, 17]
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = discs

    def __call__(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = [
            norm_f(nn.Conv(1, 16, kernel_size=15, stride=1, padding=7)),
            norm_f(nn.Conv(16, 64, kernel_size=41, stride=4, groups=4, padding=20)),
            norm_f(nn.Conv(64, 256, kernel_size=41, stride=4, groups=16, padding=20)),
            norm_f(nn.Conv(256, 1024, kernel_size=41, stride=4, groups=64, padding=20)),
            norm_f(nn.Conv(1024, 1024, kernel_size=41, stride=4, groups=256, padding=20)),
            norm_f(nn.Conv(1024, 1024, kernel_size=5, stride=1, padding=2)),
        ]
        self.conv_post = norm_f(nn.Conv(1024, 1, kernel_size=3, stride=1, padding=1))

    def __call__(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = nn.leaky_relu(x, negative_slope=modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = mx.reshape(x, (x.shape[0], -1))
        return x, fmap



# MLX doesn't have direct equivalents for these PyTorch operations
# You may need to implement custom versions or find alternatives
# from torch.nn import Conv1d, ConvTranspose1d, Conv2d
# from torch.nn.utils import remove_weight_norm
# from torch.nn.utils.parametrizations import spectral_norm, weight_norm

def weight_norm(module):
    # Implement weight normalization for MLX
    # This is a placeholder and needs to be implemented
    return module

def remove_weight_norm(module):
    # Implement remove weight normalization for MLX
    # This is a placeholder and needs to be implemented
    pass

def spectral_norm(module):
    # Implement spectral normalization for MLX
    # This is a placeholder and needs to be implemented
    return module

class TextEncoder256(nn.Module):
    def __init__(self, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, f0=True):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.emb_phone = nn.Linear(256, hidden_channels)
        self.lrelu = nn.ReLU()  # MLX doesn't have LeakyReLU, using ReLU as an alternative
        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)
        self.encoder = attentions.Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout))
        self.proj = nn.Conv(hidden_channels, out_channels * 2, kernel_size=1)

    def __call__(self, phone: mx.array, pitch: Optional[mx.array], lengths: mx.array):
        if pitch is None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)
        x = self.lrelu(x)
        x = mx.transpose(x, (0, 2, 1))
        x_mask = mx.expand_dims(commons.sequence_mask(lengths, x.shape[2]), 1)
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = mx.split(stats, 2, axis=1)
        return m, logs, x_mask

class TextEncoder768(nn.Module):
    # Similar implementation as TextEncoder256, but with 768 input channels
    pass

class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = [modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True) for _ in range(n_flows)]
        for i in range(n_flows):
            self.flows.insert(2*i+1, modules.Flip())

    def __call__(self, x: mx.array, x_mask: mx.array, g: Optional[mx.array] = None, reverse: bool = False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

class PosteriorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv(in_channels, hidden_channels, kernel_size=1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv(hidden_channels, out_channels * 2, kernel_size=1)

    def __call__(self, x: mx.array, x_lengths: mx.array, g: Optional[mx.array] = None):
        x_mask = mx.expand_dims(commons.sequence_mask(x_lengths, x.shape[2]), 1)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = mx.split(stats, 2, axis=1)
        z = (m + mx.random.normal(m.shape) * mx.exp(logs)) * x_mask
        return z, m, logs, x_mask

class Generator(nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv(initial_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3)
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(nn.ConvTranspose(
                upsample_initial_channel // (2**i),
                upsample_initial_channel // (2**(i+1)),
                kernel_size=k,
                stride=u,
                padding=(k-u)//2
            )))

        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = nn.Conv(ch, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv(gin_channels, upsample_initial_channel, kernel_size=1)

    def __call__(self, x: mx.array, g: Optional[mx.array] = None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = nn.relu(x)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = nn.relu(x)
        x = self.conv_post(x)
        x = mx.tanh(x)
        return x

class SineGen(nn.Module):
    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0, flag_for_pulse=False):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        uv = mx.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def __call__(self, f0: mx.array, upp: int):
        # Implementation remains similar, but using MLX operations
        pass

class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0, is_half=True):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.is_half = is_half
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def __call__(self, x: mx.array, upp: int = 1):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_wavs = mx.astype(sine_wavs, self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None

class GeneratorNSF(nn.Module):
    # Similar implementation as Generator, but with NSF-specific components
    pass

class SynthesizerTrnMs256NSFsid(nn.Module):
    # Implement the synthesizer using MLX components
    pass

class SynthesizerTrnMs768NSFsid(nn.Module):
    # Similar to SynthesizerTrnMs256NSFsid, but for 768-dimensional input
    pass

class SynthesizerTrnMs256NSFsid_nono(nn.Module):
    # Implement the synthesizer without pitch information
    pass

class SynthesizerTrnMs768NSFsid_nono(nn.Module):
    # Similar to SynthesizerTrnMs256NSFsid_nono, but for 768-dimensional input
    pass

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        periods = [2, 3, 5, 7, 11, 17]
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = discs

    def __call__(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = [
            norm_f(nn.Conv(1, 16, kernel_size=15, stride=1, padding=7)),
            norm_f(nn.Conv(16, 64, kernel_size=41, stride=4, groups=4, padding=20)),
            norm_f(nn.Conv(64, 256, kernel_size=41, stride=4, groups=16, padding=20)),
            norm_f(nn.Conv(256, 1024, kernel_size=41, stride=4, groups=64, padding=20)),
            norm_f(nn.Conv(1024, 1024, kernel_size=41, stride=4, groups=256, padding=20)),
            norm_f(nn.Conv(1024, 1024, kernel_size=5, stride=1, padding=2)),
        ]
        self.conv_post = norm_f(nn.Conv(1024, 1, kernel_size=3, stride=1, padding=1))

    def __call__(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = nn.leaky_relu(x, negative_slope=modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = mx.reshape(x, (x.shape[0], -1))
        return x, fmap

class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = [
            norm_f(nn.Conv(1, 32, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv(32, 128, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv(128, 512, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv(512, 1024, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv(1024, 1024, kernel_size=(kernel_size, 1), stride=1, padding=(get_padding(kernel_size, 1), 0))),
        ]
        self.conv_post = norm_f(nn.Conv(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0)))

    def __call__(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = mx.pad(x, [(0, 0), (0, 0), (0, n_pad)], mode='reflect')
            t = t + n_pad
        x = mx.reshape(x, (b, c, t // self.period, self.period))

        for l in self.convs:
            x = l(x)
            x = nn.leaky_relu(x, negative_slope=modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = mx.reshape(x, (x.shape[0], -1))

        return x, fmap

class MultiPeriodDiscriminatorV2(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        periods = [2, 3, 5, 7, 11, 17, 23, 37]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = discs

    def __call__(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

# Utility functions

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

# Additional MLX-specific implementations

def weight_norm(module):
    # Implement weight normalization for MLX
    # This is a placeholder and needs to be implemented
    return module

def remove_weight_norm(module):
    # Implement remove weight normalization for MLX
    # This is a placeholder and needs to be implemented
    pass

def spectral_norm(module):
    # Implement spectral normalization for MLX
    # This is a placeholder and needs to be implemented
    return module

# Note: The following PyTorch-specific concepts don't have direct MLX equivalents
# and may need custom implementations or alternatives:
# - torch.jit.ignore
# - torch.jit.export
# - __prepare_scriptable__
# - remove_weight_norm
# - Conv1d, ConvTranspose1d, Conv2d (replaced with nn.Conv and nn.ConvTranspose)
# - F.leaky_relu (replaced with nn.leaky_relu)
# - torch.randn_like (replaced with mx.random.normal)
# - torch.exp (replaced with mx.exp)
# - torch.unsqueeze (replaced with mx.expand_dims)
# - torch.flatten (replaced with mx.reshape)
# - torch.split (replaced with mx.split)
# - F.pad (replaced with mx.pad)

# The SynthesizerTrn classes (SynthesizerTrnMs256NSFsid, SynthesizerTrnMs768NSFsid, 
# SynthesizerTrnMs256NSFsid_nono, SynthesizerTrnMs768NSFsid_nono) should be implemented
# similarly to their PyTorch counterparts, using MLX operations and modules.

# Remember to adapt the forward and infer methods of these classes to use MLX operations.

# Example structure for SynthesizerTrnMs256NSFsid:

class SynthesizerTrnMs256NSFsid(nn.Module):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout, resblock, 
                 resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, 
                 upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim,
                 gin_channels, sr, **kwargs):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        
        self.enc_p = TextEncoder256(inter_channels, hidden_channels, filter_channels,
                                    n_heads, n_layers, kernel_size, p_dropout)
        self.dec = GeneratorNSF(inter_channels, resblock, resblock_kernel_sizes, 
                                resblock_dilation_sizes, upsample_rates, 
                                upsample_initial_channel, upsample_kernel_sizes,
                                gin_channels=gin_channels, sr=sr, is_half=kwargs["is_half"])
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 
                                      5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, 
                                          gin_channels=gin_channels)
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        
    def __call__(self, phone, phone_lengths, pitch, pitchf, y, y_lengths, ds):
        g = self.emb_g(ds)
        g = mx.expand_dims(g, -1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        pitchf = commons.slice_segments2(pitchf, ids_slice, self.segment_size)
        o = self.dec(z_slice, pitchf, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, phone, phone_lengths, pitch, nsff0, sid, rate=None):
        g = self.emb_g(sid)
        g = mx.expand_dims(g, -1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = m_p + mx.exp(logs_p) * mx.random.normal(m_p.shape) * 0.66666
        z_p = z_p * x_mask
        if rate is not None:
            head = int(z_p.shape[2] * (1 - rate))
            z_p = z_p[:, :, head:]
            x_mask = x_mask[:, :, head:]
            nsff0 = nsff0[:, head:]
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, nsff0, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)

# Implement other Synthesizer classes similarly
# Continuing from the previous implementation...

class SynthesizerTrnMs768NSFsid(nn.Module):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout, resblock, 
                 resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, 
                 upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim,
                 gin_channels, sr, **kwargs):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        
        self.enc_p = TextEncoder768(inter_channels, hidden_channels, filter_channels,
                                    n_heads, n_layers, kernel_size, p_dropout)
        self.dec = GeneratorNSF(inter_channels, resblock, resblock_kernel_sizes, 
                                resblock_dilation_sizes, upsample_rates, 
                                upsample_initial_channel, upsample_kernel_sizes,
                                gin_channels=gin_channels, sr=sr, is_half=kwargs["is_half"])
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 
                                      5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, 
                                          gin_channels=gin_channels)
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        
    def __call__(self, phone, phone_lengths, pitch, pitchf, y, y_lengths, ds):
        g = self.emb_g(ds)
        g = mx.expand_dims(g, -1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        pitchf = commons.slice_segments2(pitchf, ids_slice, self.segment_size)
        o = self.dec(z_slice, pitchf, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, phone, phone_lengths, pitch, nsff0, sid, rate=None):
        g = self.emb_g(sid)
        g = mx.expand_dims(g, -1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = m_p + mx.exp(logs_p) * mx.random.normal(m_p.shape) * 0.66666
        z_p = z_p * x_mask
        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate))
            z_p = z_p[:, :, head:]
            x_mask = x_mask[:, :, head:]
            nsff0 = nsff0[:, head:]
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, nsff0, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)

class SynthesizerTrnMs256NSFsid_nono(nn.Module):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout, resblock, 
                 resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, 
                 upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim,
                 gin_channels, **kwargs):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        
        self.enc_p = TextEncoder256(inter_channels, hidden_channels, filter_channels,
                                    n_heads, n_layers, kernel_size, p_dropout, f0=False)
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, 
                             resblock_dilation_sizes, upsample_rates, 
                             upsample_initial_channel, upsample_kernel_sizes,
                             gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 
                                      5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, 
                                          gin_channels=gin_channels)
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        
    def __call__(self, phone, phone_lengths, y, y_lengths, ds):
        g = self.emb_g(ds)
        g = mx.expand_dims(g, -1)
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, phone, phone_lengths, sid, rate=None):
        g = self.emb_g(sid)
        g = mx.expand_dims(g, -1)
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z_p = m_p + mx.exp(logs_p) * mx.random.normal(m_p.shape) * 0.66666
        z_p = z_p * x_mask
        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate))
            z_p = z_p[:, :, head:]
            x_mask = x_mask[:, :, head:]
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)

class SynthesizerTrnMs768NSFsid_nono(nn.Module):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout, resblock, 
                 resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, 
                 upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim,
                 gin_channels, **kwargs):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        
        self.enc_p = TextEncoder768(inter_channels, hidden_channels, filter_channels,
                                    n_heads, n_layers, kernel_size, p_dropout, f0=False)
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, 
                             resblock_dilation_sizes, upsample_rates, 
                             upsample_initial_channel, upsample_kernel_sizes,
                             gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 
                                      5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, 
                                          gin_channels=gin_channels)
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        
    def __call__(self, phone, phone_lengths, y, y_lengths, ds):
        g = self.emb_g(ds)
        g = mx.expand_dims(g, -1)
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, phone, phone_lengths, sid, rate=None):
        g = self.emb_g(sid)
        g = mx.expand_dims(g, -1)
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z_p = m_p + mx.exp(logs_p) * mx.random.normal(m_p.shape) * 0.66666
        z_p = z_p * x_mask
        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate))
            z_p = z_p[:, :, head:]
            x_mask = x_mask[:, :, head:]
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)

# Implement weight normalization for MLX
def weight_norm(module, name='weight', dim=0):
    if hasattr(module, name):
        weight = getattr(module, name)
        g = mx.Variable(mx.sqrt(mx.sum(weight**2, axis=dim, keepdims=True)))
        weight = weight * g / mx.sqrt(mx.sum(weight**2, axis=dim, keepdims=True))
        setattr(module, name, weight)
        setattr(module, name + '_g', g)
    return module

# Implement remove weight normalization for MLX
def remove_weight_norm(module, name='weight'):
    if hasattr(module, name + '_g'):
        delattr(module, name + '_g')
    return module

# Implement spectral normalization for MLX
def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        dim = 0 if len(module.weight.shape) > 1 else None
    
    weight = getattr(module, name)
    u = mx.random.normal(shape=(1, weight.shape[0]))
    
    def power_iteration(weight, u, n_iterations):
        for _ in range(n_iterations):
            v = mx.normalize(mx.matmul(u, weight), eps=eps)
            u = mx.normalize(mx.matmul(v, weight.transpose()), eps=eps)
        return u, v
    
    u, v = power_iteration(weight, u, n_power_iterations)
    sigma = mx.sum(u * mx.matmul(weight, v.transpose()))
    weight = weight / sigma
    setattr(module, name, weight)
    setattr(module, name + '_u', u)
    setattr(module, name + '_v', v)
    setattr(module, name + '_sigma', sigma)
    return module

# Additional utility functions

def init_weights(m, mean=0.0, std=0.01):
    if isinstance(m, (nn.Conv, nn.ConvTranspose, nn.Linear)):
        m.weight = mx.random.normal(m.weight.shape, mean, std)
        if m.bias is not None:
            m.bias = mx.zeros(m.bias.shape)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

# MLX doesn't have a direct equivalent to torch.nn.utils.parametrizations
# You may need to implement custom versions of these if needed in your model

# Remember to adapt