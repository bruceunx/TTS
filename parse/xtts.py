import sys
from pathlib import Path
import functools
from collections import namedtuple
import math
import random
from typing import Any
import os
from functools import cached_property
import re
from packaging import version
from dataclasses import dataclass, field, fields

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tokenizers import Tokenizer
from num2words import num2words

from transformers import GPT2Config, GPT2Model

from transformers import GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from spacy.lang.en import English
import textwrap

# import fsspec

LRELU_SLOPE = 0.1


def split_sentence(text, lang, text_split_length=250):
    """Preprocess the input text"""
    text_splits = []
    if text_split_length is not None and len(text) >= text_split_length:
        text_splits.append("")
        nlp = English()
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        for sentence in doc.sents:
            if len(text_splits[-1]) + len(str(sentence)) <= text_split_length:
                # if the last sentence + the current sentence is less than the text_split_length
                # then add the current sentence to the last sentence
                text_splits[-1] += " " + str(sentence)
                text_splits[-1] = text_splits[-1].lstrip()
            elif len(str(sentence)) > text_split_length:
                # if the current sentence is greater than the text_split_length
                for line in textwrap.wrap(
                        str(sentence),
                        width=text_split_length,
                        drop_whitespace=True,
                        break_on_hyphens=False,
                        tabsize=1,
                ):
                    text_splits.append(str(line))
            else:
                text_splits.append(str(sentence))

        if len(text_splits) > 1:
            if text_splits[0] == "":
                del text_splits[0]
    else:
        text_splits = [text.lstrip()]

    return text_splits


def wav_to_mel_cloning(
    wav,
    mel_norms_file="../experiments/clips_mel_norms.pth",
    mel_norms=None,
    device=torch.device("cpu"),
    n_fft=4096,
    hop_length=1024,
    win_length=4096,
    power=2,
    normalized=False,
    sample_rate=22050,
    f_min=0,
    f_max=8000,
    n_mels=80,
):
    """
    Convert waveform to mel-spectrogram with hard-coded parameters for cloning.

    Args:
        wav (torch.Tensor): Input waveform tensor.
        mel_norms_file (str): Path to mel-spectrogram normalization file.
        mel_norms (torch.Tensor): Mel-spectrogram normalization tensor.
        device (torch.device): Device to use for computation.

    Returns:
        torch.Tensor: Mel-spectrogram tensor.
    """
    mel_stft = torchaudio.transforms.MelSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        normalized=normalized,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        norm="slaney",
    ).to(device)
    wav = wav.to(device)
    mel = mel_stft(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    if mel_norms is None:
        mel_norms = torch.load(mel_norms_file, map_location=device)
    mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
    return mel


def get_user_data_dir(appname):
    TTS_HOME = os.environ.get("TTS_HOME")
    XDG_DATA_HOME = os.environ.get("XDG_DATA_HOME")
    if TTS_HOME is not None:
        ans = Path(TTS_HOME).expanduser().resolve(strict=False)
    elif XDG_DATA_HOME is not None:
        ans = Path(XDG_DATA_HOME).expanduser().resolve(strict=False)
    elif sys.platform == "win32":
        import winreg  # pylint: disable=import-outside-toplevel

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
        )
        dir_, _ = winreg.QueryValueEx(key, "Local AppData")
        ans = Path(dir_).resolve(strict=False)
    elif sys.platform == "darwin":
        ans = Path("~/Library/Application Support/").expanduser()
    else:
        ans = Path.home().joinpath(".local/share")
    return ans.joinpath(appname)


def load_fsspec(
    path,
    map_location,
    cache=True,
    **kwargs,
):
    """Like torch.load but can load from other locations (e.g. s3:// , gs://).

    Args:
        path: Any path or url supported by fsspec.
        map_location: torch.device or str.
        cache: If True, cache a remote file locally for subsequent calls. It is cached under `get_user_data_dir()/tts_cache`. Defaults to True.
        **kwargs: Keyword arguments forwarded to torch.load.

    Returns:
        Object stored in path.
    """
    with open(path, "rb") as f:
        return torch.load(f, map_location=map_location, **kwargs)


def get_padding(k, d):
    return int((k * d - d) / 2)


class ResBlock1(torch.nn.Module):
    """Residual Block Type 1. It has 3 convolutional layers in each convolutional block.

    Network::

        x -> lrelu -> conv1_1 -> conv1_2 -> conv1_3 -> z -> lrelu -> conv2_1 -> conv2_2 -> conv2_3 -> o -> + -> o
        |--------------------------------------------------------------------------------------------------|


    Args:
        channels (int): number of hidden channels for the convolutional layers.
        kernel_size (int): size of the convolution filter in each layer.
        dilations (list): list of dilation value for each conv layer in a block.
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                )),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                )),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=get_padding(kernel_size, dilation[2]),
                )),
        ])

        self.convs2 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )),
        ])

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            Tensor: output tensor.
        Shapes:
            x: [B, C, T]
        """
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_parametrizations(l, "weight")
        for l in self.convs2:
            remove_parametrizations(l, "weight")


class ResBlock2(torch.nn.Module):
    """Residual Block Type 2. It has 1 convolutional layers in each convolutional block.

    Network::

        x -> lrelu -> conv1-> -> z -> lrelu -> conv2-> o -> + -> o
        |---------------------------------------------------|


    Args:
        channels (int): number of hidden channels for the convolutional layers.
        kernel_size (int): size of the convolution filter in each layer.
        dilations (list): list of dilation value for each conv layer in a block.
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                )),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                )),
        ])

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_parametrizations(l, "weight")


class HifiganGenerator(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        resblock_type,
        resblock_dilation_sizes,
        resblock_kernel_sizes,
        upsample_kernel_sizes,
        upsample_initial_channel,
        upsample_factors,
        inference_padding=5,
        cond_channels=0,
        conv_pre_weight_norm=True,
        conv_post_weight_norm=True,
        conv_post_bias=True,
        cond_in_each_up_layer=False,
    ):
        r"""HiFiGAN Generator with Multi-Receptive Field Fusion (MRF)

        Network:
            x -> lrelu -> upsampling_layer -> resblock1_k1x1 -> z1 -> + -> z_sum / #resblocks -> lrelu -> conv_post_7x1 -> tanh -> o
                                                 ..          -> zI ---|
                                              resblockN_kNx1 -> zN ---'

        Args:
            in_channels (int): number of input tensor channels.
            out_channels (int): number of output tensor channels.
            resblock_type (str): type of the `ResBlock`. '1' or '2'.
            resblock_dilation_sizes (List[List[int]]): list of dilation values in each layer of a `ResBlock`.
            resblock_kernel_sizes (List[int]): list of kernel sizes for each `ResBlock`.
            upsample_kernel_sizes (List[int]): list of kernel sizes for each transposed convolution.
            upsample_initial_channel (int): number of channels for the first upsampling layer. This is divided by 2
                for each consecutive upsampling layer.
            upsample_factors (List[int]): upsampling factors (stride) for each upsampling layer.
            inference_padding (int): constant padding applied to the input at inference time. Defaults to 5.
        """
        super().__init__()
        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        self.cond_in_each_up_layer = cond_in_each_up_layer

        # initial upsampling layers
        self.conv_pre = weight_norm(
            Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2
        # upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_factors,
                                       upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2**(i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )))
        # MRF blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i + 1))
            for _, (k, d) in enumerate(
                    zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))
        # post convolution layer
        self.conv_post = weight_norm(
            Conv1d(ch, out_channels, 7, 1, padding=3, bias=conv_post_bias))
        if cond_channels > 0:
            self.cond_layer = nn.Conv1d(cond_channels,
                                        upsample_initial_channel, 1)

        if not conv_pre_weight_norm:
            remove_parametrizations(self.conv_pre, "weight")

        if not conv_post_weight_norm:
            remove_parametrizations(self.conv_post, "weight")

        if self.cond_in_each_up_layer:
            self.conds = nn.ModuleList()
            for i in range(len(self.ups)):
                ch = upsample_initial_channel // (2**(i + 1))
                self.conds.append(nn.Conv1d(cond_channels, ch, 1))

    def forward(self, x, g=None):
        """
        Args:
            x (Tensor): feature input tensor.
            g (Tensor): global conditioning input tensor.

        Returns:
            Tensor: output waveform.

        Shapes:
            x: [B, C, T]
            Tensor: [B, 1, T]
        """
        o = self.conv_pre(x)
        if hasattr(self, "cond_layer"):
            o = o + self.cond_layer(g)
        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)

            if self.cond_in_each_up_layer:
                o = o + self.conds[i](g)

            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i * self.num_kernels + j](o)
                else:
                    z_sum += self.resblocks[i * self.num_kernels + j](o)
            o = z_sum / self.num_kernels
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    @torch.no_grad()
    def inference(self, c):
        """
        Args:
            x (Tensor): conditioning input tensor.

        Returns:
            Tensor: output waveform.

        Shapes:
            x: [B, C, T]
            Tensor: [B, 1, T]
        """
        c = c.to(self.conv_pre.weight.device)
        c = torch.nn.functional.pad(
            c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_parametrizations(l, "weight")
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_parametrizations(self.conv_pre, "weight")
        remove_parametrizations(self.conv_post, "weight")

    def load_checkpoint(self,
                        config,
                        checkpoint_path,
                        eval=False,
                        cache=False):  # pylint: disable=unused-argument, redefined-builtin
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            assert not self.training
            self.remove_weight_norm()


class SELayer(nn.Module):

    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


def set_init_dict(model_dict, checkpoint_state, c):
    # Partial initialization: if there is a mismatch with new and old layer, it is skipped.
    for k, v in checkpoint_state.items():
        if k not in model_dict:
            print(" | > Layer missing in the model definition: {}".format(k))
    # 1. filter out unnecessary keys
    pretrained_dict = {
        k: v
        for k, v in checkpoint_state.items() if k in model_dict
    }
    # 2. filter out different size layers
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if v.numel() == model_dict[k].numel()
    }
    # 3. skip reinit layers
    if c.has("reinit_layers") and c.reinit_layers is not None:
        for reinit_layer_name in c.reinit_layers:
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if reinit_layer_name not in k
            }
    # 4. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    print(" | > {} / {} layers are restored.".format(len(pretrained_dict),
                                                     len(model_dict)))
    return model_dict


class PreEmphasis(nn.Module):

    def __init__(self, coefficient=0.97):
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer(
            "filter",
            torch.FloatTensor([-self.coefficient,
                               1.0]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        assert len(x.size()) == 2

        x = torch.nn.functional.pad(x.unsqueeze(1), (1, 0), "reflect")
        return torch.nn.functional.conv1d(x, self.filter).squeeze(1)


class ResNetSpeakerEncoder(nn.Module):
    """This is copied from 🐸TTS to remove it from the dependencies."""

    # pylint: disable=W0102
    def __init__(
        self,
        input_dim=64,
        proj_dim=512,
        layers=[3, 4, 6, 3],
        num_filters=[32, 64, 128, 256],
        encoder_type="ASP",
        log_input=False,
        use_torch_spec=False,
        audio_config=None,
    ):
        super(ResNetSpeakerEncoder, self).__init__()

        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.log_input = log_input
        self.use_torch_spec = use_torch_spec
        self.audio_config = audio_config
        self.proj_dim = proj_dim

        self.conv1 = nn.Conv2d(1,
                               num_filters[0],
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.inplanes = num_filters[0]
        self.layer1 = self.create_layer(SEBasicBlock, num_filters[0],
                                        layers[0])
        self.layer2 = self.create_layer(SEBasicBlock,
                                        num_filters[1],
                                        layers[1],
                                        stride=(2, 2))
        self.layer3 = self.create_layer(SEBasicBlock,
                                        num_filters[2],
                                        layers[2],
                                        stride=(2, 2))
        self.layer4 = self.create_layer(SEBasicBlock,
                                        num_filters[3],
                                        layers[3],
                                        stride=(2, 2))

        self.instancenorm = nn.InstanceNorm1d(input_dim)

        if self.use_torch_spec:
            self.torch_spec = torch.nn.Sequential(
                PreEmphasis(audio_config["preemphasis"]),
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=audio_config["sample_rate"],
                    n_fft=audio_config["fft_size"],
                    win_length=audio_config["win_length"],
                    hop_length=audio_config["hop_length"],
                    window_fn=torch.hamming_window,
                    n_mels=audio_config["num_mels"],
                ),
            )

        else:
            self.torch_spec = None

        outmap_size = int(self.input_dim / 8)

        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError("Undefined encoder")

        self.fc = nn.Linear(out_dim, proj_dim)

        self._init_layers()

    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def create_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # pylint: disable=R0201
    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x, l2_norm=False):
        """Forward pass of the model.

        Args:
            x (Tensor): Raw waveform signal or spectrogram frames. If input is a waveform, `torch_spec` must be `True`
                to compute the spectrogram on-the-fly.
            l2_norm (bool): Whether to L2-normalize the outputs.

        Shapes:
            - x: :math:`(N, 1, T_{in})` or :math:`(N, D_{spec}, T_{in})`
        """
        x.squeeze_(1)
        # if you torch spec compute it otherwise use the mel spec computed by the AP
        if self.use_torch_spec:
            x = self.torch_spec(x)

        if self.log_input:
            x = (x + 1e-6).log()
        x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size()[0], -1, x.size()[-1])

        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum(
                (x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, sg), 1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        if l2_norm:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

    def load_checkpoint(
        self,
        checkpoint_path: str,
        eval: bool = False,
        use_cuda: bool = False,
        criterion=None,
        cache=False,
    ):
        state = load_fsspec(checkpoint_path,
                            map_location=torch.device("cpu"),
                            cache=cache)
        try:
            self.load_state_dict(state["model"])
            print(" > Model fully restored. ")
        except (KeyError, RuntimeError) as error:
            # If eval raise the error
            if eval:
                raise error

            print(" > Partial model initialization.")
            model_dict = self.state_dict()
            model_dict = set_init_dict(model_dict, state["model"])
            self.load_state_dict(model_dict)
            del model_dict

        # load the criterion for restore_path
        if criterion is not None and "criterion" in state:
            try:
                criterion.load_state_dict(state["criterion"])
            except (KeyError, RuntimeError) as error:
                print(" > Criterion load ignored because of:", error)

        if use_cuda:
            self.cuda()
            if criterion is not None:
                criterion = criterion.cuda()

        if eval:
            self.eval()
            assert not self.training

        if not eval:
            return criterion, state["step"]
        return criterion


class HifiDecoder(torch.nn.Module):

    def __init__(
        self,
        input_sample_rate=22050,
        output_sample_rate=24000,
        output_hop_length=256,
        ar_mel_length_compression=1024,
        decoder_input_dim=1024,
        resblock_type_decoder="1",
        resblock_dilation_sizes_decoder=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        resblock_kernel_sizes_decoder=[3, 7, 11],
        upsample_rates_decoder=[8, 8, 2, 2],
        upsample_initial_channel_decoder=512,
        upsample_kernel_sizes_decoder=[16, 16, 4, 4],
        d_vector_dim=512,
        cond_d_vector_in_each_upsampling_layer=True,
        speaker_encoder_audio_config={
            "fft_size": 512,
            "win_length": 400,
            "hop_length": 160,
            "sample_rate": 16000,
            "preemphasis": 0.97,
            "num_mels": 64,
        },
    ):
        super().__init__()
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.output_hop_length = output_hop_length
        self.ar_mel_length_compression = ar_mel_length_compression
        self.speaker_encoder_audio_config = speaker_encoder_audio_config
        self.waveform_decoder = HifiganGenerator(
            decoder_input_dim,
            1,
            resblock_type_decoder,
            resblock_dilation_sizes_decoder,
            resblock_kernel_sizes_decoder,
            upsample_kernel_sizes_decoder,
            upsample_initial_channel_decoder,
            upsample_rates_decoder,
            inference_padding=0,
            cond_channels=d_vector_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
            cond_in_each_up_layer=cond_d_vector_in_each_upsampling_layer,
        )
        self.speaker_encoder = ResNetSpeakerEncoder(
            input_dim=64,
            proj_dim=512,
            log_input=True,
            use_torch_spec=True,
            audio_config=speaker_encoder_audio_config,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, latents, g=None):
        """
        Args:
            x (Tensor): feature input tensor (GPT latent).
            g (Tensor): global conditioning input tensor.

        Returns:
            Tensor: output waveform.

        Shapes:
            x: [B, C, T]
            Tensor: [B, 1, T]
        """

        z = torch.nn.functional.interpolate(
            latents.transpose(1, 2),
            scale_factor=[
                self.ar_mel_length_compression / self.output_hop_length
            ],
            mode="linear",
        ).squeeze(1)
        # upsample to the right sr
        if self.output_sample_rate != self.input_sample_rate:
            z = torch.nn.functional.interpolate(
                z,
                scale_factor=[
                    self.output_sample_rate / self.input_sample_rate
                ],
                mode="linear",
            ).squeeze(0)
        o = self.waveform_decoder(z, g=g)
        return o

    @torch.no_grad()
    def inference(self, c, g):
        """
        Args:
            x (Tensor): feature input tensor (GPT latent).
            g (Tensor): global conditioning input tensor.

        Returns:
            Tensor: output waveform.

        Shapes:
            x: [B, C, T]
            Tensor: [B, 1, T]
        """
        return self.forward(c, g=g)

    def load_checkpoint(self, checkpoint_path, eval=False):  # pylint: disable=unused-argument, redefined-builtin
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"))
        # remove unused keys
        state = state["model"]
        states_keys = list(state.keys())
        for key in states_keys:
            if "waveform_decoder." not in key and "speaker_encoder." not in key:
                del state[key]

        self.load_state_dict(state)
        if eval:
            self.eval()
            assert not self.training
            self.waveform_decoder.remove_weight_norm()


class GPT2InferenceModel(GPT2PreTrainedModel):
    """Override GPT2LMHeadModel to allow for prefix conditioning."""

    def __init__(self, config, gpt, pos_emb, embeddings, norm, linear,
                 kv_cache):
        super().__init__(config)
        self.transformer = gpt
        self.pos_embedding = pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache

    def store_prefix_emb(self, prefix_emb):
        self.cached_prefix_emb = prefix_emb

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        if not self.kv_cache:
            past_key_values = None

        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert self.cached_prefix_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # assert len(past_key_values) + len(input_ids) == attention_mask.shape[1]

        # Create embedding
        prefix_len = self.cached_prefix_emb.shape[1]
        if input_ids.shape[1] != 1:
            gen_inputs = input_ids[:, prefix_len:]
            gen_emb = self.embeddings(gen_inputs)
            gen_emb = gen_emb + self.pos_embedding(gen_emb)
            if self.cached_prefix_emb.shape[0] != gen_emb.shape[0]:
                prefix_emb = self.cached_prefix_emb.repeat_interleave(
                    gen_emb.shape[0] // self.cached_prefix_emb.shape[0], 0)
            else:
                prefix_emb = self.cached_prefix_emb.to(gen_emb.dtype)
            emb = torch.cat([prefix_emb, gen_emb], dim=1)
        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.pos_embedding.get_fixed_embedding(
                attention_mask.shape[1] -
                (prefix_len + 1), attention_mask.device)
        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits, ) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past) for layer_past in past)


def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))


class GroupNorm32(nn.GroupNorm):

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def normalization(channels):
    groups = 32
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    while channels % groups != 0:
        groups = int(groups / 2)
    assert groups > 2
    return GroupNorm32(groups, channels)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()


class QKVAttention(nn.Module):

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None, qk_bias=0):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch,
                                                                       dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale,
            k * scale)  # More stable with f16 than dividing afterwards
        weight = weight + qk_bias
        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)
            weight[mask.logical_not()] = -torch.inf
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """An attention block that allows spatial positions to attend to each other."""

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        out_channels=None,
        do_activation=False,
    ):
        super().__init__()
        self.channels = channels
        out_channels = channels if out_channels is None else out_channels
        self.do_activation = do_activation
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, out_channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        self.x_proj = nn.Identity() if out_channels == channels else conv_nd(
            1, channels, out_channels, 1)
        self.proj_out = zero_module(conv_nd(1, out_channels, out_channels, 1))

    def forward(self, x, mask=None, qk_bias=0):
        b, c, *spatial = x.shape
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0).repeat(x.shape[0], 1, 1)
            if mask.shape[1] != x.shape[-1]:
                mask = mask[:, :x.shape[-1], :x.shape[-1]]

        x = x.reshape(b, c, -1)
        x = self.norm(x)
        if self.do_activation:
            x = F.silu(x, inplace=True)
        qkv = self.qkv(x)
        h = self.attention(qkv, mask=mask, qk_bias=qk_bias)
        h = self.proj_out(h)
        xp = self.x_proj(x)
        return (xp + h).reshape(b, xp.shape[1], *spatial)


class ConditioningEncoder(nn.Module):

    def __init__(
        self,
        spec_dim,
        embedding_dim,
        attn_blocks=6,
        num_attn_heads=4,
    ):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim

    def forward(self, x):
        """
        x: (b, 80, s)
        """
        h = self.init(x)
        h = self.attn(h)
        return h


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim),
                       device=range.device)


class LearnedPositionEmbeddings(nn.Module):

    def __init__(self, seq_len, model_dim, init=0.02, relative=False):
        super().__init__()
        # nn.Embedding
        self.emb = torch.nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)
        self.relative = relative
        self.seq_len = seq_len

    def forward(self, x):
        sl = x.shape[1]
        if self.relative:
            start = random.randint(sl, self.seq_len) - sl
            return self.emb(torch.arange(start, start + sl, device=x.device))
        else:
            return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


def build_hf_gpt_transformer(
    layers,
    model_dim,
    heads,
    max_mel_seq_len,
    max_text_seq_len,
    max_prompt_len,
    checkpointing,
):
    """
    GPT-2 implemented by the HuggingFace library.
    """

    gpt_config = GPT2Config(
        vocab_size=256,  # Unused.
        n_positions=max_mel_seq_len + max_text_seq_len + max_prompt_len,
        n_ctx=max_mel_seq_len + max_text_seq_len + max_prompt_len,
        n_embd=model_dim,
        n_layer=layers,
        n_head=heads,
        gradient_checkpointing=checkpointing,
        use_cache=not checkpointing,
    )
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte

    mel_pos_emb = (LearnedPositionEmbeddings(max_mel_seq_len, model_dim)
                   if max_mel_seq_len != -1 else functools.partial(
                       null_position_embeddings, dim=model_dim))
    text_pos_emb = (LearnedPositionEmbeddings(max_text_seq_len, model_dim)
                    if max_mel_seq_len != -1 else functools.partial(
                        null_position_embeddings, dim=model_dim))
    # gpt = torch.compile(gpt, mode="reduce-overhead", fullgraph=True)
    return gpt, mel_pos_emb, text_pos_emb, None, None


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class RMSNorm(nn.Module):

    def __init__(self, dim, scale=True, dim_cond=None):
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = nn.Linear(dim_cond, dim *
                                       2) if self.cond else None

        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x, cond=None):
        gamma = default(self.gamma, 1)
        out = F.normalize(x, dim=-1) * self.scale * gamma

        if not self.cond:
            return out

        assert exists(cond)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        gamma, beta = map(lambda t: rearrange(t, "b d -> b 1 d"),
                          (gamma, beta))
        return out * gamma + beta


class Attend(nn.Module):

    def __init__(self, dropout=0.0, causal=False, use_flash=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash = use_flash
        assert not (
            use_flash
            and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        # determine efficient attention configs for cuda and cpu
        self.config = namedtuple(
            "EfficientAttentionConfig",
            ["enable_flash", "enable_math", "enable_mem_efficient"])
        self.cpu_config = self.config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(
            torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            print(
                "A100 GPU detected, using flash attention if input tensor is on cuda"
            )
            self.cuda_config = self.config(True, False, False)
        else:
            print(
                "Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda"
            )
            self.cuda_config = self.config(False, True, True)

    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = rearrange(k, "b ... -> b 1 ...").expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, "b ... -> b 1 ...").expand_as(q)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal)

        return out

    def forward(self, q, k, v, mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device = q.shape[-2], q.device

        scale = q.shape[-1]**-0.5

        if self.use_flash:
            return self.flash_attn(q, k, v, mask=mask)

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        # similarity

        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # key padding mask

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        *,
        dim_context=None,
        causal=False,
        dim_head=64,
        heads=8,
        dropout=0.0,
        use_flash=False,
        cross_attn_include_queries=False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attend = Attend(causal=causal,
                             dropout=dropout,
                             use_flash=use_flash)
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x, context=None, mask=None):
        h, has_context = self.heads, exists(context)

        context = default(context, x)

        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim=-2)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
                      (q, k, v))

        out = self.attend(q, k, v, mask=mask)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CausalConv1d(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (kernel_size, ) = self.kernel_size
        (dilation, ) = self.dilation
        (stride, ) = self.stride

        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        causal_padded_x = F.pad(x, (self.causal_padding, 0), value=0.0)
        return super().forward(causal_padded_x)


class GEGLU(nn.Module):

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, causal_conv=False):
    dim_inner = int(dim * mult * 2 / 3)

    conv = None
    if causal_conv:
        conv = nn.Sequential(
            Rearrange("b n d -> b d n"),
            CausalConv1d(dim_inner, dim_inner, 3),
            Rearrange("b d n -> b n d"),
        )

    return Sequential(nn.Linear(dim, dim_inner * 2), GEGLU(), conv,
                      nn.Linear(dim_inner, dim))


class PerceiverResampler(nn.Module):

    def __init__(
        self,
        *,
        dim,
        depth=2,
        dim_context=None,
        num_latents=32,
        dim_head=64,
        heads=8,
        ff_mult=4,
        use_flash_attn=False,
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        self.proj_context = nn.Linear(
            dim_context, dim) if dim_context != dim else nn.Identity()

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(
                        dim=dim,
                        dim_head=dim_head,
                        heads=heads,
                        use_flash=use_flash_attn,
                        cross_attn_include_queries=True,
                    ),
                    FeedForward(dim=dim, mult=ff_mult),
                ]))

        self.norm = RMSNorm(dim)

    def forward(self, x, mask=None):
        batch = x.shape[0]

        x = self.proj_context(x)

        latents = repeat(self.latents, "n d -> b n d", b=batch)

        for attn, ff in self.layers:
            latents = attn(latents, x, mask=mask) + latents
            latents = ff(latents) + latents

        return self.norm(latents)


class GPT(nn.Module):

    def __init__(
        self,
        start_text_token=261,
        stop_text_token=0,
        layers=8,
        model_dim=512,
        heads=8,
        max_text_tokens=120,
        max_mel_tokens=250,
        max_prompt_tokens=70,
        max_conditioning_inputs=1,
        code_stride_len=1024,
        number_text_tokens=256,
        num_audio_tokens=8194,
        start_audio_token=8192,
        stop_audio_token=8193,
        train_solo_embeddings=False,
        checkpointing=False,
        average_conditioning_embeddings=False,
        label_smoothing=0.0,
        use_perceiver_resampler=False,
        perceiver_cond_length_compression=256,
    ):
        """
        Args:

        """
        super().__init__()

        self.label_smoothing = label_smoothing
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.num_audio_tokens = num_audio_tokens
        self.start_audio_token = start_audio_token
        self.stop_audio_token = stop_audio_token
        self.start_prompt_token = start_audio_token
        self.stop_prompt_token = stop_audio_token
        self.layers = layers
        self.heads = heads
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.max_gen_mel_tokens = max_mel_tokens - self.max_conditioning_inputs - 2
        self.max_mel_tokens = -1 if max_mel_tokens == -1 else max_mel_tokens + 2 + self.max_conditioning_inputs
        self.max_text_tokens = -1 if max_text_tokens == -1 else max_text_tokens + 2
        self.max_prompt_tokens = max_prompt_tokens
        self.code_stride_len = code_stride_len
        self.conditioning_encoder = ConditioningEncoder(80,
                                                        model_dim,
                                                        num_attn_heads=heads)
        self.conditioning_dropout = nn.Dropout1d(0.1)
        self.average_conditioning_embeddings = average_conditioning_embeddings
        self.use_perceiver_resampler = use_perceiver_resampler
        self.perceiver_cond_length_compression = perceiver_cond_length_compression

        self.text_embedding = nn.Embedding(self.number_text_tokens, model_dim)
        self.mel_embedding = nn.Embedding(self.num_audio_tokens, model_dim)

        (
            self.gpt,
            self.mel_pos_embedding,
            self.text_pos_embedding,
            self.mel_layer_pos_embedding,
            self.text_layer_pos_embedding,
        ) = build_hf_gpt_transformer(
            layers,
            model_dim,
            heads,
            self.max_mel_tokens,
            self.max_text_tokens,
            self.max_prompt_tokens,
            checkpointing,
        )
        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(
                torch.randn(1, 1, model_dim) * 0.02, requires_grad=True)
            self.text_solo_embedding = nn.Parameter(
                torch.randn(1, 1, model_dim) * 0.02, requires_grad=True)
        else:
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens)
        self.mel_head = nn.Linear(model_dim, self.num_audio_tokens)

        if self.use_perceiver_resampler:
            # XTTS v2
            self.conditioning_perceiver = PerceiverResampler(
                dim=model_dim,
                depth=2,
                dim_context=model_dim,
                num_latents=32,
                dim_head=64,
                heads=8,
                ff_mult=4,
                use_flash_attn=False,
            )
        else:
            # XTTS v1
            self.prompt_embedding = nn.Embedding(self.num_audio_tokens,
                                                 model_dim)
            self.prompt_pos_embedding = LearnedPositionEmbeddings(
                24 * 9, model_dim)

    def get_grad_norm_parameter_groups(self):
        return {
            "conditioning_encoder":
            list(self.conditioning_encoder.parameters()),
            "conditioning_perceiver":
            list(self.conditioning_perceiver.parameters())
            if self.use_perceiver_resampler else None,
            "gpt":
            list(self.gpt.parameters()),
            "heads":
            list(self.text_head.parameters()) +
            list(self.mel_head.parameters()),
        }

    def init_gpt_for_inference(self, kv_cache=True, use_deepspeed=False):
        seq_length = self.max_prompt_tokens + self.max_mel_tokens + self.max_text_tokens + 1
        gpt_config = GPT2Config(
            vocab_size=self.max_mel_tokens,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            gradient_checkpointing=False,
            use_cache=True,
        )
        self.gpt_inference = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        self.gpt.wte = self.mel_embedding

        if use_deepspeed:
            import deepspeed

            self.ds_engine = deepspeed.init_inference(
                model=self.gpt_inference.half(),  # Transformers models
                mp_size=1,  # Number of GPU
                dtype=torch.float32,  # desired data type of output
                replace_method=
                "auto",  # Lets DS autmatically identify the layer to replace
                replace_with_kernel_inject=
                True,  # replace the model with the kernel injector
            )
            self.gpt_inference = self.ds_engine.module.eval()

    def set_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, code_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with stop_audio_token in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        # Set padding areas within MEL (currently it is coded with the MEL code for <zero>).
        for b in range(len(code_lengths)):
            actual_end = code_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_audio_token
        return mel_input_tokens

    def get_logits(
        self,
        first_inputs,
        first_head,
        second_inputs=None,
        second_head=None,
        prompt=None,
        get_attns=False,
        return_latent=False,
        attn_mask_cond=None,
        attn_mask_text=None,
        attn_mask_mel=None,
    ):
        if prompt is not None:
            offset = prompt.shape[1]
            if second_inputs is not None:
                emb = torch.cat([prompt, first_inputs, second_inputs], dim=1)
            else:
                emb = torch.cat([prompt, first_inputs], dim=1)

        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        attn_mask = None
        if attn_mask_text is not None:
            attn_mask = torch.cat([attn_mask_text, attn_mask_mel], dim=1)
            if prompt is not None:
                attn_mask_cond = torch.ones(prompt.shape[0],
                                            offset,
                                            dtype=torch.bool,
                                            device=emb.device)
                attn_mask = torch.cat([attn_mask_cond, attn_mask], dim=1)

        gpt_out = self.gpt(
            inputs_embeds=emb,
            return_dict=True,
            output_attentions=get_attns,
            attention_mask=attn_mask,
        )

        if get_attns:
            return gpt_out.attentions

        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)

        if return_latent:
            return enc[:, :first_inputs.
                       shape[1]], enc[:, -second_inputs.shape[1]:]

        first_logits = enc[:, :first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        if second_inputs is not None:
            second_logits = enc[:, -second_inputs.shape[1]:]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        else:
            return first_logits

    def get_conditioning(self, speech_conditioning_input):
        speech_conditioning_input = (speech_conditioning_input.unsqueeze(1)
                                     if len(speech_conditioning_input.shape)
                                     == 3 else speech_conditioning_input)
        conds = []
        for j in range(speech_conditioning_input.shape[1]):
            conds.append(
                self.conditioning_encoder(speech_conditioning_input[:, j]))
        conds = torch.stack(conds, dim=1)
        conds = conds.mean(dim=1)
        return conds

    def get_prompts(self, prompt_codes):
        """
        Create a prompt from the mel codes. This is used to condition the model on the mel codes.
        Pad the prompt with start and stop mel tokens.
        """
        prompt = prompt_codes
        if self.training:
            lengths = []
            # Compute the real prompt length based on the first encounter with the token 83 used for padding
            for i in range(prompt_codes.shape[0]):
                length = 0
                for j in range(prompt_codes.shape[1]):
                    if prompt_codes[i, j] == 83:
                        break
                    else:
                        length += 1
                lengths.append(length)

            # prompt_len = random.randint(1, 9)  # in secs
            prompt_len = 3
            prompt_len = prompt_len * 24  # in frames
            if prompt_codes.shape[-1] >= prompt_len:
                for i in range(prompt_codes.shape[0]):
                    if lengths[i] < prompt_len:
                        start = 0
                    else:
                        start = random.randint(0, lengths[i] - prompt_len)
                prompt = prompt_codes[:, start:start + prompt_len]

        # add start and stop tokens
        prompt = F.pad(prompt, (1, 0), value=self.start_prompt_token)
        prompt = F.pad(prompt, (0, 1), value=self.stop_prompt_token)
        return prompt

    def get_style_emb(self, cond_input, return_latent=False):
        """
        cond_input: (b, 80, s) or (b, 1, 80, s)
        conds: (b, 1024, s)
        """
        conds = None
        if not return_latent:
            if cond_input.ndim == 4:
                cond_input = cond_input.squeeze(1)
            conds = self.conditioning_encoder(cond_input)  # (b, d, s)
            if self.use_perceiver_resampler:
                conds = self.conditioning_perceiver(conds.permute(
                    0, 2, 1)).transpose(1, 2)  # (b, d, 32)
        else:
            # already computed
            conds = cond_input.unsqueeze(1)
        return conds

    def forward(
        self,
        text_inputs,
        text_lengths,
        audio_codes,
        wav_lengths,
        cond_mels=None,
        cond_idxs=None,
        cond_lens=None,
        cond_latents=None,
        return_attentions=False,
        return_latent=False,
    ):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        cond_mels: MEL float tensor, (b, 1, 80,s)
        cond_idxs: cond start and end indexs, (b, 2)

        If return_attentions is specified, only logits are returned.
        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        """
        # ❗ FIXIT
        if self.max_conditioning_inputs == 0:
            assert cond_mels is None, " ❗ cond_mels is not None, but max_conditioning_inputs == 0"

        max_text_len = text_lengths.max()
        code_lengths = torch.ceil(
            wav_lengths / self.code_stride_len).long() + 3

        if cond_lens is not None:
            if self.use_perceiver_resampler:
                cond_lens = cond_lens // self.perceiver_cond_length_compression
            else:
                cond_lens = cond_lens // self.code_stride_len

        if cond_idxs is not None:
            # recompute cond idxs for mel lengths
            for idx in range(cond_idxs.size(0)):
                if self.use_perceiver_resampler:
                    cond_idxs[idx] = cond_idxs[
                        idx] // self.perceiver_cond_length_compression
                else:
                    cond_idxs[idx] = cond_idxs[idx] // self.code_stride_len

        # ensure that the cond_mel does not have padding
        # if cond_lens is not None and cond_idxs is None:
        #     min_cond_len = torch.min(cond_lens)
        #     cond_mels = cond_mels[:, :, :, :min_cond_len]

        # If len(codes) + 3 is larger than maxiumum allowed length, we truncate the codes.
        max_mel_len = code_lengths.max()

        if max_mel_len > audio_codes.shape[-1]:
            audio_codes = F.pad(audio_codes,
                                (0, max_mel_len - audio_codes.shape[-1]))

        # 💖 Lovely assertions
        assert (
            max_mel_len <= audio_codes.shape[-1]
        ), f" ❗ max_mel_len ({max_mel_len}) > audio_codes.shape[-1] ({audio_codes.shape[-1]})"
        assert (
            max_text_len <= text_inputs.shape[-1]
        ), f" ❗ max_text_len ({max_text_len}) > text_inputs.shape[-1] ({text_inputs.shape[-1]})"

        # Append stop token to text inputs
        text_inputs = F.pad(text_inputs[:, :max_text_len], (0, 1),
                            value=self.stop_text_token)

        # Append silence token to mel codes
        audio_codes = F.pad(audio_codes[:, :max_mel_len], (0, 1),
                            value=self.stop_audio_token)

        # Pad mel codes with stop_audio_token
        audio_codes = self.set_mel_padding(
            audio_codes, code_lengths - 3
        )  # -3 to get the real code lengths without consider start and stop tokens that was not added yet

        # Build input and target tensors
        # Prepend start token to inputs and append stop token to targets
        text_inputs, text_targets = self.set_inputs_and_targets(
            text_inputs, self.start_text_token, self.stop_text_token)
        audio_codes, mel_targets = self.set_inputs_and_targets(
            audio_codes, self.start_audio_token, self.stop_audio_token)

        # Set attn_mask
        attn_mask_cond = None
        attn_mask_text = None
        attn_mask_mel = None
        if not return_latent:
            attn_mask_cond = torch.ones(
                cond_mels.shape[0],
                cond_mels.shape[-1],
                dtype=torch.bool,
                device=text_inputs.device,
            )
            attn_mask_text = torch.ones(
                text_inputs.shape[0],
                text_inputs.shape[1],
                dtype=torch.bool,
                device=text_inputs.device,
            )
            attn_mask_mel = torch.ones(
                audio_codes.shape[0],
                audio_codes.shape[1],
                dtype=torch.bool,
                device=audio_codes.device,
            )

            if cond_idxs is not None:
                # use masking approach
                for idx, r in enumerate(cond_idxs):
                    l = r[1] - r[0]
                    attn_mask_cond[idx, l:] = 0.0
            elif cond_lens is not None:
                for idx, l in enumerate(cond_lens):
                    attn_mask_cond[idx, l:] = 0.0

            for idx, l in enumerate(text_lengths):
                attn_mask_text[idx, l + 1:] = 0.0

            for idx, l in enumerate(code_lengths):
                attn_mask_mel[idx, l + 1:] = 0.0

        # Compute text embeddings + positional embeddings
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(
            text_inputs)

        # Compute mel embeddings + positional embeddings
        mel_emb = self.mel_embedding(audio_codes) + self.mel_pos_embedding(
            audio_codes)

        # Compute speech conditioning input
        if cond_latents is None:
            cond_latents = self.get_style_emb(cond_mels).transpose(1, 2)

        # Get logits
        sub = -5  # don't ask me why 😄
        if self.training:
            sub = -1

        text_logits, mel_logits = self.get_logits(
            text_emb,
            self.text_head,
            mel_emb,
            self.mel_head,
            prompt=cond_latents,
            get_attns=return_attentions,
            return_latent=return_latent,
            attn_mask_cond=attn_mask_cond,
            attn_mask_text=attn_mask_text,
            attn_mask_mel=attn_mask_mel,
        )
        if return_latent:
            return mel_logits[:, :sub]  # sub to prevent bla.

        if return_attentions:
            return mel_logits

        # Set paddings to -1 to ignore them in loss
        for idx, l in enumerate(text_lengths):
            text_targets[idx, l + 1:] = -1

        for idx, l in enumerate(code_lengths):
            mel_targets[idx, l + 1:] = -1

        # check if stoptoken is in every row of mel_targets
        assert (mel_targets == self.stop_audio_token).sum(
        ) >= mel_targets.shape[
            0], f" ❗ mel_targets does not contain stop token ({self.stop_audio_token}) in every row."

        # ignore the loss for the segment used for conditioning
        # coin flip for the segment to be ignored
        if cond_idxs is not None:
            cond_start = cond_idxs[idx, 0]
            cond_end = cond_idxs[idx, 1]
            mel_targets[idx, cond_start:cond_end] = -1

        # Compute losses
        loss_text = F.cross_entropy(text_logits,
                                    text_targets.long(),
                                    ignore_index=-1,
                                    label_smoothing=self.label_smoothing)
        loss_mel = F.cross_entropy(mel_logits,
                                   mel_targets.long(),
                                   ignore_index=-1,
                                   label_smoothing=self.label_smoothing)
        return loss_text.mean(), loss_mel.mean(), mel_logits

    def inference(self, cond_latents, text_inputs, **hf_generate_kwargs):
        self.compute_embeddings(cond_latents, text_inputs)
        return self.generate(cond_latents, text_inputs, **hf_generate_kwargs)

    def compute_embeddings(
        self,
        cond_latents,
        text_inputs,
    ):
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs = F.pad(text_inputs, (1, 0), value=self.start_text_token)
        emb = self.text_embedding(text_inputs) + self.text_pos_embedding(
            text_inputs)
        emb = torch.cat([cond_latents, emb], dim=1)
        self.gpt_inference.store_prefix_emb(emb)
        gpt_inputs = torch.full(
            (
                emb.shape[0],
                emb.shape[1] + 1,  # +1 for the start_audio_token
            ),
            fill_value=1,
            dtype=torch.long,
            device=text_inputs.device,
        )
        gpt_inputs[:, -1] = self.start_audio_token
        return gpt_inputs

    def generate(
        self,
        cond_latents,
        text_inputs,
        **hf_generate_kwargs,
    ):
        gpt_inputs = self.compute_embeddings(cond_latents, text_inputs)
        gen = self.gpt_inference.generate(
            gpt_inputs,
            bos_token_id=self.start_audio_token,
            pad_token_id=self.stop_audio_token,
            eos_token_id=self.stop_audio_token,
            max_length=self.max_gen_mel_tokens + gpt_inputs.shape[-1],
            **hf_generate_kwargs,
        )
        if "return_dict_in_generate" in hf_generate_kwargs:
            return gen.sequences[:, gpt_inputs.shape[1]:], gen
        return gen[:, gpt_inputs.shape[1]:]

    def get_generator(self, fake_inputs, **hf_generate_kwargs):
        return self.gpt_inference.generate_stream(
            fake_inputs,
            bos_token_id=self.start_audio_token,
            pad_token_id=self.stop_audio_token,
            eos_token_id=self.stop_audio_token,
            max_length=self.max_gen_mel_tokens + fake_inputs.shape[-1],
            do_stream=True,
            **hf_generate_kwargs,
        )


_symbols_multilingual = {
    "en": [(re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
           for x in [
               ("&", " and "),
               ("@", " at "),
               ("%", " percent "),
               ("#", " hash "),
               ("$", " dollar "),
               ("£", " pound "),
               ("°", " degree "),
           ]],
}

_abbreviations = {
    "en": [(re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1]) for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]],
}

_whitespace_re = re.compile(r"\s+")

_currency_re = {
    "USD": re.compile(r"((\$[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+\$))"),
    "GBP": re.compile(r"((£[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+£))"),
    "EUR": re.compile(r"(([0-9\.\,]*[0-9]+€)|((€[0-9\.\,]*[0-9]+)))"),
}

_comma_number_re = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")

_ordinal_re = {
    "en": re.compile(r"([0-9]+)(st|nd|rd|th)"),
    "es": re.compile(r"([0-9]+)(º|ª|er|o|a|os|as)"),
    "fr": re.compile(r"([0-9]+)(º|ª|er|re|e|ème)"),
    "de": re.compile(r"([0-9]+)(st|nd|rd|th|º|ª|\.(?=\s|$))"),
    "pt": re.compile(r"([0-9]+)(º|ª|o|a|os|as)"),
    "it": re.compile(r"([0-9]+)(º|°|ª|o|a|i|e)"),
    "pl": re.compile(r"([0-9]+)(º|ª|st|nd|rd|th)"),
    "ar": re.compile(r"([0-9]+)(ون|ين|ث|ر|ى)"),
    "cs": re.compile(
        r"([0-9]+)\.(?=\s|$)"
    ),  # In Czech, a dot is often used after the number to indicate ordinals.
    "ru": re.compile(r"([0-9]+)(-й|-я|-е|-ое|-ье|-го)"),
    "nl": re.compile(r"([0-9]+)(de|ste|e)"),
    "tr": re.compile(r"([0-9]+)(\.|inci|nci|uncu|üncü|\.)"),
    "hu": re.compile(r"([0-9]+)(\.|adik|edik|odik|edik|ödik|ödike|ik)"),
    "ko": re.compile(r"([0-9]+)(번째|번|차|째)"),
}
_number_re = re.compile(r"[0-9]+")


class BaseTTS(nn.Module):
    """Base `tts` class. Every new `tts` model must inherit this.

    It defines common `tts` specific functions on top of `Model` implementation.
    """

    MODEL_TYPE = "tts"

    def __init__(
        self,
        config,
        ap,
        tokenizer,
        speaker_manager=None,
        language_manager=None,
    ):
        super().__init__()
        self.config = config
        self.ap = ap
        self.tokenizer = tokenizer
        self.speaker_manager = speaker_manager
        self.language_manager = language_manager
        self._set_model_args(config)

    def _set_model_args(self, config):
        """Setup model args based on the config type (`ModelConfig` or `ModelArgs`).

        `ModelArgs` has all the fields reuqired to initialize the model architecture.

        `ModelConfig` has all the fields required for training, inference and containes `ModelArgs`.

        If the config is for training with a name like "*Config", then the model args are embeded in the
        config.model_args

        If the config is for the model with a name like "*Args", then we assign the directly.
        """
        # don't use isintance not to import recursively
        if "Config" in config.__class__.__name__:
            config_num_chars = (self.config.model_args.num_chars if hasattr(
                self.config, "model_args") else self.config.num_chars)
            num_chars = config_num_chars if self.tokenizer is None else self.tokenizer.characters.num_chars
            if "characters" in config:
                self.config.num_chars = num_chars
                if hasattr(self.config, "model_args"):
                    config.model_args.num_chars = num_chars
                    self.args = self.config.model_args
            else:
                self.config = config
                self.args = config.model_args
        elif "Args" in config.__class__.__name__:
            self.args = config
        else:
            raise ValueError("config must be either a *Config or *Args")

    def get_aux_input(self, **kwargs):
        """Prepare and return `aux_input` used by `forward()`"""
        return {
            "speaker_id": None,
            "style_wav": None,
            "d_vector": None,
            "language_id": None
        }

    def get_aux_input_from_test_sentences(self, sentence_info):
        if hasattr(self.config, "model_args"):
            config = self.config.model_args
        else:
            config = self.config

        # extract speaker and language info
        text, speaker_name, style_wav, language_name = None, None, None, None

        if isinstance(sentence_info, list):
            if len(sentence_info) == 1:
                text = sentence_info[0]
            elif len(sentence_info) == 2:
                text, speaker_name = sentence_info
            elif len(sentence_info) == 3:
                text, speaker_name, style_wav = sentence_info
            elif len(sentence_info) == 4:
                text, speaker_name, style_wav, language_name = sentence_info
        else:
            text = sentence_info

        # get speaker  id/d_vector
        speaker_id, d_vector, language_id = None, None, None
        if self.speaker_manager is not None:
            if config.use_d_vector_file:
                if speaker_name is None:
                    d_vector = self.speaker_manager.get_random_embedding()
                else:
                    d_vector = self.speaker_manager.get_d_vector_by_name(
                        speaker_name)
            elif config.use_speaker_embedding:
                if speaker_name is None:
                    speaker_id = self.speaker_manager.get_random_id()
                else:
                    speaker_id = self.speaker_manager.name_to_id[speaker_name]

        # get language id
        if self.language_manager is not None and config.use_language_embedding and language_name is not None:
            language_id = self.language_manager.name_to_id[language_name]

        return {
            "text": text,
            "speaker_id": speaker_id,
            "style_wav": style_wav,
            "d_vector": d_vector,
            "language_id": language_id,
        }

    def format_batch(self, batch):
        """Generic batch formatting for `TTSDataset`.

        You must override this if you use a custom dataset.

        Args:
            batch (Dict): [description]

        Returns:
            Dict: [description]
        """
        # setup input batch
        text_input = batch["token_id"]
        text_lengths = batch["token_id_lengths"]
        speaker_names = batch["speaker_names"]
        linear_input = batch["linear"]
        mel_input = batch["mel"]
        mel_lengths = batch["mel_lengths"]
        stop_targets = batch["stop_targets"]
        item_idx = batch["item_idxs"]
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]
        attn_mask = batch["attns"]
        waveform = batch["waveform"]
        pitch = batch["pitch"]
        energy = batch["energy"]
        language_ids = batch["language_ids"]
        max_text_length = torch.max(text_lengths.float())
        max_spec_length = torch.max(mel_lengths.float())

        # compute durations from attention masks
        durations = None
        if attn_mask is not None:
            durations = torch.zeros(attn_mask.shape[0], attn_mask.shape[2])
            for idx, am in enumerate(attn_mask):
                # compute raw durations
                c_idxs = am[:, :text_lengths[idx], :mel_lengths[idx]].max(1)[1]
                # c_idxs, counts = torch.unique_consecutive(c_idxs, return_counts=True)
                c_idxs, counts = torch.unique(c_idxs, return_counts=True)
                dur = torch.ones([text_lengths[idx]]).to(counts.dtype)
                dur[c_idxs] = counts
                # smooth the durations and set any 0 duration to 1
                # by cutting off from the largest duration indeces.
                extra_frames = dur.sum() - mel_lengths[idx]
                largest_idxs = torch.argsort(-dur)[:extra_frames]
                dur[largest_idxs] -= 1
                assert (
                    dur.sum() == mel_lengths[idx]
                ), f" [!] total duration {dur.sum()} vs spectrogram length {mel_lengths[idx]}"
                durations[idx, :text_lengths[idx]] = dur

        # set stop targets wrt reduction factor
        stop_targets = stop_targets.view(text_input.shape[0],
                                         stop_targets.size(1) // self.config.r,
                                         -1)
        stop_targets = (stop_targets.sum(2)
                        > 0.0).unsqueeze(2).float().squeeze(2)
        stop_target_lengths = torch.divide(mel_lengths, self.config.r).ceil_()

        return {
            "text_input": text_input,
            "text_lengths": text_lengths,
            "speaker_names": speaker_names,
            "mel_input": mel_input,
            "mel_lengths": mel_lengths,
            "linear_input": linear_input,
            "stop_targets": stop_targets,
            "stop_target_lengths": stop_target_lengths,
            "attn_mask": attn_mask,
            "durations": durations,
            "speaker_ids": speaker_ids,
            "d_vectors": d_vectors,
            "max_text_length": float(max_text_length),
            "max_spec_length": float(max_spec_length),
            "item_idx": item_idx,
            "waveform": waveform,
            "pitch": pitch,
            "energy": energy,
            "language_ids": language_ids,
            "audio_unique_names": batch["audio_unique_names"],
        }

    def get_sampler(self, config, dataset, num_gpus=1):
        weights = None
        data_items = dataset.samples

        if getattr(config, "use_language_weighted_sampler", False):
            alpha = getattr(config, "language_weighted_sampler_alpha", 1.0)
            print(" > Using Language weighted sampler with alpha:", alpha)
            weights = get_language_balancer_weights(data_items) * alpha

        if getattr(config, "use_speaker_weighted_sampler", False):
            alpha = getattr(config, "speaker_weighted_sampler_alpha", 1.0)
            print(" > Using Speaker weighted sampler with alpha:", alpha)
            if weights is not None:
                weights += get_speaker_balancer_weights(data_items) * alpha
            else:
                weights = get_speaker_balancer_weights(data_items) * alpha

        if getattr(config, "use_length_weighted_sampler", False):
            alpha = getattr(config, "length_weighted_sampler_alpha", 1.0)
            print(" > Using Length weighted sampler with alpha:", alpha)
            if weights is not None:
                weights += get_length_balancer_weights(data_items) * alpha
            else:
                weights = get_length_balancer_weights(data_items) * alpha

        if weights is not None:
            sampler = WeightedRandomSampler(weights, len(weights))
        else:
            sampler = None

        # sampler for DDP
        if sampler is None:
            sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        else:  # If a sampler is already defined use this sampler and DDP sampler together
            sampler = DistributedSamplerWrapper(
                sampler) if num_gpus > 1 else sampler

        return sampler

    def get_data_loader(
        self,
        config,
        assets,
        is_eval,
        samples,
        verbose,
        num_gpus,
        rank=None,
    ):
        if is_eval and not config.run_eval:
            loader = None
        else:
            # setup multi-speaker attributes
            if self.speaker_manager is not None:
                if hasattr(config, "model_args"):
                    speaker_id_mapping = (
                        self.speaker_manager.name_to_id
                        if config.model_args.use_speaker_embedding else None)
                    d_vector_mapping = self.speaker_manager.embeddings if config.model_args.use_d_vector_file else None
                    config.use_d_vector_file = config.model_args.use_d_vector_file
                else:
                    speaker_id_mapping = self.speaker_manager.name_to_id if config.use_speaker_embedding else None
                    d_vector_mapping = self.speaker_manager.embeddings if config.use_d_vector_file else None
            else:
                speaker_id_mapping = None
                d_vector_mapping = None

            # setup multi-lingual attributes
            if self.language_manager is not None:
                language_id_mapping = self.language_manager.name_to_id if self.args.use_language_embedding else None
            else:
                language_id_mapping = None

            # init dataloader
            dataset = TTSDataset(
                outputs_per_step=config.r if "r" in config else 1,
                compute_linear_spec=config.model.lower() == "tacotron"
                or config.compute_linear_spec,
                compute_f0=config.get("compute_f0", False),
                f0_cache_path=config.get("f0_cache_path", None),
                compute_energy=config.get("compute_energy", False),
                energy_cache_path=config.get("energy_cache_path", None),
                samples=samples,
                ap=self.ap,
                return_wav=config.return_wav
                if "return_wav" in config else False,
                batch_group_size=0 if is_eval else config.batch_group_size *
                config.batch_size,
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                use_noise_augment=False
                if is_eval else config.use_noise_augment,
                verbose=verbose,
                speaker_id_mapping=speaker_id_mapping,
                d_vector_mapping=d_vector_mapping
                if config.use_d_vector_file else None,
                tokenizer=self.tokenizer,
                start_by_longest=config.start_by_longest,
                language_id_mapping=language_id_mapping,
            )

            # wait all the DDP process to be ready
            if num_gpus > 1:
                dist.barrier()

            # sort input sequences from short to long
            dataset.preprocess_samples()

            # get samplers
            sampler = self.get_sampler(config, dataset, num_gpus)

            loader = DataLoader(
                dataset,
                batch_size=config.eval_batch_size
                if is_eval else config.batch_size,
                shuffle=config.shuffle
                if sampler is None else False,  # if there is no other sampler
                collate_fn=dataset.collate_fn,
                drop_last=config.
                drop_last,  # setting this False might cause issues in AMP training.
                sampler=sampler,
                num_workers=config.num_eval_loader_workers
                if is_eval else config.num_loader_workers,
                pin_memory=False,
            )
        return loader

    def _get_test_aux_input(self):
        d_vector = None
        if self.config.use_d_vector_file:
            d_vector = [
                self.speaker_manager.embeddings[name]["embedding"]
                for name in self.speaker_manager.embeddings
            ]
            d_vector = (random.sample(sorted(d_vector), 1), )

        aux_inputs = {
            "speaker_id":
            None if not self.config.use_speaker_embedding else random.sample(
                sorted(self.speaker_manager.name_to_id.values()), 1),
            "d_vector":
            d_vector,
            "style_wav":
            None,  # TODO: handle GST style input
        }
        return aux_inputs

    def test_run(self, assets):
        """Generic test run for `tts` models used by `Trainer`.

        You can override this for a different behaviour.

        Args:
            assets (dict): A dict of training assets. For `tts` models, it must include `{'audio_processor': ap}`.

        Returns:
            Tuple[Dict, Dict]: Test figures and audios to be projected to Tensorboard.
        """
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        aux_inputs = self._get_test_aux_input()
        for idx, sen in enumerate(test_sentences):
            if isinstance(sen, list):
                aux_inputs = self.get_aux_input_from_test_sentences(sen)
                sen = aux_inputs["text"]
            outputs_dict = synthesis(
                self,
                sen,
                self.config,
                "cuda" in str(next(self.parameters()).device),
                speaker_id=aux_inputs["speaker_id"],
                d_vector=aux_inputs["d_vector"],
                style_wav=aux_inputs["style_wav"],
                use_griffin_lim=True,
                do_trim_silence=False,
            )
            test_audios["{}-audio".format(idx)] = outputs_dict["wav"]
            test_figures["{}-prediction".format(idx)] = plot_spectrogram(
                outputs_dict["outputs"]["model_outputs"],
                self.ap,
                output_fig=False)
            test_figures["{}-alignment".format(idx)] = plot_alignment(
                outputs_dict["outputs"]["alignments"], output_fig=False)
        return test_figures, test_audios

    def on_init_start(self, trainer):
        """Save the speaker.pth and language_ids.json at the beginning of the training. Also update both paths."""
        if self.speaker_manager is not None:
            output_path = os.path.join(trainer.output_path, "speakers.pth")
            self.speaker_manager.save_ids_to_file(output_path)
            trainer.config.speakers_file = output_path
            # some models don't have `model_args` set
            if hasattr(trainer.config, "model_args"):
                trainer.config.model_args.speakers_file = output_path
            trainer.config.save_json(
                os.path.join(trainer.output_path, "config.json"))
            print(f" > `speakers.pth` is saved to {output_path}.")
            print(" > `speakers_file` is updated in the config.json.")

        if self.language_manager is not None:
            output_path = os.path.join(trainer.output_path,
                                       "language_ids.json")
            self.language_manager.save_ids_to_file(output_path)
            trainer.config.language_ids_file = output_path
            if hasattr(trainer.config, "model_args"):
                trainer.config.model_args.language_ids_file = output_path
            trainer.config.save_json(
                os.path.join(trainer.output_path, "config.json"))
            print(f" > `language_ids.json` is saved to {output_path}.")
            print(" > `language_ids_file` is updated in the config.json.")


def _expand_currency(m, lang="en", currency="USD"):
    amount = float((re.sub(r"[^\d.]", "", m.group(0).replace(",", "."))))
    full_amount = num2words(amount,
                            to="currency",
                            currency=currency,
                            lang=lang if lang != "cs" else "cz")

    and_equivalents = {
        "en": ", ",
        "es": " con ",
        "fr": " et ",
        "de": " und ",
        "pt": " e ",
        "it": " e ",
        "pl": ", ",
        "cs": ", ",
        "ru": ", ",
        "nl": ", ",
        "ar": ", ",
        "tr": ", ",
        "hu": ", ",
        "ko": ", ",
    }

    if amount.is_integer():
        last_and = full_amount.rfind(and_equivalents[lang])
        if last_and != -1:
            full_amount = full_amount[:last_and]

    return full_amount


def _expand_ordinal(m, lang="en"):
    return num2words(int(m.group(1)),
                     ordinal=True,
                     lang=lang if lang != "cs" else "cz")


def _expand_number(m, lang="en"):
    return num2words(int(m.group(0)), lang=lang if lang != "cs" else "cz")


def _remove_commas(m):
    text = m.group(0)
    if "," in text:
        text = text.replace(",", "")
    return text


def expand_numbers_multilingual(text, lang="en"):
    if lang in ["en", "ru"]:
        text = re.sub(_comma_number_re, _remove_commas, text)
    try:
        text = re.sub(_currency_re["GBP"],
                      lambda m: _expand_currency(m, lang, "GBP"), text)
        text = re.sub(_currency_re["USD"],
                      lambda m: _expand_currency(m, lang, "USD"), text)
        text = re.sub(_currency_re["EUR"],
                      lambda m: _expand_currency(m, lang, "EUR"), text)
    except:
        pass
    text = re.sub(_ordinal_re[lang], lambda m: _expand_ordinal(m, lang), text)
    text = re.sub(_number_re, lambda m: _expand_number(m, lang), text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def expand_abbreviations_multilingual(text, lang="en"):
    for regex, replacement in _abbreviations[lang]:
        text = re.sub(regex, replacement, text)
    return text


def expand_symbols_multilingual(text, lang="en"):
    for regex, replacement in _symbols_multilingual[lang]:
        text = re.sub(regex, replacement, text)
        text = text.replace("  ", " ")  # Ensure there are no double spaces
    return text.strip()


def multilingual_cleaners(text, lang):
    text = text.replace('"', "")
    if lang == "tr":
        text = text.replace("İ", "i")
        text = text.replace("Ö", "ö")
        text = text.replace("Ü", "ü")
    text = lowercase(text)
    text = expand_numbers_multilingual(text, lang)
    text = expand_abbreviations_multilingual(text, lang)
    text = expand_symbols_multilingual(text, lang=lang)
    text = collapse_whitespace(text)
    return text


class VoiceBpeTokenizer:

    def __init__(self, vocab_file=None):
        self.tokenizer = None
        if vocab_file is not None:
            self.tokenizer = Tokenizer.from_file(vocab_file)
        self.char_limits = {
            "en": 250,
            "de": 253,
            "fr": 273,
            "es": 239,
            "it": 213,
            "pt": 203,
            "pl": 224,
            "zh": 82,
            "ar": 166,
            "cs": 186,
            "ru": 182,
            "nl": 251,
            "tr": 226,
            "ja": 71,
            "hu": 224,
            "ko": 95,
        }

    @cached_property
    def katsu(self):
        import cutlet

        return cutlet.Cutlet()

    def check_input_length(self, txt, lang):
        lang = lang.split("-")[0]  # remove the region
        limit = self.char_limits.get(lang, 250)
        if len(txt) > limit:
            print(
                f"[!] Warning: The text length exceeds the character limit of {limit} for language '{lang}', this might cause truncated audio."
            )

    def preprocess_text(self, txt, lang):
        if lang in {
                "ar", "cs", "de", "en", "es", "fr", "hu", "it", "nl", "pl",
                "pt", "ru", "tr", "zh", "ko"
        }:
            txt = multilingual_cleaners(txt, lang)
            if lang == "zh":
                txt = chinese_transliterate(txt)
            if lang == "ko":
                txt = korean_transliterate(txt)
        elif lang == "ja":
            txt = japanese_cleaners(txt, self.katsu)
        elif lang == "hi":
            # @manmay will implement this
            txt = basic_cleaners(txt)
        else:
            raise NotImplementedError(f"Language '{lang}' is not supported.")
        return txt

    def encode(self, txt, lang):
        lang = lang.split("-")[0]  # remove the region
        self.check_input_length(txt, lang)
        txt = self.preprocess_text(txt, lang)
        lang = "zh-cn" if lang == "zh" else lang
        txt = f"[{lang}]{txt}"
        txt = txt.replace(" ", "[SPACE]")
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(
            " ", "")
        txt = txt.replace("[SPACE]", " ")
        txt = txt.replace("[STOP]", "")
        txt = txt.replace("[UNK]", "")
        return txt

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def get_number_tokens(self):
        return max(self.tokenizer.get_vocab().values()) + 1


class Xtts(BaseTTS):
    """ⓍTTS model implementation.

    ❗ Currently it only supports inference.

    Examples:
        >>> from TTS.tts.configs.xtts_config import XttsConfig
        >>> from TTS.tts.models.xtts import Xtts
        >>> config = XttsConfig()
        >>> model = Xtts.inif_from_config(config)
        >>> model.load_checkpoint(config, checkpoint_dir="paths/to/models_dir/", eval=True)
    """

    def __init__(self, config):
        super().__init__(config, ap=None, tokenizer=None)
        self.mel_stats_path = None
        self.config = config
        self.gpt_batch_size = self.args.gpt_batch_size  # 1

        # self.tokenizer = VoiceBpeTokenizer()
        # self.gpt = None
        # self.init_models()
        self.register_buffer("mel_stats", torch.ones(80))

    def init_models(self):
        """Initialize the models. We do it here since we need to load the tokenizer first."""
        if self.tokenizer.tokenizer is not None:
            self.args.gpt_number_text_tokens = self.tokenizer.get_number_tokens(
            )
            self.args.gpt_start_text_token = self.tokenizer.tokenizer.token_to_id(
                "[START]")
            self.args.gpt_stop_text_token = self.tokenizer.tokenizer.token_to_id(
                "[STOP]")

        if self.args.gpt_number_text_tokens:
            self.gpt = GPT(
                layers=self.args.gpt_layers,
                model_dim=self.args.gpt_n_model_channels,
                start_text_token=self.args.gpt_start_text_token,
                stop_text_token=self.args.gpt_stop_text_token,
                heads=self.args.gpt_n_heads,
                max_text_tokens=self.args.gpt_max_text_tokens,
                max_mel_tokens=self.args.gpt_max_audio_tokens,
                max_prompt_tokens=self.args.gpt_max_prompt_tokens,
                number_text_tokens=self.args.gpt_number_text_tokens,
                num_audio_tokens=self.args.gpt_num_audio_tokens,
                start_audio_token=self.args.gpt_start_audio_token,
                stop_audio_token=self.args.gpt_stop_audio_token,
                use_perceiver_resampler=self.args.gpt_use_perceiver_resampler,
                code_stride_len=self.args.gpt_code_stride_len,
            )

        self.hifigan_decoder = HifiDecoder(
            input_sample_rate=self.args.input_sample_rate,
            output_sample_rate=self.args.output_sample_rate,
            output_hop_length=self.args.output_hop_length,
            ar_mel_length_compression=self.args.gpt_code_stride_len,
            decoder_input_dim=self.args.decoder_input_dim,
            d_vector_dim=self.args.d_vector_dim,
            cond_d_vector_in_each_upsampling_layer=self.args.
            cond_d_vector_in_each_upsampling_layer,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def get_gpt_cond_latents(self,
                             audio,
                             sr,
                             length: int = 30,
                             chunk_length: int = 6):
        """Compute the conditioning latents for the GPT model from the given audio.

        Args:
            audio (tensor): audio tensor.
            sr (int): Sample rate of the audio.
            length (int): Length of the audio in seconds. If < 0, use the whole audio. Defaults to 30.
            chunk_length (int): Length of the audio chunks in seconds. When `length == chunk_length`, the whole audio
                is being used without chunking. It must be < `length`. Defaults to 6.
        """
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        if length > 0:
            audio = audio[:, :22050 * length]
        if self.args.gpt_use_perceiver_resampler:
            style_embs = []
            for i in range(0, audio.shape[1], 22050 * chunk_length):
                audio_chunk = audio[:, i:i + 22050 * chunk_length]

                # if the chunk is too short ignore it
                if audio_chunk.size(-1) < 22050 * 0.33:
                    continue

                mel_chunk = wav_to_mel_cloning(
                    audio_chunk,
                    mel_norms=self.mel_stats.cpu(),
                    n_fft=2048,
                    hop_length=256,
                    win_length=1024,
                    power=2,
                    normalized=False,
                    sample_rate=22050,
                    f_min=0,
                    f_max=8000,
                    n_mels=80,
                )
                style_emb = self.gpt.get_style_emb(mel_chunk.to(self.device),
                                                   None)
                style_embs.append(style_emb)

            # mean style embedding
            cond_latent = torch.stack(style_embs).mean(dim=0)
        else:
            mel = wav_to_mel_cloning(
                audio,
                mel_norms=self.mel_stats.cpu(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                power=2,
                normalized=False,
                sample_rate=22050,
                f_min=0,
                f_max=8000,
                n_mels=80,
            )
            cond_latent = self.gpt.get_style_emb(mel.to(self.device))
        return cond_latent.transpose(1, 2)

    @torch.inference_mode()
    def get_speaker_embedding(self, audio, sr):
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        return (self.hifigan_decoder.speaker_encoder.forward(
            audio_16k.to(self.device),
            l2_norm=True).unsqueeze(-1).to(self.device))

    @torch.inference_mode()
    def get_conditioning_latents(
        self,
        audio_path,
        max_ref_length=30,
        gpt_cond_len=6,
        gpt_cond_chunk_len=6,
        librosa_trim_db=None,
        sound_norm_refs=False,
        load_sr=22050,
    ):
        """Get the conditioning latents for the GPT model from the given audio.

        Args:
            audio_path (str or List[str]): Path to reference audio file(s).
            max_ref_length (int): Maximum length of each reference audio in seconds. Defaults to 30.
            gpt_cond_len (int): Length of the audio used for gpt latents. Defaults to 6.
            gpt_cond_chunk_len (int): Chunk length used for gpt latents. It must be <= gpt_conf_len. Defaults to 6.
            librosa_trim_db (int, optional): Trim the audio using this value. If None, not trimming. Defaults to None.
            sound_norm_refs (bool, optional): Whether to normalize the audio. Defaults to False.
            load_sr (int, optional): Sample rate to load the audio. Defaults to 24000.
        """
        # deal with multiples references
        if not isinstance(audio_path, list):
            audio_paths = [audio_path]
        else:
            audio_paths = audio_path

        speaker_embeddings = []
        audios = []
        speaker_embedding = None
        for file_path in audio_paths:
            audio = load_audio(file_path, load_sr)
            audio = audio[:, :load_sr * max_ref_length].to(self.device)
            if sound_norm_refs:
                audio = (audio / torch.abs(audio).max()) * 0.75
            if librosa_trim_db is not None:
                audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]

            # compute latents for the decoder
            speaker_embedding = self.get_speaker_embedding(audio, load_sr)
            speaker_embeddings.append(speaker_embedding)

            audios.append(audio)

        # merge all the audios and compute the latents for the gpt
        full_audio = torch.cat(audios, dim=-1)
        gpt_cond_latents = self.get_gpt_cond_latents(
            full_audio,
            load_sr,
            length=gpt_cond_len,
            chunk_length=gpt_cond_chunk_len)  # [1, 1024, T]

        if speaker_embeddings:
            speaker_embedding = torch.stack(speaker_embeddings)
            speaker_embedding = speaker_embedding.mean(dim=0)

        return gpt_cond_latents, speaker_embedding

    def synthesize(self,
                   text,
                   config,
                   speaker_wav,
                   language,
                   speaker_id=None,
                   **kwargs):
        """Synthesize speech with the given input text.

        Args:
            text (str): Input text.
            config (XttsConfig): Config with inference parameters.
            speaker_wav (list): List of paths to the speaker audio files to be used for cloning.
            language (str): Language ID of the speaker.
            **kwargs: Inference settings. See `inference()`.

        Returns:
            A dictionary of the output values with `wav` as output waveform, `deterministic_seed` as seed used at inference,
            `text_input` as text token IDs after tokenizer, `voice_samples` as samples used for cloning, `conditioning_latents`
            as latents used at inference.

        """
        assert (
            "zh-cn" if language == "zh" else language in self.config.languages
        ), f" ❗ Language {language} is not supported. Supported languages are {self.config.languages}"
        # Use generally found best tuning knobs for generation.
        settings = {
            "temperature": config.temperature,
            "length_penalty": config.length_penalty,
            "repetition_penalty": config.repetition_penalty,
            "top_k": config.top_k,
            "top_p": config.top_p,
        }
        settings.update(
            kwargs)  # allow overriding of preset settings with kwargs
        if speaker_id is not None:
            gpt_cond_latent, speaker_embedding = self.speaker_manager.speakers[
                speaker_id].values()
            return self.inference(text, language, gpt_cond_latent,
                                  speaker_embedding, **settings)
        settings.update({
            "gpt_cond_len": config.gpt_cond_len,
            "gpt_cond_chunk_len": config.gpt_cond_chunk_len,
            "max_ref_len": config.max_ref_len,
            "sound_norm_refs": config.sound_norm_refs,
        })
        return self.full_inference(text, speaker_wav, language, **settings)

    @torch.inference_mode()
    def full_inference(
        self,
        text,
        ref_audio_path,
        language,
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        # Cloning
        gpt_cond_len=30,
        gpt_cond_chunk_len=6,
        max_ref_len=10,
        sound_norm_refs=False,
        **hf_generate_kwargs,
    ):
        """
        This function produces an audio clip of the given text being spoken with the given reference voice.

        Args:
            text: (str) Text to be spoken.

            ref_audio_path: (str) Path to a reference audio file to be used for cloning. This audio file should be >3
                seconds long.

            language: (str) Language of the voice to be generated.

            temperature: (float) The softmax temperature of the autoregressive model. Defaults to 0.65.

            length_penalty: (float) A length penalty applied to the autoregressive decoder. Higher settings causes the
                model to produce more terse outputs. Defaults to 1.0.

            repetition_penalty: (float) A penalty that prevents the autoregressive decoder from repeating itself during
                decoding. Can be used to reduce the incidence of long silences or "uhhhhhhs", etc. Defaults to 2.0.

            top_k: (int) K value used in top-k sampling. [0,inf]. Lower values mean the decoder produces more "likely"
                (aka boring) outputs. Defaults to 50.

            top_p: (float) P value used in nucleus sampling. (0,1]. Lower values mean the decoder produces more "likely"
                (aka boring) outputs. Defaults to 0.8.

            gpt_cond_len: (int) Length of the audio used for cloning. If audio is shorter, then audio length is used
                else the first `gpt_cond_len` secs is used. Defaults to 30 seconds.

            gpt_cond_chunk_len: (int) Chunk length used for cloning. It must be <= `gpt_cond_len`.
                If gpt_cond_len == gpt_cond_chunk_len, no chunking. Defaults to 6 seconds.

            hf_generate_kwargs: (**kwargs) The huggingface Transformers generate API is used for the autoregressive
                transformer. Extra keyword args fed to this function get forwarded directly to that API. Documentation
                here: https://huggingface.co/docs/transformers/internal/generation_utils

        Returns:
            Generated audio clip(s) as a torch tensor. Shape 1,S if k=1 else, (k,1,S) where S is the sample length.
            Sample rate is 24kHz.
        """
        (gpt_cond_latent, speaker_embedding) = self.get_conditioning_latents(
            audio_path=ref_audio_path,
            gpt_cond_len=gpt_cond_len,
            gpt_cond_chunk_len=gpt_cond_chunk_len,
            max_ref_length=max_ref_len,
            sound_norm_refs=sound_norm_refs,
        )

        return self.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            **hf_generate_kwargs,
        )

    @torch.inference_mode()
    def inference(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        num_beams=1,
        speed=1.0,
        enable_text_splitting=False,
        **hf_generate_kwargs,
    ):
        language = language.split("-")[0]  # remove the country code
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)
        if enable_text_splitting:
            text = split_sentence(text, language,
                                  self.tokenizer.char_limits[language])
        else:
            text = [text]

        wavs = []
        gpt_latents_list = []
        for sent in text:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(
                self.tokenizer.encode(sent, lang=language)).unsqueeze(0).to(
                    self.device)

            assert (
                text_tokens.shape[-1] < self.args.gpt_max_text_tokens
            ), " ❗ XTTS can only generate text with a maximum of 400 tokens."

            with torch.no_grad():
                gpt_codes = self.gpt.generate(
                    cond_latents=gpt_cond_latent,
                    text_inputs=text_tokens,
                    input_tokens=None,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=self.gpt_batch_size,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    output_attentions=False,
                    **hf_generate_kwargs,
                )
                expected_output_len = torch.tensor(
                    [gpt_codes.shape[-1] * self.gpt.code_stride_len],
                    device=text_tokens.device)

                text_len = torch.tensor([text_tokens.shape[-1]],
                                        device=self.device)
                gpt_latents = self.gpt(
                    text_tokens,
                    text_len,
                    gpt_codes,
                    expected_output_len,
                    cond_latents=gpt_cond_latent,
                    return_attentions=False,
                    return_latent=True,
                )

                if length_scale != 1.0:
                    gpt_latents = F.interpolate(gpt_latents.transpose(1, 2),
                                                scale_factor=length_scale,
                                                mode="linear").transpose(1, 2)

                gpt_latents_list.append(gpt_latents.cpu())
                wavs.append(
                    self.hifigan_decoder(gpt_latents,
                                         g=speaker_embedding).cpu().squeeze())

        return {
            "wav": torch.cat(wavs, dim=0).numpy(),
            "gpt_latents": torch.cat(gpt_latents_list, dim=1).numpy(),
            "speaker_embedding": speaker_embedding,
        }

    def handle_chunks(self, wav_gen, wav_gen_prev, wav_overlap, overlap_len):
        """Handle chunk formatting in streaming mode"""
        wav_chunk = wav_gen[:-overlap_len]
        if wav_gen_prev is not None:
            wav_chunk = wav_gen[(wav_gen_prev.shape[0] -
                                 overlap_len):-overlap_len]
        if wav_overlap is not None:
            # cross fade the overlap section
            if overlap_len > len(wav_chunk):
                # wav_chunk is smaller than overlap_len, pass on last wav_gen
                if wav_gen_prev is not None:
                    wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_len):]
                else:
                    # not expecting will hit here as problem happens on last chunk
                    wav_chunk = wav_gen[-overlap_len:]
                return wav_chunk, wav_gen, None
            else:
                crossfade_wav = wav_chunk[:overlap_len]
                crossfade_wav = crossfade_wav * torch.linspace(
                    0.0, 1.0, overlap_len).to(crossfade_wav.device)
                wav_chunk[:overlap_len] = wav_overlap * torch.linspace(
                    1.0, 0.0, overlap_len).to(wav_overlap.device)
                wav_chunk[:overlap_len] += crossfade_wav

        wav_overlap = wav_gen[-overlap_len:]
        wav_gen_prev = wav_gen
        return wav_chunk, wav_gen_prev, wav_overlap

    @torch.inference_mode()
    def inference_stream(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        # Streaming
        stream_chunk_size=20,
        overlap_wav_len=1024,
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        speed=1.0,
        enable_text_splitting=False,
        **hf_generate_kwargs,
    ):
        language = language.split("-")[0]  # remove the country code
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)
        if enable_text_splitting:
            text = split_sentence(text, language,
                                  self.tokenizer.char_limits[language])
        else:
            text = [text]

        for sent in text:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(
                self.tokenizer.encode(sent, lang=language)).unsqueeze(0).to(
                    self.device)

            assert (
                text_tokens.shape[-1] < self.args.gpt_max_text_tokens
            ), " ❗ XTTS can only generate text with a maximum of 400 tokens."

            fake_inputs = self.gpt.compute_embeddings(
                gpt_cond_latent.to(self.device),
                text_tokens,
            )
            gpt_generator = self.gpt.get_generator(
                fake_inputs=fake_inputs,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=1,
                num_return_sequences=1,
                length_penalty=float(length_penalty),
                repetition_penalty=float(repetition_penalty),
                output_attentions=False,
                output_hidden_states=True,
                **hf_generate_kwargs,
            )

            last_tokens = []
            all_latents = []
            wav_gen_prev = None
            wav_overlap = None
            is_end = False

            while not is_end:
                try:
                    x, latent = next(gpt_generator)
                    last_tokens += [x]
                    all_latents += [latent]
                except StopIteration:
                    is_end = True

                if is_end or (stream_chunk_size > 0
                              and len(last_tokens) >= stream_chunk_size):
                    gpt_latents = torch.cat(all_latents, dim=0)[None, :]
                    if length_scale != 1.0:
                        gpt_latents = F.interpolate(
                            gpt_latents.transpose(1, 2),
                            scale_factor=length_scale,
                            mode="linear").transpose(1, 2)
                    wav_gen = self.hifigan_decoder(gpt_latents,
                                                   g=speaker_embedding.to(
                                                       self.device))
                    wav_chunk, wav_gen_prev, wav_overlap = self.handle_chunks(
                        wav_gen.squeeze(), wav_gen_prev, wav_overlap,
                        overlap_wav_len)
                    last_tokens = []
                    yield wav_chunk

    def forward(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training"
        )

    def eval_step(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training"
        )

    def eval(self):  # pylint: disable=redefined-builtin
        """Sets the model to evaluation mode. Overrides the default eval() method to also set the GPT model to eval mode."""
        self.gpt.init_gpt_for_inference()
        super().eval()

    def get_compatible_checkpoint_state_dict(self, model_path):

        with open(model_path, "rb") as f:
            checkpoint = torch.load(f,
                                    map_location=torch.device("cpu"))["model"]

        ignore_keys = [
            "torch_mel_spectrogram_style_encoder",
            "torch_mel_spectrogram_dvae", "dvae"
        ]
        for key in list(checkpoint.keys()):
            # check if it is from the coqui Trainer if so convert it
            if key.startswith("xtts."):
                new_key = key.replace("xtts.", "")
                checkpoint[new_key] = checkpoint[key]
                del checkpoint[key]
                key = new_key

            # remove unused keys
            if key.split(".")[0] in ignore_keys:
                del checkpoint[key]

        return checkpoint

    def load_checkpoint(
        self,
        checkpoint_dir,
        eval=True,
        strict=True,
        use_deepspeed=False,
        speaker_file_path=None,
    ):
        """
        Loads a checkpoint from disk and initializes the model's state and tokenizer.

        Args:
            config (dict): The configuration dictionary for the model.
            checkpoint_dir (str, optional): The directory where the checkpoint is stored. Defaults to None.
            checkpoint_path (str, optional): The path to the checkpoint file. Defaults to None.
            vocab_path (str, optional): The path to the vocabulary file. Defaults to None.
            eval (bool, optional): Whether to set the model to evaluation mode. Defaults to True.
            strict (bool, optional): Whether to strictly enforce that the keys in the checkpoint match the keys in the model. Defaults to True.

        Returns:
            None
        """

        model_path = os.path.join(checkpoint_dir, "model.pth")
        vocab_path = os.path.join(checkpoint_dir, "vocab.json")

        if os.path.exists(vocab_path):
            self.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)

        self.init_models()

        checkpoint = self.get_compatible_checkpoint_state_dict(model_path)

        self.load_state_dict(checkpoint, strict=strict)

        if eval:
            self.hifigan_decoder.eval()
            self.gpt.init_gpt_for_inference(
                kv_cache=self.args.kv_cache,  # kv_cacke = True
                use_deepspeed=use_deepspeed)
            self.gpt.eval()


@dataclass
class BaseConfig:

    def from_dict(self, data):
        if not isinstance(data, dict):
            raise ValueError()
        data = data.copy()
        init_kwargs = {}
        for _field in fields(self):
            # if field.name == 'dataset_config':
            if _field.name not in data:
                if _field.name in vars(self):
                    init_kwargs[field.name] = vars(self)[_field.name]
                    continue
                raise ValueError(f' [!] Missing required field "{field.name}"')
            #TODO handle default value
            # value = data.get(field.name, _default_value(field))
            value = data.get(field.name, None)
            if value is None:
                init_kwargs[field.name] = value
                continue
            #TODO handle _deserialize
            # value = _deserialize(value, _field.type)
            init_kwargs[field.name] = value
        for k, v in init_kwargs.items():
            setattr(self, k, v)
        return self


@dataclass
class XttsArgs(BaseConfig):
    """A dataclass to represent XTTS model arguments that define the model structure.

    Args:
        gpt_batch_size (int): The size of the auto-regressive batch.
        enable_redaction (bool, optional): Whether to enable redaction. Defaults to True.
        kv_cache (bool, optional): Whether to use the kv_cache. Defaults to True.
        gpt_checkpoint (str, optional): The checkpoint for the autoregressive model. Defaults to None.
        clvp_checkpoint (str, optional): The checkpoint for the ConditionalLatentVariablePerseq model. Defaults to None.
        decoder_checkpoint (str, optional): The checkpoint for the DiffTTS model. Defaults to None.
        num_chars (int, optional): The maximum number of characters to generate. Defaults to 255.

        For GPT model:
        gpt_max_audio_tokens (int, optional): The maximum mel tokens for the autoregressive model. Defaults to 604.
        gpt_max_text_tokens (int, optional): The maximum text tokens for the autoregressive model. Defaults to 402.
        gpt_max_prompt_tokens (int, optional): The maximum prompt tokens or the autoregressive model. Defaults to 70.
        gpt_layers (int, optional): The number of layers for the autoregressive model. Defaults to 30.
        gpt_n_model_channels (int, optional): The model dimension for the autoregressive model. Defaults to 1024.
        gpt_n_heads (int, optional): The number of heads for the autoregressive model. Defaults to 16.
        gpt_number_text_tokens (int, optional): The number of text tokens for the autoregressive model. Defaults to 255.
        gpt_start_text_token (int, optional): The start text token for the autoregressive model. Defaults to 255.
        gpt_checkpointing (bool, optional): Whether to use checkpointing for the autoregressive model. Defaults to False.
        gpt_train_solo_embeddings (bool, optional): Whether to train embeddings for the autoregressive model. Defaults to False.
        gpt_code_stride_len (int, optional): The hop_size of dvae and consequently of the gpt output. Defaults to 1024.
        gpt_use_masking_gt_prompt_approach (bool, optional):  If True, it will use ground truth as prompt and it will mask the loss to avoid repetition. Defaults to True.
        gpt_use_perceiver_resampler (bool, optional):  If True, it will use perceiver resampler from flamingo paper - https://arxiv.org/abs/2204.14198. Defaults to False.
    """

    gpt_batch_size: int = 1
    enable_redaction: bool = False
    kv_cache: bool = True
    gpt_checkpoint: str = None
    clvp_checkpoint: str = None
    decoder_checkpoint: str = None
    num_chars: int = 255

    # XTTS GPT Encoder params
    tokenizer_file: str = ""
    gpt_max_audio_tokens: int = 605
    gpt_max_text_tokens: int = 402
    gpt_max_prompt_tokens: int = 70
    gpt_layers: int = 30
    gpt_n_model_channels: int = 1024
    gpt_n_heads: int = 16
    gpt_number_text_tokens: int = None
    gpt_start_text_token: int = None
    gpt_stop_text_token: int = None
    gpt_num_audio_tokens: int = 8194
    gpt_start_audio_token: int = 8192
    gpt_stop_audio_token: int = 8193
    gpt_code_stride_len: int = 1024
    gpt_use_masking_gt_prompt_approach: bool = True
    gpt_use_perceiver_resampler: bool = False

    # HifiGAN Decoder params
    input_sample_rate: int = 22050
    output_sample_rate: int = 24000
    output_hop_length: int = 256
    decoder_input_dim: int = 1024
    d_vector_dim: int = 512
    cond_d_vector_in_each_upsampling_layer: bool = True

    # constants
    duration_const: int = 102400


@dataclass
class XttsAudioConfig(BaseConfig):
    """
    Configuration class for audio-related parameters in the XTTS model.

    Args:
        sample_rate (int): The sample rate in which the GPT operates.
        output_sample_rate (int): The sample rate of the output audio waveform.
    """

    sample_rate: int = 22050
    output_sample_rate: int = 24000


@dataclass
class XTTSConfig(BaseConfig):
    num_loader_workers: int = 0
    num_eval_loader_workers: int = 0
    use_noise_augment: bool = False
    use_phonemes: bool = False
    phonemizer: str = None
    phoneme_language: str = None
    compute_input_seq_cache: bool = False
    text_cleaner: str = None
    enable_eos_bos_chars: bool = False
    test_sentences_file: str = ""
    phoneme_cache_path: str = None
    # vocabulary parameters
    add_blank: bool = False
    # training params
    batch_group_size: int = 0
    loss_masking: bool = None
    # dataloading
    min_audio_len: int = 1
    max_audio_len: int = float("inf")
    min_text_len: int = 1
    max_text_len: int = float("inf")
    compute_f0: bool = False
    compute_energy: bool = False
    compute_linear_spec: bool = False
    precompute_num_workers: int = 0
    use_noise_augment: bool = False
    start_by_longest: bool = False
    shuffle: bool = False
    drop_last: bool = False
    # dataset
    # optimizer
    # evaluation
    eval_split_max_size: int = None
    eval_split_size: float = 0.01
    # weighted samplers
    use_speaker_weighted_sampler: bool = False
    speaker_weighted_sampler_alpha: float = 1.0
    use_language_weighted_sampler: bool = False
    language_weighted_sampler_alpha: float = 1.0
    use_length_weighted_sampler: bool = False
    length_weighted_sampler_alpha: float = 1.0
    model: str = "xtts"
    model_args = field(default_factory=XttsArgs)
    audio = field(default_factory=XttsAudioConfig)
    model_dir = None
    languages: list[str] = field(default_factory=lambda: [
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "pl",
        "tr",
        "ru",
        "nl",
        "cs",
        "ar",
        "zh-cn",
        "hu",
        "ko",
        "ja",
        "hi",
    ])

    # inference params
    temperature: float = 0.85
    length_penalty: float = 1.0
    repetition_penalty: float = 2.0
    top_k: int = 50
    top_p: float = 0.85
    num_gpt_outputs: int = 1

    # cloning
    gpt_cond_len: int = 12
    gpt_cond_chunk_len: int = 4
    max_ref_len: int = 10
    sound_norm_refs: bool = False
