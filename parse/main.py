from io import BytesIO
import os
import re
import time
import json
import yaml
import fsspec
import importlib

import scipy
import numpy as np

import torch
import torch.nn as nn


def save_wav(*,
             wav: np.ndarray,
             path: str,
             sample_rate: int = None,
             pipe_out=None,
             **kwargs) -> None:
    """Save float waveform to a file using Scipy.

    Args:
        wav (np.ndarray): Waveform with float values in range [-1, 1] to save.
        path (str): Path to a output file.
        sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
        pipe_out (BytesIO, optional): Flag to stdout the generated TTS wav file for shell pipe.
    """
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

    wav_norm = wav_norm.astype(np.int16)
    if pipe_out:
        wav_buffer = BytesIO()
        scipy.io.wavfile.write(wav_buffer, sample_rate, wav_norm)
        wav_buffer.seek(0)
        pipe_out.buffer.write(wav_buffer.read())
    scipy.io.wavfile.write(path, sample_rate, wav_norm)


def trim_silence(wav, ap):
    return wav[:ap.find_endpoint(wav)]


def interpolate_vocoder_input(scale_factor, spec):
    """Interpolate spectrogram by the scale factor.
    It is mainly used to match the sampling rates of
    the tts and vocoder models.

    Args:
        scale_factor (float): scale factor to interpolate the spectrogram
        spec (np.array): spectrogram to be interpolated

    Returns:
        torch.tensor: interpolated spectrogram.
    """
    print(" > before interpolation :", spec.shape)
    spec = torch.tensor(spec).unsqueeze(0).unsqueeze(0)  # pylint: disable=not-callable
    spec = torch.nn.functional.interpolate(spec,
                                           scale_factor=scale_factor,
                                           recompute_scale_factor=True,
                                           mode="bilinear",
                                           align_corners=False).squeeze(0)
    print(" > after interpolation :", spec.shape)
    return spec


def to_camel(text):
    text = text.capitalize()
    text = re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), text)
    text = text.replace("Tts", "TTS")
    text = text.replace("vc", "VC")
    return text


def find_module(module_path: str, module_name: str):
    module_name = module_name.lower()
    module = importlib.import_module(module_path + "." + module_name)
    class_name = to_camel(module_name)
    return getattr(module, class_name)


def setup_model(config, samples=None):
    print(" > Using model: {}".format(config.model))
    MyModel = find_module("TTS.tts.models", config.model.lower())
    model = MyModel.init_from_config(config=config, samples=samples)
    return model


def read_json_with_comments(json_path):
    """for backward compat."""
    # fallback to json
    with fsspec.open(json_path, "r", encoding="utf-8") as f:
        input_str = f.read()
    # handle comments but not urls with //
    input_str = re.sub(
        r"(\"(?:[^\"\\]|\\.)*\")|(/\*(?:.|[\\n\\r])*?\*/)|(//.*)",
        lambda m: m.group(1) or m.group(2) or "", input_str)
    return json.loads(input_str)


def load_config(config_path: str):
    """Import `json` or `yaml` files as TTS configs. First, load the input file as a `dict` and check the model name
    to find the corresponding Config class. Then initialize the Config.

    Args:
        config_path (str): path to the config file.

    Raises:
        TypeError: given config file has an unknown type.

    Returns:
        Coqpit: TTS config object.
    """
    config_dict = {}
    ext = os.path.splitext(config_path)[1]
    if ext in (".yml", ".yaml"):
        with fsspec.open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif ext == ".json":
        try:
            with fsspec.open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.decoder.JSONDecodeError:
            # backwards compat.
            data = read_json_with_comments(config_path)
    else:
        raise TypeError(f" [!] Unknown config file type {ext}")
    config_dict.update(data)
    model_name = _process_model_name(config_dict)
    config_class = register_config(model_name.lower())
    config = config_class()
    config.from_dict(config_dict)
    return config


def _process_model_name(config_dict: dict) -> str:
    """Format the model name as expected. It is a band-aid for the old `vocoder` model names.

    Args:
        config_dict (Dict): A dictionary including the config fields.

    Returns:
        str: Formatted modelname.
    """
    model_name = config_dict[
        "model"] if "model" in config_dict else config_dict["generator_model"]
    model_name = model_name.replace("_generator",
                                    "").replace("_discriminator", "")
    return model_name


def register_config(model_name: str):
    """Find the right config for the given model name.

    Args:
        model_name (str): Model name.

    Raises:
        ModuleNotFoundError: No matching config for the model name.

    Returns:
        Coqpit: config class.
    """
    config_class = None
    config_name = model_name + "_config"

    if model_name == "xtts":
        from TTS.tts.configs.xtts_config import XttsConfig

        config_class = XttsConfig
    paths = [
        "TTS.tts.configs", "TTS.vocoder.configs", "TTS.encoder.configs",
        "TTS.vc.configs"
    ]
    for path in paths:
        try:
            config_class = find_module(path, config_name)
        except ModuleNotFoundError:
            pass
    if config_class is None:
        raise ModuleNotFoundError(
            f" [!] Config for {model_name} cannot be found.")
    return config_class


class Synthesizer(nn.Module):

    def __init__(self, tts_checkpoint="", tts_config_path="", use_cuda=False):
        super().__init__()
        self.tts_checkpoint = tts_checkpoint
        self.tts_config_path = tts_config_path

        self.use_cuda = use_cuda

        self.tts_model = None
        # self.vocoder_model = None
        # self.vc_model = None
        # self.speaker_manager = None
        # self.tts_speakers = {}
        self.language_manager = None
        self.num_languages = 0
        self.tts_languages = {}
        self.d_vector_dim = 0
        self.seg = self._get_segmenter("en")

        if self.use_cuda:
            assert torch.cuda.is_available(
            ), "CUDA is not availabe on this machine."

        if tts_checkpoint:
            self._load_tts(tts_checkpoint, tts_config_path, use_cuda)
            self.output_sample_rate = self.tts_config.audio["sample_rate"]

    def _load_tts(self, tts_checkpoint: str, tts_config_path: str,
                  use_cuda: bool) -> None:

        self.tts_config = load_config(tts_config_path)

        self.tts_model = setup_model(config=self.tts_config)

        self.tts_model.load_checkpoint(self.tts_config,
                                       tts_checkpoint,
                                       eval=True)
        if use_cuda:
            self.tts_model.cuda()

    def tts(
        self,
        text: str = "",
        language_name: str = "en",
        speaker_wav=None,
        # style_wav=None,
        # style_text=None,
        # reference_wav=None,
        # reference_speaker_name=None,
        split_sentences: bool = True,
        **kwargs,
    ) -> list[int]:
        """🐸 TTS magic. Run all the models and generate speech.

        Args:
            text (str): input text.
            speaker_name (str, optional): speaker id for multi-speaker models. Defaults to "".
            language_name (str, optional): language id for multi-language models. Defaults to "".
            speaker_wav (Union[str, List[str]], optional): path to the speaker wav for voice cloning. Defaults to None.
            style_wav ([type], optional): style waveform for GST. Defaults to None.
            style_text ([type], optional): transcription of style_wav for Capacitron. Defaults to None.
            reference_wav ([type], optional): reference waveform for voice conversion. Defaults to None.
            reference_speaker_name ([type], optional): speaker id of reference waveform. Defaults to None.
            split_sentences (bool, optional): split the input text into sentences. Defaults to True.
            **kwargs: additional arguments to pass to the TTS model.
        Returns:
            List[int]: [description]
        """
        start_time = time.time()
        wavs = []

        if text:
            sens = [text]
            if split_sentences:
                print(" > Text splitted to sentences.")
                sens = self.split_into_sentences(text)
            print(sens)

        # handle multi-speaker
        # if "voice_dir" in kwargs:
        #     self.voice_dir = kwargs["voice_dir"]
        #     kwargs.pop("voice_dir")
        # speaker_embedding = None
        # speaker_id = None

        # handle multi-lingual

        # compute a new d_vector from the given clip.

        vocoder_device = "cpu"
        use_gl = self.vocoder_model is None
        if not use_gl:
            vocoder_device = next(self.vocoder_model.parameters()).device
        if self.use_cuda:
            vocoder_device = "cuda"

        for sen in sens:
            outputs = self.tts_model.synthesize(
                text=sen,
                config=self.tts_config,
                speaker_wav=speaker_wav,
                language=language_name,
                **kwargs,
            )
            waveform = outputs["wav"]
            if not use_gl:
                mel_postnet_spec = outputs["outputs"]["model_outputs"][
                    0].detach().cpu().numpy()
                # denormalize tts output based on tts audio config
                mel_postnet_spec = self.tts_model.ap.denormalize(
                    mel_postnet_spec.T).T
                # renormalize spectrogram based on vocoder config
                vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                # compute scale factor for possible sample rate mismatch
                scale_factor = [
                    1,
                    self.vocoder_config["audio"]["sample_rate"] /
                    self.tts_model.ap.sample_rate,
                ]
                if scale_factor[1] != 1:
                    print(" > interpolating tts model output.")
                    vocoder_input = interpolate_vocoder_input(
                        scale_factor, vocoder_input)
                else:
                    vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                # run vocoder model
                # [1, T, C]
                waveform = self.vocoder_model.inference(
                    vocoder_input.to(vocoder_device))
            if torch.is_tensor(waveform) and waveform.device != torch.device(
                    "cpu") and not use_gl:
                waveform = waveform.cpu()
            if not use_gl:
                waveform = waveform.numpy()
            waveform = waveform.squeeze()

            # trim silence
            if "do_trim_silence" in self.tts_config.audio and self.tts_config.audio[
                    "do_trim_silence"]:
                waveform = trim_silence(waveform, self.tts_model.ap)

            wavs += list(waveform)
            wavs += [0] * 10000

        # compute stats
        process_time = time.time() - start_time
        audio_time = len(wavs) / self.tts_config.audio["sample_rate"]
        print(f" > Processing time: {process_time}")
        print(f" > Real-time factor: {process_time / audio_time}")
        return wavs

    def save_wav(self, wav: list[int], path: str, pipe_out=None) -> None:
        """Save the waveform as a file.

        Args:
            wav (List[int]): waveform as a list of values.
            path (str): output path to save the waveform.
            pipe_out (BytesIO, optional): Flag to stdout the generated TTS wav file for shell pipe.
        """
        # if tensor convert to numpy
        new_wav = np.array(wav)
        save_wav(wav=new_wav,
                 path=path,
                 sample_rate=self.output_sample_rate,
                 pipe_out=pipe_out)


class TTS(nn.Module):
    """TODO: Add voice conversion and Capacitron support."""

    def __init__(
        self,
        model_path: str = None,
        config_path: str = None,
        gpu=False,
    ):
        super().__init__()
        self.config = load_config(config_path) if config_path else None
        self.synthesizer: Synthesizer | None = None
        if model_path:
            self.load_tts_model_by_path(model_path, config_path, gpu=gpu)

    def load_tts_model_by_path(self,
                               model_path: str,
                               config_path: str = None,
                               gpu: bool = False):

        self.synthesizer = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            use_cuda=gpu,
        )

    def tts(
        self,
        text: str,
        # speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        # emotion: str = None,
        # speed: float = None,
        split_sentences: bool = True,
        **kwargs,
    ):
        """Convert text to speech.

        Args:
            text (str):
                Input text to synthesize.
            speaker (str, optional):
                Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
                `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
            language (str): Language of the text. If None, the default language of the speaker is used. Language is only
                supported by `XTTS` model.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
            emotion (str, optional):
                Emotion to use for 🐸Coqui Studio models. If None, Studio models use "Neutral". Defaults to None.
            speed (float, optional):
                Speed factor to use for 🐸Coqui Studio models, between 0 and 2.0. If None, Studio models use 1.0.
                Defaults to None.
            split_sentences (bool, optional):
                Split text into sentences, synthesize them separately and concatenate the file audio.
                Setting it False uses more VRAM and possibly hit model specific text length or VRAM limits. Only
                applicable to the 🐸TTS models. Defaults to True.
            kwargs (dict, optional):
                Additional arguments for the model.
        """
        if self.synthesizer is None:
            return
        wav = self.synthesizer.tts(
            text=text,
            # speaker_name=speaker or "",
            language_name=language or "en",
            speaker_wav=speaker_wav,
            # reference_wav=None,
            # style_wav=None,
            # style_text=None,
            # reference_speaker_name=None,
            split_sentences=split_sentences,
            **kwargs,
        )
        return wav

    def tts_to_file(
        self,
        text: str,
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        # emotion: str = None,
        # speed: float = 1.0,
        # pipe_out=None,
        file_path: str = "output.wav",
        split_sentences: bool = True,
        **kwargs,
    ):
        """Convert text to speech.

            Args:
                text (str):
                    Input text to synthesize.
                speaker (str, optional):
                    Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
                    `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
                language (str, optional):
                    Language code for multi-lingual models. You can check whether loaded model is multi-lingual
                    `tts.is_multi_lingual` and list available languages by `tts.languages`. Defaults to None.
                speaker_wav (str, optional):
                    Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                    Defaults to None.
                emotion (str, optional):
                    Emotion to use for 🐸Coqui Studio models. Defaults to "Neutral".
                speed (float, optional):
                    Speed factor to use for 🐸Coqui Studio models, between 0.0 and 2.0. Defaults to None.
                pipe_out (BytesIO, optional):
                    Flag to stdout the generated TTS wav file for shell pipe.
                file_path (str, optional):
                    Output file path. Defaults to "output.wav".
                split_sentences (bool, optional):
                    Split text into sentences, synthesize them separately and concatenate the file audio.
                    Setting it False uses more VRAM and possibly hit model specific text length or VRAM limits. Only
                    applicable to the 🐸TTS models. Defaults to True.
                kwargs (dict, optional):
                    Additional arguments for the model.
            """

        wav = self.tts(
            text=text,
            speaker=speaker,
            language=language,
            speaker_wav=speaker_wav,
            split_sentences=split_sentences,
            **kwargs,
        )
        if self.synthesizer is None:
            return
        self.synthesizer.save_wav(wav=wav, path=file_path)
        return file_path
