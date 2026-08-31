from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Fixed audio protocol shared by training and inference.
AUDIO_SAMPLE_RATE = 16_000
SEGMENT_SECONDS = 15.0
SEGMENT_SAMPLES = int(AUDIO_SAMPLE_RATE * SEGMENT_SECONDS)
TARGET_INTERVAL_SECONDS = 0.5
FEATURE_PROTOCOL_VERSION = "daridge-audio-v1"
HEAD_DIM = 128
PROJECTION_PATTERN = "self_attn.o_proj"


# Prompts and adjective groups define the semantic evidence space.
BASE_PROMPT = "Estimate the continuous valence and arousal of this music segment."
PROMPT_TEMPLATES = (
    (
        "Complete the sentence with one adjective. "
        "The emotional tone conveyed by this music is"
    ),
    (
        "Describe the perceived emotional character of this music with one adjective. "
        "This music feels"
    ),
)
ADJECTIVE_GROUPS = {
    "positive_low": ("peaceful", "calm", "relaxed"),
    "positive_high": ("joyful", "exciting", "upbeat"),
    "negative_low": ("sad", "lonely", "bleak"),
    "negative_high": ("angry", "tense", "anxious"),
    "neutral": ("neutral", "ambiguous"),
}
CANDIDATE_ADJECTIVES = tuple(
    adjective for adjectives in ADJECTIVE_GROUPS.values() for adjective in adjectives
)


ACOUSTIC_WINDOWS = (2, 4, 8, 16, 32)


@dataclass(frozen=True)
class AudioFeatures:
    target_times: np.ndarray
    base: np.ndarray
    semantic: np.ndarray
    heads: np.ndarray
    acoustic: np.ndarray

    def __post_init__(self):
        target_times = np.asarray(self.target_times, dtype=np.float32)
        branches = {
            "base": np.asarray(self.base, dtype=np.float32),
            "semantic": np.asarray(self.semantic, dtype=np.float32),
            "heads": np.asarray(self.heads, dtype=np.float32),
            "acoustic": np.asarray(self.acoustic, dtype=np.float32),
        }
        if target_times.ndim != 1 or not len(target_times):
            raise ValueError("target_times must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(target_times)):
            raise ValueError("target_times must be finite")
        for name, value in branches.items():
            if value.ndim != 2 or value.shape[0] != len(target_times):
                raise ValueError(name + " must have one row per target time")
            if not np.all(np.isfinite(value)):
                raise ValueError(name + " features must be finite")
        if branches["semantic"].shape[1] != 8:
            raise ValueError("semantic features must have exactly eight columns")
        if branches["acoustic"].shape[1] != 132:
            raise ValueError("acoustic features must have exactly 132 columns")
        if branches["heads"].shape[1] % HEAD_DIM:
            raise ValueError("head features must be concatenated 128-dimensional heads")
        object.__setattr__(self, "target_times", target_times)
        for name, value in branches.items():
            object.__setattr__(self, name, value)


def build_target_times(duration_seconds):
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError) as error:
        raise ValueError("audio duration must be a finite number") from error
    if not np.isfinite(duration):
        raise ValueError("audio duration must be a finite number")
    if duration < SEGMENT_SECONDS:
        raise ValueError("audio must be at least 15 seconds long")
    count = int(np.floor((duration - SEGMENT_SECONDS) / TARGET_INTERVAL_SECONDS)) + 1
    return (
        SEGMENT_SECONDS + np.arange(count, dtype=np.float32) * np.float32(TARGET_INTERVAL_SECONDS)
    ).astype(np.float32)


def decode_audio(path, sampling_rate=AUDIO_SAMPLE_RATE):
    try:
        requested_rate = int(sampling_rate)
    except (TypeError, ValueError) as error:
        raise ValueError("sampling_rate must be a positive integer") from error
    if requested_rate <= 0 or requested_rate != sampling_rate:
        raise ValueError("sampling_rate must be a positive integer")
    try:
        import av
        import torch
        import torchaudio
    except ImportError as error:
        raise RuntimeError("PyAV and torchaudio are required to decode audio") from error

    channel_chunks = []
    source_rate = None
    with av.open(str(Path(path))) as container:
        if not container.streams.audio:
            raise ValueError("input file has no audio stream")
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            frame_rate = int(frame.sample_rate)
            if source_rate is None:
                source_rate = frame_rate
            elif frame_rate != source_rate:
                raise ValueError("audio sample rate changed during decoding")

            values = frame.to_ndarray()
            channel_count = len(frame.layout.channels)
            if frame.format.is_planar:
                values = values.reshape(channel_count, -1)
            else:
                values = values.reshape(-1, channel_count).T
            if np.issubdtype(values.dtype, np.integer):
                limits = np.iinfo(values.dtype)
                scale = float(max(abs(limits.min), limits.max))
                values = values.astype(np.float32) / scale
            else:
                values = values.astype(np.float32, copy=False)
            channel_chunks.append(values)

    if source_rate is None or not channel_chunks:
        raise ValueError("decoded audio must be non-empty")
    channels = np.concatenate(channel_chunks, axis=1)
    waveform = torch.from_numpy(channels.mean(axis=0, dtype=np.float32))
    if int(source_rate) != requested_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            int(source_rate),
            requested_rate,
        )
    waveform = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
    if waveform.ndim != 1 or not len(waveform):
        raise ValueError("decoded audio must be a non-empty mono waveform")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("decoded audio must contain only finite samples")
    return waveform


def load_music_segment(path, start, end):
    start_value = float(start)
    end_value = float(end)
    if end_value <= start_value:
        raise ValueError("segment end must be greater than start")
    try:
        import librosa
    except ImportError as error:
        raise RuntimeError("librosa is required for Music Flamingo audio") from error
    duration = end_value - start_value
    audio, _ = librosa.load(
        str(Path(path)),
        sr=AUDIO_SAMPLE_RATE,
        mono=True,
        offset=max(0.0, start_value),
        duration=duration,
    )
    expected_samples = round(duration * AUDIO_SAMPLE_RATE)
    missing_samples = expected_samples - audio.size
    max_padding_samples = round(0.5 * AUDIO_SAMPLE_RATE)
    if missing_samples > max_padding_samples:
        raise ValueError("audio segment exceeds terminal padding tolerance")
    if missing_samples > 0:
        audio = np.pad(audio, (0, missing_samples), mode="constant")
    elif missing_samples < 0:
        audio = audio[:expected_samples]
    return np.asarray(audio, dtype=np.float32)


def _log_mean_exp(values, axis):
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(
        maximum + np.log(np.mean(np.exp(values - maximum), axis=axis, keepdims=True)),
        axis=axis,
    )


# Subtract null-audio logits before aggregating adjective evidence.
def build_semantic_features(real_logits, null_logits):
    real = np.asarray(real_logits, dtype=np.float64)
    null = np.asarray(null_logits, dtype=np.float64)
    if real.ndim != 3 or real.shape[1:] != (2, 14) or not len(real):
        raise ValueError("real_logits must have shape [n, 2, 14]")
    if null.shape != (2, 14):
        raise ValueError("null_logits must have shape [2, 14]")
    if not np.all(np.isfinite(real)) or not np.all(np.isfinite(null)):
        raise ValueError("semantic logits must be finite")

    calibrated = real - null[None, :, :]
    evidence = np.empty((len(real), 2, len(ADJECTIVE_GROUPS)), dtype=np.float64)
    for group_index, adjectives in enumerate(ADJECTIVE_GROUPS.values()):
        indices = [CANDIDATE_ADJECTIVES.index(adjective) for adjective in adjectives]
        evidence[:, :, group_index] = _log_mean_exp(calibrated[:, :, indices], axis=2)

    group_evidence = np.mean(evidence, axis=1)
    valence_low = group_evidence[:, 0] - group_evidence[:, 2]
    valence_high = group_evidence[:, 1] - group_evidence[:, 3]
    valence_balanced = 0.5 * (valence_low + valence_high)
    arousal_disagreement = np.abs(valence_low - valence_high)
    arousal_evidence = 0.5 * (
        group_evidence[:, 1] + group_evidence[:, 3] - group_evidence[:, 0] - group_evidence[:, 2]
    )
    shifted = group_evidence - np.max(group_evidence, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    entropy = -np.sum(probabilities * np.log(probabilities), axis=1) / np.log(5.0)
    return np.column_stack(
        (
            valence_low,
            valence_high,
            valence_balanced,
            arousal_disagreement,
            arousal_evidence,
            group_evidence[:, 4],
            entropy,
            valence_balanced * (1.0 - entropy),
        )
    ).astype(np.float32)


def _validate_acoustic_inputs(waveform, sample_rate, target_times):
    try:
        rate = int(sample_rate)
    except (TypeError, ValueError) as error:
        raise ValueError("sample_rate must be a positive integer") from error
    audio = np.asarray(waveform, dtype=np.float32)
    times = np.asarray(target_times, dtype=np.float64)
    if rate <= 0 or rate != sample_rate:
        raise ValueError("sample_rate must be a positive integer")
    if audio.ndim != 1 or not len(audio) or not np.all(np.isfinite(audio)):
        raise ValueError("waveform must be a non-empty finite one-dimensional array")
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)):
        raise ValueError("target_times must be a non-empty finite one-dimensional array")
    if np.any(times[1:] < times[:-1]):
        raise ValueError("target_times must be sorted")
    if len(times) > 1 and not np.allclose(np.diff(times), 0.5, rtol=1e-4, atol=1e-4):
        raise ValueError("target_times must be spaced at 0.5 seconds")
    if np.any((times <= 0) | (times > len(audio) / float(rate))):
        raise ValueError("target_times must satisfy 0 < time <= duration")
    return audio, rate, times


# Append first differences and causal rolling statistics to 11 frame features.
def _append_causal_statistics(base):
    delta = np.zeros_like(base)
    delta[1:] = base[1:] - base[:-1]
    parts = [base, delta]
    for window in ACOUSTIC_WINDOWS:
        means = np.empty_like(base)
        stds = np.empty_like(base)
        for index in range(len(base)):
            history = base[max(0, index - window + 1) : index + 1]
            means[index] = history.mean(axis=0)
            stds[index] = history.std(axis=0, ddof=0)
        parts.extend((means, stds))
    return np.concatenate(parts, axis=1).astype(np.float32)


# Compute 11 low-level descriptors for each 0.5-second frame.
def _extract_acoustic_features_torch(audio, sample_rate, target_times, device):
    import torch

    waveform = torch.as_tensor(audio, dtype=torch.float32, device=device)
    times = torch.as_tensor(target_times, dtype=torch.float64, device=waveform.device)
    frame_length = max(1, round(0.5 * sample_rate))
    end_indices = torch.round(times * sample_rate).to(torch.long)
    offsets = torch.arange(-frame_length, 0, device=waveform.device)
    indices = end_indices[:, None] + offsets[None, :]
    valid = (indices >= 0) & (indices < waveform.numel())
    frames = waveform[indices.clamp(0, waveform.numel() - 1)] * valid.to(waveform.dtype)
    window = torch.hann_window(frame_length, device=waveform.device)
    frequencies = torch.fft.rfftfreq(frame_length, d=1.0 / sample_rate).to(waveform.device)
    magnitude = torch.fft.rfft(frames * window[None, :], dim=1).abs()
    power = magnitude.square() + 1e-12
    normalized = magnitude / magnitude.sum(dim=1, keepdim=True).clamp_min(1e-12)
    rms = frames.square().mean(dim=1).sqrt()
    zcr = (frames[:, 1:] * frames[:, :-1] < 0).float().mean(dim=1)
    difference = torch.zeros_like(normalized)
    difference[1:] = normalized[1:] - normalized[:-1]
    centroid = (frequencies[None, :] * normalized).sum(dim=1)
    bandwidth = (
        ((frequencies[None, :] - centroid[:, None]).square() * normalized).sum(dim=1).sqrt()
    )
    cumulative = torch.cumsum(power, dim=1)
    rolloff = frequencies[(cumulative >= cumulative[:, -1:] * 0.85).to(torch.int64).argmax(dim=1)]
    low = power[:, frequencies < 250].mean(dim=1)
    mid = power[:, (frequencies >= 250) & (frequencies < 2000)].mean(dim=1)
    high = power[:, frequencies >= 2000].mean(dim=1)
    nyquist = float(sample_rate) / 2.0
    base = (
        torch.stack(
            (
                torch.log1p(rms),
                zcr,
                difference.square().sum(dim=1).sqrt(),
                difference.clamp_min(0).sum(dim=1),
                centroid / nyquist,
                bandwidth / nyquist,
                rolloff / nyquist,
                torch.log1p(low),
                torch.log1p(mid),
                torch.log1p(high),
                torch.log1p(high) - torch.log1p(low),
            ),
            dim=1,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    return _append_causal_statistics(base)


def extract_acoustic_features(waveform, sample_rate, target_times, device="cpu"):
    audio, rate, times = _validate_acoustic_inputs(waveform, sample_rate, target_times)
    return _extract_acoustic_features_torch(audio, rate, times, device)


class FrozenAudioExtractor:

    def __init__(
        self,
        backbone_path,
        selected_heads,
        device="cuda:0",
        semantic_device="cuda:1",
    ):
        self.backbone_path = Path(backbone_path)
        self.selected_heads = self._normalize_selected_heads(selected_heads)
        self.requested_base_device = str(device)
        self.requested_semantic_device = str(semantic_device)
        self._torch = None
        self._base_device = None
        self._semantic_device = None
        self._base_dtype = None
        self._semantic_dtype = None
        self._processor = None
        self._base_model = None
        self._semantic_model = None
        self._projection_modules = ()
        self._adjective_token_ids = None
        self._null_logits = None
        self._null_audio_token_counts = None
        self._null_audio_pool_dimension = None

    @staticmethod
    def _normalize_selected_heads(selected_heads):
        result = []
        for item in selected_heads:
            if isinstance(item, dict):
                layer, head = item["layer"], item["head"]
            else:
                layer, head = item
            if int(layer) < 0 or int(head) < 0:
                raise ValueError("selected head coordinates must be non-negative")
            result.append((int(layer), int(head)))
        if not result:
            raise ValueError("at least one selected head is required")
        return tuple(result)

    def _load_runtime(self):
        if self._torch is not None:
            return
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "torch is required for Music Flamingo feature extraction"
            ) from error
        self._torch = torch
        self._base_device = self._resolve_device(
            self.requested_base_device,
        )
        self._semantic_device = self._resolve_device(
            self.requested_semantic_device,
        )
        self._base_dtype = torch.bfloat16 if self._base_device.type == "cuda" else torch.float32
        self._semantic_dtype = (
            torch.bfloat16 if self._semantic_device.type == "cuda" else torch.float32
        )

    def _resolve_device(self, requested):
        requested_name = str(requested)
        if requested_name == "cpu":
            return self._torch.device("cpu")
        if not requested_name.startswith("cuda"):
            raise ValueError("device must be cpu or cuda")
        if not self._torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is unavailable")
        device = self._torch.device(requested_name)
        index = 0 if device.index is None else int(device.index)
        if index >= self._torch.cuda.device_count():
            raise ValueError("requested CUDA device is unavailable")
        return self._torch.device("cuda:" + str(index))

    def _freeze(self, model):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    # Load two frozen views of the same backbone on the requested devices.
    def _load_models(self):
        if self._base_model is not None and self._semantic_model is not None:
            return
        self._load_runtime()
        try:
            from transformers import (
                AutoModel,
                AutoProcessor,
                MusicFlamingoForConditionalGeneration,
            )
        except ImportError as error:
            raise RuntimeError("transformers with Music Flamingo support is required") from error
        base_load_kwargs = {
            "trust_remote_code": True,
            "dtype": self._base_dtype,
            "low_cpu_mem_usage": True,
        }
        if self._base_device.type == "cuda":
            base_load_kwargs["device_map"] = {"": self._base_device.index or 0}
        self._base_model = AutoModel.from_pretrained(
            str(self.backbone_path),
            **base_load_kwargs,
        )
        semantic_load_kwargs = {
            "trust_remote_code": True,
            "dtype": self._semantic_dtype,
            "low_cpu_mem_usage": True,
        }
        if self._semantic_device.type == "cuda":
            semantic_load_kwargs["device_map"] = {"": self._semantic_device.index or 0}
        self._semantic_model = MusicFlamingoForConditionalGeneration.from_pretrained(
            str(self.backbone_path),
            **semantic_load_kwargs,
        )
        if self._base_device.type == "cpu":
            self._base_model.to(self._base_device)
        if self._semantic_device.type == "cpu":
            self._semantic_model.to(self._semantic_device)
        self._freeze(self._base_model)
        self._freeze(self._semantic_model)
        self._processor = AutoProcessor.from_pretrained(
            str(self.backbone_path), trust_remote_code=True
        )
        self._projection_modules = tuple(
            (name, module)
            for name, module in self._base_model.named_modules()
            if name and PROJECTION_PATTERN in name.lower()
        )
        if not self._projection_modules:
            raise RuntimeError("no self_attn.o_proj modules were found in the backbone")
        self._adjective_token_ids = self._resolve_adjective_token_ids(self._processor.tokenizer)

    @staticmethod
    def _resolve_adjective_token_ids(tokenizer):
        token_ids = {}
        used_ids = set()
        for adjective in CANDIDATE_ADJECTIVES:
            encoded = tokenizer.encode(" " + adjective, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError("each candidate adjective must map to one leading-space token")
            token_id = int(encoded[0])
            if token_id in used_ids:
                raise ValueError("candidate adjectives must have unique token IDs")
            token_ids[adjective] = token_id
            used_ids.add(token_id)
        return token_ids

    def _prepare_inputs(self, audio, prompt, device, dtype):
        text = [str(self._processor.audio_token) + "\n" + prompt]
        inputs = self._processor(
            text=text,
            audio=[audio],
            sampling_rate=AUDIO_SAMPLE_RATE,
            return_tensors="pt",
        )
        prepared = {}
        for key, value in inputs.items():
            if self._torch.is_tensor(value):
                if key == "input_features":
                    prepared[key] = value.to(device=device, dtype=dtype)
                else:
                    prepared[key] = value.to(device)
            else:
                prepared[key] = value
        return prepared

    # Capture last-token projection inputs as per-head features.
    def _extract_base_and_heads(self, audio):
        handles = []
        captures = {}

        def make_hook(name):

            def hook(_module, input_args):
                if not input_args or not self._torch.is_tensor(input_args[0]):
                    return
                value = input_args[0]
                if value.ndim == 3:
                    captures[name] = value[:, -1, :].detach().cpu()
                elif value.ndim == 2:
                    captures[name] = value.detach().cpu()

            return hook

        try:
            for name, module in self._projection_modules:
                handles.append(module.register_forward_pre_hook(make_hook(name)))
            with self._torch.inference_mode():
                output = self._base_model(
                    **self._prepare_inputs(
                        audio,
                        BASE_PROMPT,
                        self._base_device,
                        self._base_dtype,
                    ),
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
        finally:
            for handle in handles:
                handle.remove()
        if not hasattr(output, "hidden_states") or output.hidden_states is None:
            raise RuntimeError("Music Flamingo base output does not contain hidden_states")
        missing = [name for name, _module in self._projection_modules if name not in captures]
        if missing:
            raise RuntimeError("no projection input captured for " + missing[0])
        head_flat = self._torch.stack(
            [captures[name] for name, _module in self._projection_modules], dim=1
        ).squeeze(0)
        if head_flat.ndim != 2 or head_flat.shape[1] % HEAD_DIM:
            raise RuntimeError("projection inputs cannot be reshaped to 128-dimensional heads")
        head_features = head_flat.reshape(head_flat.shape[0], -1, HEAD_DIM).float().numpy()
        global_context = output.hidden_states[-1][:, -1, :].detach().float().cpu().numpy()
        return global_context[0], head_features

    def _semantic_forward(self, audio, prompt):
        precision = (
            self._torch.autocast(
                device_type="cuda",
                dtype=self._semantic_dtype,
            )
            if self._semantic_device.type == "cuda"
            else nullcontext()
        )
        with self._torch.inference_mode(), precision:
            output = self._semantic_model(
                **self._prepare_inputs(
                    audio,
                    prompt,
                    self._semantic_device,
                    self._semantic_dtype,
                ),
                logits_to_keep=1,
                return_dict=True,
                use_cache=False,
            )
        if not hasattr(output, "logits") or output.logits is None:
            raise RuntimeError("Music Flamingo semantic output does not contain logits")
        if not hasattr(output, "audio_hidden_states") or output.audio_hidden_states is None:
            raise RuntimeError(
                "Music Flamingo semantic output does not contain audio_hidden_states"
            )
        logits = output.logits
        if logits.ndim == 3:
            if logits.shape[1] != 1:
                raise RuntimeError("semantic logits must retain exactly one next-token position")
            logits = logits[:, 0, :]
        ordered_ids = [self._adjective_token_ids[word] for word in CANDIDATE_ADJECTIVES]
        if min(ordered_ids) < 0 or max(ordered_ids) >= logits.shape[-1]:
            raise RuntimeError("candidate adjective token ID falls outside the vocabulary")
        selected = (
            logits.index_select(
                -1, self._torch.tensor(ordered_ids, dtype=self._torch.long, device=logits.device)
            )
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        hidden = output.audio_hidden_states
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        if hidden.ndim != 3 or hidden.shape[1] == 0:
            raise RuntimeError("audio_hidden_states has an unexpected shape")
        pool = (
            self._torch.cat(
                (self._torch.mean(hidden, dim=1), self._torch.std(hidden, dim=1, unbiased=False)),
                dim=1,
            )
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        return selected, pool, int(hidden.shape[1])

    # Cache the zero-audio reference used to calibrate semantic logits.
    def _ensure_null_logits(self):
        if self._null_logits is not None:
            return
        zero_audio = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
        outputs = [self._semantic_forward(zero_audio, prompt) for prompt in PROMPT_TEMPLATES]
        self._null_logits = np.stack([selected[0] for selected, _pool, _count in outputs]).astype(
            np.float32
        )
        self._null_audio_token_counts = tuple(count for _selected, _pool, count in outputs)
        dimensions = {pool.shape[1] for _selected, pool, _count in outputs}
        if self._null_logits.shape != (2, 14) or len(dimensions) != 1:
            raise RuntimeError("null-audio semantic reference has an unexpected shape")
        self._null_audio_pool_dimension = int(next(iter(dimensions)))

    def _extract_semantic(self, audio):
        self._ensure_null_logits()
        outputs = [self._semantic_forward(audio, prompt) for prompt in PROMPT_TEMPLATES]
        counts = tuple(count for _selected, _pool, count in outputs)
        if counts != self._null_audio_token_counts:
            raise ValueError(
                "real and null audio-token counts must match for every semantic prompt"
            )
        pools = [pool for _selected, pool, _count in outputs]
        if any(pool.shape[1] != self._null_audio_pool_dimension for pool in pools):
            raise ValueError("real and null audio-pool dimensions must match")
        logits = np.stack([selected[0] for selected, _pool, _count in outputs])[None, :, :]
        return build_semantic_features(logits.astype(np.float32), self._null_logits)[0]

    def _select_heads(self, all_heads):
        selected = []
        for layer, head in self.selected_heads:
            if layer >= all_heads.shape[0] or head >= all_heads.shape[1]:
                raise ValueError(
                    "selected head coordinate falls outside captured projection inputs"
                )
            selected.append(all_heads[layer, head, :])
        return np.concatenate(selected, axis=0).astype(np.float32)

    # Use a 15-second look-back segment for every 0.5-second target point.
    def extract_times(self, audio_path, target_times, waveform=None):
        times = np.asarray(target_times, dtype=np.float32).reshape(-1)
        if waveform is None:
            waveform = decode_audio(
                audio_path,
                sampling_rate=AUDIO_SAMPLE_RATE,
            )
        self._load_models()
        self._ensure_null_logits()
        base_rows = []
        semantic_rows = []
        head_rows = []
        for target_time in times:
            segment = load_music_segment(
                audio_path,
                float(target_time) - SEGMENT_SECONDS,
                float(target_time),
            )
            if segment.shape != (SEGMENT_SAMPLES,):
                raise ValueError("every Music Flamingo input must contain exactly 240000 samples")
            base, all_heads = self._extract_base_and_heads(segment)
            base_rows.append(base)
            head_rows.append(self._select_heads(all_heads))
            semantic_rows.append(self._extract_semantic(segment))
        acoustic = extract_acoustic_features(
            waveform,
            AUDIO_SAMPLE_RATE,
            times,
            device=self._base_device,
        )
        return AudioFeatures(
            target_times=times,
            base=np.asarray(base_rows, dtype=np.float32),
            semantic=np.asarray(semantic_rows, dtype=np.float32),
            heads=np.asarray(head_rows, dtype=np.float32),
            acoustic=acoustic,
        )

    def extract_song(self, audio_path):
        waveform = decode_audio(audio_path, sampling_rate=AUDIO_SAMPLE_RATE)
        target_times = build_target_times(len(waveform) / float(AUDIO_SAMPLE_RATE))
        return self.extract_times(
            audio_path,
            target_times,
            waveform=waveform,
        )
