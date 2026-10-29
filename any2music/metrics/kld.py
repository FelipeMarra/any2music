import torch
import torch.nn.functional as F
import os
import contextlib
import typing as tp
from functools import partial
# ---------------------------------------------------------
# MONKEY PATCH: Fix for PyTorch 2.0+ compatibility
# The hear21passt library breaks in newer PyTorch versions 
# because it doesn't declare 'return_complex'. We intercept the function here.
# ---------------------------------------------------------
# _original_stft = torch.stft

# def _patched_stft(*args, **kwargs):
#     # Newer PyTorch requires return_complex=True
#     kwargs['return_complex'] = True
#     complex_tensor = _original_stft(*args, **kwargs)
#     # Converting to the old format (Real/Imag) that hear21passt expects
#     return torch.view_as_real(complex_tensor)

# # We replace the original function with our patched one
# torch.stft = _patched_stft
# # ---------------------------------------------------------



class _patch_passt_stft:
    """Decorator to patch torch.stft in PaSST."""
    def __init__(self):
        self.old_stft = torch.stft

    def __enter__(self):
        # return_complex is a mandatory parameter in latest torch versions
        # torch is throwing RuntimeErrors when not set
        torch.stft = partial(torch.stft, return_complex=False)

    def __exit__(self, *exc):
        torch.stft = self.old_stft

from hear21passt.base import get_basic_model
from any2music.base.metric import BaseAudioMetric

MODEL_SAMPLE_RATE = 32_000 # in Hz

class KLDMetric(BaseAudioMetric):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(expected_sample_rate=MODEL_SAMPLE_RATE, device=device)

        self._initialize_model()
        self.device = device
        self.reset()


    def _initialize_model(self):
        """Initialize underlying PaSST audio classifier and inner variables."""
        from hear21passt.base import get_basic_model  # type: ignore
        # original PASST was trained on AudioSet with 10s-long audio samples
        max_duration = 10 # in seconds
        min_duration = 0.15
        min_duration = 0.15
        # the PaSST model supports a maximum of 10 seconds at 32kHz (320,000 samples)
        max_input_frames = int(max_duration * MODEL_SAMPLE_RATE)
        min_input_frames = int(min_duration * MODEL_SAMPLE_RATE)
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            model = get_basic_model(mode='logits')
        
        self.max_input_frames = max_input_frames
        self.min_input_frames = min_input_frames
        self.model_sample_rate = MODEL_SAMPLE_RATE
        self.model = model
        self.model.eval()
        self.model.to(self.device)

    def reset(self):
        """Clears the accumulated distributions to start a new evaluation."""
        self.kld_scores = torch.tensor(0.).to(self.device)
        self.weight = torch.tensor(0.).to(self.device)

    def _process_audio(self, wav: torch.Tensor) -> tp.List[torch.Tensor]:
        if wav.size(-1) <= self.min_input_frames:
            return []
        # dont use padding for small segments
        # https://github.com/facebookresearch/audiocraft/blob/896ec7c47f5e5d1e5aa1e4b260c4405328bf009d/audiocraft/metrics/kld.py#L178
        segments = list(torch.split(wav, self.max_input_frames, dim=-1))
        if segments[-1].size(-1) <= self.min_input_frames:
            segments = segments[:-1]
        return segments

    def _get_probs(self, wav: torch.Tensor) -> torch.Tensor:
        """Passes the audio through the PaSST model and extracts probabilities, chunking if it's too long."""
        # assumes wav is a single audio
        if wav.dim() == 3 and wav.size(1) > 1:
            wav = wav.mean(dim=1) # in case theres more than one channel
        elif wav.dim() == 3 and wav.size(1) == 1:
            wav = wav.squeeze(1)
            
        all_probs = []
        # we don't pad but return a list of audio segments as this otherwise affects the KLD computation
        # we chunk the audio into blocks of maximum 10s
        segments = self._process_audio(wav)
        
        for i, seg in enumerate(segments):
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                with torch.no_grad(), _patch_passt_stft():
                    logits = self.model(seg.to(self.device))
                    probs = torch.softmax(logits, dim=-1)
                    print("Passt output shape: {}".format(probs.shape))
                    # probs.mean(dim=0)
                    all_probs.append(probs)
                
        if len(all_probs) == 0:
            return None

        # return a mean probability for the recording
        return torch.stack(all_probs, dim=0).mean(dim=0)

    def update(self, preds: torch.Tensor, targets: torch.Tensor, pred_sr: int, target_sr:int):
        """Resamples to 32kHz, extracts the probabilities and moves them to RAM."""
        preds_32k = self._resample_if_needed(preds, pred_sr)
        targets_32k = self._resample_if_needed(targets, target_sr)

        preds_probs = self._get_probs(preds_32k)
        targets_probs = self._get_probs(targets_32k)

        print(preds_probs.shape)
        print(targets_probs.shape)
        epsilon = 1e-6

        if preds_probs is not None and targets_probs is not None:
            # on audiocraft implementation they do this limitation, but this isnt
            # a real limitation to the KLD as the population p and q may have different sizes.
            # in other words, audiocraft assume the generated and the target has almost the same duration
            # assert preds_probs.shape == targets_probs.shape
            kld_score = F.kl_div((preds_probs + epsilon).log(), targets_probs, reduction="batchmean")
            assert not torch.isnan(kld_score), "kld_scores contains NaN value(s)!"
            self.kld_scores += kld_score
            self.weight += 1

            
    def compute(self) -> dict:
        """Takes the accumulated probabilities and calculates the KL Divergence."""
        weight: float = float(self.weight.item())
        if weight == 0:
            return {'kld': 0.0}

        kl_div = self.kld_scores.item() / weight  
        self.reset()
        
        return {'kld': kl_div}