import torch
import torch.nn.functional as F
import os
import contextlib
import typing as tp

# ---------------------------------------------------------
# MONKEY PATCH: Fix for PyTorch 2.0+ compatibility
# The hear21passt library breaks in newer PyTorch versions 
# because it doesn't declare 'return_complex'. We intercept the function here.
# ---------------------------------------------------------
_original_stft = torch.stft

def _patched_stft(*args, **kwargs):
    # Newer PyTorch requires return_complex=True
    kwargs['return_complex'] = True
    complex_tensor = _original_stft(*args, **kwargs)
    # Converting to the old format (Real/Imag) that hear21passt expects
    return torch.view_as_real(complex_tensor)

# We replace the original function with our patched one
torch.stft = _patched_stft
# ---------------------------------------------------------

from hear21passt.base import get_basic_model
from any2music.base.metric import BaseAudioMetric

class KLDMetric(BaseAudioMetric):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(expected_sample_rate=32000, device=device)

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
        model_sample_rate = 32_000 # in Hz
        max_input_frames = int(max_duration * model_sample_rate)
        min_input_frames = int(min_duration * model_sample_rate)
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            model = get_basic_model(mode='logits')
        
        self.min_input_frames = min_frames
        self.max_input_frames = max_frames
        self.model_sample_rate = sr
        self.model = model
        self.model.eval()
        self.model.to(self.device)

    def reset(self):
        """Clears the accumulated distributions to start a new evaluation."""
        self.pred_probs_list = []
        self.target_probs_list = []

    def _process_audio(self, wav: torch.Tensor) -> tp.List[torch.Tensor]:
        if wav.size(1) <= self.min_input_frames:
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
            
            with torch.no_grad():
                logits = self.model(seg.to(self.device))
                probs = torch.softmax(logits, dim=-1)
                print("Passt output shape: {}".format(probs.shape))
                # probs.mean(dim=0)
                all_probs.append(probs)
                
        if len(all_probs) == 0:
            return None

        # return torch.cat(probs_list, dim=0).mean(dim=0, keepdim=True)
        return torch.stack(all_probs, dim=0)

    def update(self, preds: torch.Tensor, targets: torch.Tensor, pred_sr: int, target_sr:int):
        """Resamples to 32kHz, extracts the probabilities and moves them to RAM."""
        preds_32k = self._resample_if_needed(preds, pred_sr)
        targets_32k = self._resample_if_needed(targets, target_sr)

        self.pred_probs_list.append(self._get_probs(preds_32k).cpu())
        self.target_probs_list.append(self._get_probs(targets_32k).cpu())

    def compute(self) -> dict:
        """Takes the accumulated probabilities and calculates the KL Divergence."""
        if not self.pred_probs_list:
            return {'kld': 0.0}
            
        all_preds = torch.cat(self.pred_probs_list, dim=0)
        all_targets = torch.cat(self.target_probs_list, dim=0)

        epsilon = 1e-6
        kl_div = F.kl_div((all_preds + epsilon).log(), all_targets, reduction="batchmean")
        
        self.reset()
        
        return {'kld': float(kl_div.item())}