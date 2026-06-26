import torch
import torch.nn.functional as F
import os
import contextlib

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
        
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            self.model = get_basic_model(mode='logits')
        
        self.model.to(self.device)
        self.model.eval()
        self.reset()

    def reset(self):
        """Clears the accumulated distributions to start a new evaluation."""
        self.pred_probs_list = []
        self.target_probs_list = []

    def _get_probs(self, wav: torch.Tensor) -> torch.Tensor:
        """Passes the audio through the PaSST model and extracts probabilities, chunking if it's too long."""
        if wav.dim() == 3 and wav.size(1) > 1:
            wav = wav.mean(dim=1)
        elif wav.dim() == 3 and wav.size(1) == 1:
            wav = wav.squeeze(1)
            
        # The PaSST model supports a maximum of 10 seconds at 32kHz (320,000 samples)
        max_length = 320000
        probs_list = []
        
        # We chunk the audio into blocks of maximum 10s
        for start in range(0, wav.size(1), max_length):
            chunk = wav[:, start:start + max_length]
            
            with torch.no_grad():
                logits = self.model(chunk.to(self.device))
                probs = torch.softmax(logits, dim=-1)
                probs_list.append(probs)
                
        # We concatenate the probabilities of all generated blocks
        return torch.cat(probs_list, dim=0)

    def update(self, preds: torch.Tensor, targets: torch.Tensor, current_sr: int, texts: list = None):
        """Resamples to 32kHz, extracts the probabilities and moves them to RAM."""
        preds_32k = self._resample_if_needed(preds, current_sr)
        targets_32k = self._resample_if_needed(targets, current_sr)

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