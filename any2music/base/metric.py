from abc import ABC, abstractmethod
import torch
import torchaudio

class BaseAudioMetric(ABC):
    def __init__(self, expected_sample_rate: int, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        expected_sample_rate: The sample rate required by the evaluator model (e.g., 16000 for FAD).
        """
        self.expected_sample_rate = expected_sample_rate
        self.device = device
        self.reset()

    def _resample_if_needed(self, waveform: torch.Tensor, current_sr: int) -> torch.Tensor:
        """Automatically upsamples or downsamples the audio if the sample rate does not match the expected one."""
        if current_sr == self.expected_sample_rate:
            return waveform.to(self.device)
        
        # Creates the resampler dynamically
        resampler = torchaudio.transforms.Resample(
            orig_freq=current_sr, 
            new_freq=self.expected_sample_rate
        ).to(self.device)
        
        return resampler(waveform.to(self.device))

    @abstractmethod
    def reset(self):
        """Clears state variables to start a new validation."""
        pass

    @abstractmethod
    def update(self, preds: torch.Tensor, targets: torch.Tensor, current_sr: int, texts: list = None):
        """
        Receives a batch of generated (preds) and target (targets) audio samples.
        If the metric uses text (such as CLAP), it also receives a list of strings.
        """
        pass

    @abstractmethod
    def compute(self) -> float:
        """Runs the final math and returns the score."""
        pass