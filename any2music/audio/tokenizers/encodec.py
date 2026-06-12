import math
import typing as tp

import numpy as np
import torch
from transformers import EncodecModel as HFEncodecModel

from any2music.base import BaseAudioTokenizer

from .. import quantization as qt

# from https://github.com/facebookresearch/audiocraft/blob/896ec7c47f5e5d1e5aa1e4b260c4405328bf009d/audiocraft/models/encodec.py#L323
class HFEncodecCompressionModel(BaseAudioTokenizer):
    """Wrapper around HuggingFace Encodec.
    """
    def __init__(self, model: HFEncodecModel):
        super().__init__()
        self.model = model

        bws = self.model.config.target_bandwidths
        num_codebooks = [
            bw * 1000 / (self.frame_rate * math.log2(self.cardinality))
            for bw in bws
        ]
        deltas = [nc - int(nc) for nc in num_codebooks]
        # Checking we didn't do some bad maths and we indeed have integers!
        assert all(d <= 1e-3 for d in deltas), deltas

        self.possible_num_codebooks = [int(nc) for nc in num_codebooks]
        self.set_num_codebooks(max(self.possible_num_codebooks))

    @staticmethod
    def get_pretrained(name: str, device: tp.Union[torch.device, str] = 'cuda') -> 'HFEncodecCompressionModel':
        model = HFEncodecModel.from_pretrained(name).to(device).eval()
        return HFEncodecCompressionModel(model)

    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        # We don't support training with this.
        raise NotImplementedError("Forward and training with HF EncodecModel not supported.")

    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        bandwidth_index = self.possible_num_codebooks.index(self.num_codebooks)
        bandwidth = self.model.config.target_bandwidths[bandwidth_index]

        res = self.model.encode(x, None, bandwidth)
        assert len(res[0]) == 1
        assert len(res[1]) == 1
        return res[0][0], res[1][0]

    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        if scale is None:
            scales = [None]  # type: ignore
        else:
            scales = scale  # type: ignore
        res = self.model.decode(codes[None], scales)
        return res[0]

    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        return self.model.quantizer.decode(codes.transpose(0, 1))

    @property
    def channels(self) -> int:
        return self.model.config.audio_channels

    @property
    def frame_rate(self) -> float:
        hop_length = int(np.prod(self.model.config.upsampling_ratios))
        return self.sample_rate / hop_length

    @property
    def sample_rate(self) -> int:
        return self.model.config.sampling_rate

    @property
    def cardinality(self) -> int:
        return self.model.config.codebook_size

    @property
    def num_codebooks(self) -> int:
        return self._num_codebooks

    @property
    def total_codebooks(self) -> int:
        return max(self.possible_num_codebooks)

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer.
        """
        if n not in self.possible_num_codebooks:
            raise ValueError(f"Allowed values for num codebooks: {self.possible_num_codebooks}")
        self._num_codebooks = n

    @property
    def vocab_size(self) -> int:
        return self.model.config.codebook_size