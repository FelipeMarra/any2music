import typing as tp

import torch
import dac
from dac import DACFile
from audiotools import AudioSignal

from any2music.base import BaseAudioTokenizer

class DACCompressionModel(BaseAudioTokenizer):
    """Wrapper around DAC (https://github.com/descriptinc/descript-audio-codec)."""
    def __init__(self, model: dac.DAC):
        super().__init__()
        self.model = model
        self.possible_num_codebooks = range(1,10)

    @staticmethod
    def get_pretrained(
            name: str="44khz",
            device: tp.Union[torch.device, str] = 'cuda'
        ) -> 'DACCompressionModel':

        model_path = dac.utils.download(model_type=name)
        model = dac.DAC.load(str(model_path.absolute())).to(device).eval()
        return DACCompressionModel(model)

    def forward(self, x: torch.Tensor):
        # We don't support training with this.
        raise NotImplementedError("Forward and training with HF DAC not supported.")

    def encode(self, audio_signal: AudioSignal, n_quantizers: tp.Optional[int] = None) -> tp.Tuple[torch.Tensor, dict]:
        if not n_quantizers:
            n_quantizers = self.model.n_codebooks

        if audio_signal.num_channels > 1:
            audio_signal = audio_signal.to_mono()

        dac_file = self.model.compress(audio_signal, win_duration=audio_signal.duration, n_quantizers=n_quantizers)
        meta = {
            "chunk_length":dac_file.chunk_length,
            "original_length":dac_file.original_length,
            "input_db":dac_file.input_db,
            "channels":dac_file.channels,
            "sample_rate":dac_file.sample_rate,
            "padding":dac_file.padding,
            "dac_version":dac_file.dac_version
        }
        return dac_file.codes, meta

    def decode(self, codes: torch.Tensor, meta:dict) -> AudioSignal:
        return self.model.decompress(DACFile(codes, **meta))

    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        return self.model.decode(codes)['z']

    @property
    def channels(self) -> int:
        return 1

    @property
    def frame_rate(self) -> float:
        return self.model.sample_rate / int(self.model.hop_length)

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    @property
    def cardinality(self) -> int:
        return self.model.codebook_size

    @property
    def num_codebooks(self) -> int:
        return self.model.n_codebooks

    @property
    def total_codebooks(self) -> int:
        return max(self.possible_num_codebooks)
    
    @property
    def vocab_size(self) -> int:
        return self.model.codebook_size

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer.
        """
        if n not in self.possible_num_codebooks:
            raise ValueError(f"Allowed values for num codebooks: {self.possible_num_codebooks}")
        self.model.n_codebooks = n