import math
import torch
from audiotools import AudioSignal
from any2music.audio.tokenizers import DACCompressionModel

AUDIO_PATH = "./samples/audio/legend_of_zelda_nes.mp3"
TEST_SECs = 15

def test_dac_encode_decode():
    dac = DACCompressionModel.get_pretrained("24khz")
    dac.set_num_codebooks(4) # Use the same amount of codebooks used by MusicGen

    print(f"DAC info: \n\tfr={dac.frame_rate}\n\tsr={dac.sample_rate}\n\tcodebook_size={dac.cardinality}\n\ttotal_n_codebooks={dac.total_codebooks}\n\tn_codebooks={dac.num_codebooks}")

    audio_tensor = AudioSignal(AUDIO_PATH).to_mono()
    audio_tensor = audio_tensor[:, :, :dac.sample_rate*TEST_SECs].cuda() # subsample & cuda
    print(f"audio_tensor.shape: {audio_tensor.shape}")

    codes, meta = dac.encode(audio_tensor)
    print(f"codes.shape: {codes.shape}")
    print(f"codes meta: {meta}")
    assert codes.shape[0] == 1
    assert codes.shape[1] == 4
    assert math.isclose(codes.shape[-1], TEST_SECs*dac.frame_rate, rel_tol=1, abs_tol=1)

    decoded_audio = dac.decode(codes, meta).cpu()
    print(f"decoded_audio.shape: {decoded_audio.shape}")
    assert decoded_audio.shape == torch.Size([1, 1, dac.sample_rate*TEST_SECs])

    decoded_audio.write('test_dac.wav')