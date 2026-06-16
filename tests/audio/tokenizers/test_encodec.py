import torch

from torchaudio import save as save_audio

from any2music.audio.tokenizers import HFEncodecCompressionModel
from any2music.audio.utils import load_mono_and_resample

AUDIO_PATH = "./samples/audio/legend_of_zelda_nes.mp3"
TEST_SECs = 15

def test_encode_decode():
    encodec = HFEncodecCompressionModel.get_pretrained('facebook/encodec_32khz')

    audio_tensor = load_mono_and_resample(AUDIO_PATH, encodec.sample_rate)[0]
    audio_tensor = audio_tensor.unsqueeze(0) # add batch dim
    audio_tensor = audio_tensor[:, :, :encodec.sample_rate*TEST_SECs].cuda() # subsample & cuda
    print(audio_tensor.shape)

    codes, scale = encodec.encode(audio_tensor)
    print(codes.shape)
    assert codes.shape == torch.Size([1, 4, TEST_SECs*50])

    decoded_audio = encodec.decode(codes, scale)
    print(decoded_audio.shape)
    assert decoded_audio.shape == torch.Size([1, 1, encodec.sample_rate*TEST_SECs])

    save_audio("test_encodec_nes.wav", decoded_audio.squeeze().cpu(), sample_rate=encodec.sample_rate)
