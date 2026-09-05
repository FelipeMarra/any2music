import torch
import torchaudio
import warnings
from any2music.metrics.fad import FADMetric
from any2music.metrics.kld import KLDMetric
from any2music.metrics.clap_score import ClapScoreMetric

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def test_real_audio_pipeline():
    print("=== Evaluating Real Music with all 3 Metrics ===")
    
    # Using the 1943 audio from the samples folder
    source_wav = "./samples/audio/002_1943_TheBattleofMidway_00_01Title.wav"

    # Using torchaudio maintains consistency and avoids new dependencies
    wav_tensor_raw, sr = torchaudio.load(source_wav)

    # torchaudio returns [Channels, Time]. We add the Batch dimension: [1, Channels, Time]
    wav_tensor = wav_tensor_raw.unsqueeze(0)

    preds = wav_tensor
    targets = wav_tensor
    
    # From 002_1943_TheBattleofMidway_00_01Title_music_track_description
    texts = ["The music is fast-paced and energetic, with a strong emphasis on percussion and rhythmic elements. It's likely to be played in a high-energy environment, such as a sci-fi menu or a futuristic cityscape. The overall atmosphere is one of tension and excitement, with a sense of urgency and momentum driving the music forward. The genre is likely to be electronic or synth-heavy, with a focus on pulsing beats and futuristic sounds. In terms of gameplay context, the player may be navigating a fast-paced action sequence or exploring a futuristic world. The music is designed to keep the player engaged and energized, with a sense of constant movement and progression."]

    fad = FADMetric()
    kld = KLDMetric()
    clap = ClapScoreMetric()

    print(f"\nFeeding the metrics with the audio at {sr}Hz...")
    fad.update(preds, targets, current_sr=sr)
    kld.update(preds, targets, current_sr=sr)
    clap.update(preds, targets, current_sr=sr, texts=texts)

    print("\n======")
    print(f"FAD Score : {fad.compute()['fad_score']:.6f}")
    print(f"KLD Score : {kld.compute()['kld']:.6f}")
    print(f"CLAP Score: {clap.compute()['clap_score']:.6f}")

if __name__ == "__main__":
    test_real_audio_pipeline()