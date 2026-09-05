import os
import shutil
import torchaudio
from frechet_audio_distance import FrechetAudioDistance
from any2music.base.metric import BaseAudioMetric

class FADMetric(BaseAudioMetric):
    def __init__(self, device="cpu"):
        self.target_dir = "temp_fad_targets"
        self.pred_dir = "temp_fad_preds"
        
        super().__init__(expected_sample_rate=16000, device=device)
        
        self.frechet_oficial = FrechetAudioDistance(
            model_name="vggish", 
            sample_rate=16000, 
            use_pca=False, 
            use_activation=False,
            verbose=False
        )

    def reset(self):
        """Clears the folders to avoid mixing audios from one test with another."""
        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)
        if os.path.exists(self.pred_dir):
            shutil.rmtree(self.pred_dir)
            
        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)
        self.file_counter = 0

    def update(self, preds, targets, current_sr, texts=None):
        """Instead of complex math, it just saves the audios to disk."""

        preds_16k = self._resample_if_needed(preds, current_sr)
        targets_16k = self._resample_if_needed(targets, current_sr)

        for i in range(preds_16k.shape[0]):
            pred_path = os.path.join(self.pred_dir, f"audio_{self.file_counter}.wav")
            target_path = os.path.join(self.target_dir, f"audio_{self.file_counter}.wav")
            
            torchaudio.save(pred_path, preds_16k[i].cpu(), 16000)
            torchaudio.save(target_path, targets_16k[i].cpu(), 16000)
            
            self.file_counter += 1

    def compute(self) -> dict:
        """Calls the official library to read the folders in the simplest way possible."""
        if self.file_counter == 0:
            return {'fad_score': 0.0}

        score_final = self.frechet_oficial.score(self.target_dir, self.pred_dir)

        self.reset()

        return {'fad_score': float(score_final)}