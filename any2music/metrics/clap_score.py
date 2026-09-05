import torch
import torch.nn.functional as F
import laion_clap
from any2music.base.metric import BaseAudioMetric

class ClapScoreMetric(BaseAudioMetric):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(expected_sample_rate=48000, device=device)
        
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
        self.model.load_ckpt()
        self.model.to(self.device)
        self.model.eval()
        
        self.reset()

    def reset(self):
        """Clears the accumulated lists to start a new evaluation."""
        self.accumulated_audios = []
        self.accumulated_texts = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor, current_sr: int, texts: list = None):
        """
        Receives the batch of generated audios and stores the tensors in RAM for computation.
        """
        if texts is None:
            raise ValueError("CLAP Score requires a list of texts (prompts) for evaluation.")
            
        preds_48k = self._resample_if_needed(preds, current_sr)
        
        if preds_48k.dim() == 3 and preds_48k.size(1) > 1:
            preds_48k = preds_48k.mean(dim=1)
        elif preds_48k.dim() == 3 and preds_48k.size(1) == 1:
            preds_48k = preds_48k.squeeze(1)
            
        for i in range(preds_48k.size(0)):
            self.accumulated_audios.append(preds_48k[i].detach().cpu())
            self.accumulated_texts.append(texts[i])

    def compute(self) -> dict:
        """Processes everything in batch using the official LAION engine directly from memory."""
        if not self.accumulated_audios:
            return {'clap_score': 0.0}
            
        total_similarity = 0.0
        num_samples = len(self.accumulated_audios)
        
        for audio, text in zip(self.accumulated_audios, self.accumulated_texts):
            audio_input = audio.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                audio_embed = self.model.get_audio_embedding_from_data(x=audio_input, use_tensor=True)
                text_embed = self.model.get_text_embedding([text], use_tensor=True)
                
                similarity = F.cosine_similarity(audio_embed, text_embed, dim=-1)
                total_similarity += similarity.item()
        
        final_score = total_similarity / num_samples
        
        self.reset()
        
        return {'clap_score': float(final_score)}