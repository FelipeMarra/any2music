import torch
from torchaudio import save as save_audio

from any2music.audio.tokenizers import HFEncodecCompressionModel
from any2music.audio.utils import load_mono_and_resample

from any2music.audio.decoders.musicgen import DelayProvider, MusicGenTransformer, MusicGenSize

AUDIO_PATH = "./audio_sample/legend_of_zelda.mp3"
TEST_SECs = 15
UPDATES = 500
SAVE_GEN = False

def test_delay_pattern():
    input_tensor = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0,],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0,],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0,],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0,]
    ])
    input_tensor = input_tensor.unsqueeze(0)
    print(f"input_tensor shape:\n{input_tensor.shape}\n")

    # if the padding token is 0
    delayed_input_tensor = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0],
        [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0],
        [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0],
        [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0]
    ])
    print(f"delayed_input_tensor shape:\n{delayed_input_tensor.shape}\n")
    delayed_input_tensor = delayed_input_tensor.unsqueeze(0)

    hf_input_tensor, hf_delay_pattern_mask = DelayProvider.build_delay_pattern_mask(input_tensor, 0, 1504)

    print(f"hf_input_tensor:\n{hf_input_tensor}\nshape: {hf_input_tensor.shape}\n")
    print(f"hf_delay_pattern_mask:\n{hf_delay_pattern_mask}\nshape: {hf_delay_pattern_mask.shape}\n")

    applyied_delay = DelayProvider.apply_delay_pattern_mask(hf_input_tensor, hf_delay_pattern_mask)
    print(f"hf_applyied_mask:\n{applyied_delay}\nshape:{applyied_delay.shape}\n")

    reverted_delay = DelayProvider.revert_delay_pattern(applyied_delay)
    print(f"reverted_delay:\n{reverted_delay}\nshape:{reverted_delay.shape}\n")

    assert torch.equal(reverted_delay, input_tensor[:, :, :reverted_delay.shape[-1]])

def test_musicgen_training():
    encodec = HFEncodecCompressionModel.get_pretrained('facebook/encodec_32khz').cuda()
    model = MusicGenTransformer(model_size=MusicGenSize.TEST).cuda()

    audio_tensor = load_mono_and_resample(AUDIO_PATH, encodec.sample_rate)[0]
    audio_tensor = audio_tensor.unsqueeze(0) # add batch dim
    audio_tensor = audio_tensor[:, :, :encodec.sample_rate*TEST_SECs].cuda()

    # Tokenize the audio
    with torch.no_grad():
        encoded_audio, _ = encodec.encode(audio_tensor)
        # encoded_audio shape: (Batch, Codebooks, SeqLen)
        print(f"encoded_audio.shape: {encoded_audio.shape}\n")

    # Apply the delay pattern for MusicGen
    # This shifts codebook 1 by 0, codebook 2 by 1, codebook 3 by 2, etc.
    delayed_audio, _ = DelayProvider.build_delay_pattern_mask(
        input_ids=encoded_audio,
        pad_token_id=model.pad_token_id,
        max_length=encoded_audio.shape[-1] + model.num_codebooks, # Add room for the shifts
        audio_channels=1
    )

    # Create Inputs and Labels (shifted by 1)
    # Input is everything except the very last timestep
    # Label is everything except the very first timestep
    model_input = delayed_audio[:, :, :-1]
    target_tokens = delayed_audio[:, :, 1:]

    # NOTICE: All hyperparams here are for test
    criterium = torch.nn.CrossEntropyLoss(ignore_index=model.pad_token_id) # Ignore the padding tokens in the loss calculation
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0)
    # NOTICE: Cosine scheduler is used in musicgen but we wont use it for testing
    # scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100, eta_min=0.01)

    model.train()
    for update in range(1, UPDATES):
        optim.zero_grad()

        logits = model(src=None, tgt=model_input)

        # Reshape for CrossEntropyLoss
        # Logits: (B, K, S, Vocab) -> (B * K * S, Vocab)
        # Targets: (B, K, S) -> (B * K * S)
        flat_logits = logits.reshape(-1, model.vocab_size)
        flat_targets = target_tokens.reshape(-1)

        # Compute loss and step
        loss = criterium(flat_logits, flat_targets)
        loss.backward()
        optim.step()

        print(f"Update {update} | Loss: {loss.item():.4f}")

    # Run generation
    model.eval()
    print("Generating audio tokens...")
    audio_tokens = model.generate(
        src=None,
        max_new_tokens=int(TEST_SECs * encodec.frame_rate), 
        temperature=1.0,
        top_k=250
    )

    # Decode back to audio
    with torch.no_grad():
        decoded_audio = encodec.decode(audio_tokens)

    print(f"Decoded audio shape: {decoded_audio.shape}")
    save_audio("test.wav", decoded_audio.squeeze(0).cpu(), sample_rate=encodec.sample_rate)