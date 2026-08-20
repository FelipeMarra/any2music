import random
import torch
import typing as tp

from torchaudio import save as save_audio

from audiotools import AudioSignal
from any2music.audio.tokenizers import HFEncodecCompressionModel
from any2music.audio.tokenizers import DACCompressionModel
from any2music.audio.utils import load_mono_and_resample
from any2music.text.encoders import T5Conditioner
from any2music.audio.decoders.musicgen import DelayProvider, MusicGenTransformer, MusicGenSize, MUSICGEN_SIZES

AUDIO_PATH = "./samples/audio/legend_of_zelda_nes.mp3"
AUDIO_PATH_SNES = "./samples/audio/legend_of_zelda_snes.mp3"
AUDIO_PATH_JAMENDO = "/home/es119256/dados/repos/any2music/samples/audio/mtg_jamendo_220.wav"
TEST_SECs = 10
UPDATES = 100

def test_delay_pattern():
    input_tensor = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0]
    ])
    input_tensor = input_tensor.unsqueeze(0)
    print(f"input_tensor shape:\n{input_tensor.shape}\n")

    hf_input_tensor, hf_delay_pattern_mask = DelayProvider.build_delay_pattern_mask(input_tensor, 0, 1504)

    print(f"hf_input_tensor:\n{hf_input_tensor}\nshape: {hf_input_tensor.shape}\n")
    print(f"hf_delay_pattern_mask:\n{hf_delay_pattern_mask}\nshape: {hf_delay_pattern_mask.shape}\n")

    applyied_delay = DelayProvider.apply_delay_pattern_mask(hf_input_tensor, hf_delay_pattern_mask)
    print(f"hf_applyied_mask:\n{applyied_delay}\nshape:{applyied_delay.shape}\n")

    reverted_delay = DelayProvider.revert_delay_pattern(applyied_delay)
    print(f"reverted_delay:\n{reverted_delay}\nshape:{reverted_delay.shape}\n")

    assert torch.equal(reverted_delay, input_tensor[:, :, :reverted_delay.shape[-1]])


def tokenize_audio(wav:tp.Union[str, torch.Tensor], audio_secs:int, audio_tokenizer):
    sr = audio_tokenizer.sample_rate
    max_audio_len = sr*audio_secs
    max_audio_tokens = int(audio_tokenizer.frame_rate*audio_secs)

    if isinstance(wav, str):
        audio_tensor = AudioSignal(wav, device='cuda').resample(sr).to_mono()[:, :, :max_audio_len].cuda()
    else:
        audio_tensor = wav

    # Tokenize the audio
    encoded_audio, meta = audio_tokenizer.encode(audio_tensor)
    encoded_audio = encoded_audio[:, :, :max_audio_tokens]

    print(f"encoded_audio.shape after max_audio_tokens: {encoded_audio.shape}\nmeta: {meta}")

    # Add special tokens
    B, K, S = encoded_audio.shape

    bos = torch.full((B, K, 1), audio_tokenizer.bos_token_id, dtype=torch.long, device='cuda')
    eos = torch.full((B, K, 1), audio_tokenizer.eos_token_id, dtype=torch.long, device='cuda')
    padding = torch.full((B, K, 1), audio_tokenizer.pad_token_id, dtype=torch.long, device='cuda')
    print(f"padding: {audio_tokenizer.pad_token_id} |  eos: {audio_tokenizer.eos_token_id} | bos: {audio_tokenizer.bos_token_id} \n")

    min_padding = audio_tokenizer.num_codebooks-1 #  num_codebooks -1 for the delay pattern
    delta = max_audio_tokens - S
    padding_size = delta + min_padding # If delta==0 we get min_padding. If deta is a positive num, we get the value to get up to max_audio_tokens + min_padding
    padding_size = max(min_padding, delta + min_padding)

    final_padding = padding.expand((-1, -1, padding_size)) 
    encoded_audio = torch.cat([bos, encoded_audio, eos, final_padding], dim=-1)

    print(f"encoded_audio.shape after special tokens: {encoded_audio.shape}\n")

    assert encoded_audio.shape[-1] == max_audio_tokens + min_padding + 2

    # Apply the delay pattern for MusicGen
    # This shifts codebook 1 by 0, codebook 2 by 1, codebook 3 by 2, etc.
    delayed_audio, _ = DelayProvider.build_delay_pattern_mask(
        input_ids=encoded_audio,
        pad_token_id=audio_tokenizer.pad_token_id,
        max_length=encoded_audio.shape[-1] + audio_tokenizer.num_codebooks,
        audio_channels=1
    )

    print(f"delayed_audio shape: {delayed_audio.shape}\n")
    print(f"delayed_audio: {delayed_audio[0, :, :4]}\n")

    input_tokens = delayed_audio[:, :, :-1]
    target_tokens = delayed_audio[:, :, 1:]

    return input_tokens, target_tokens, meta


def test_musicgen_encodec():
    encodec = HFEncodecCompressionModel.get_pretrained('facebook/encodec_32khz').cuda()
    model = MusicGenTransformer(
            vocab_size=encodec.vocab_size,
            pad_token_id=encodec.pad_token_id,
            eos_token_id=encodec.eos_token_id,
            bos_token_id=encodec.bos_token_id,
            frame_rate=int(encodec.frame_rate),
            audio_duration=TEST_SECs,
            model_size=MusicGenSize.TEST
        ).cuda()

    audio_tensor = load_mono_and_resample(AUDIO_PATH, encodec.sample_rate)[0]
    audio_tensor = audio_tensor.unsqueeze(0).cuda() # add batch dim

    # Tokenize the audio
    input_tokens, target_tokens, scale = tokenize_audio(audio_tensor, TEST_SECs, encodec)

    # NOTICE: All hyperparams here are for test
    criterium = torch.nn.CrossEntropyLoss(ignore_index=model.pad_token_id) # Ignore the padding tokens in the loss calculation
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.0)

    # NOTICE: Cosine scheduler is used in musicgen but we wont use it for testing
    # scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100, eta_min=0.01)

    model.train()
    for update in range(1, UPDATES):
        optim.zero_grad()

        logits = model(src=None, tgt=input_tokens)

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
        max_new_tokens=model.max_seq_len,
        temperature=1e-4, # < 1 -> eliminate randomness | = 1 -> the distribution learned | > 1 -> aproximate a uniform distribution
        top_k=1
    )

    # Decode back to audio
    with torch.no_grad():
        decoded_audio = encodec.decode(audio_tokens, scale)

    print(f"Decoded audio shape: {decoded_audio.shape}")
    save_audio("test_musicgen_encodec.wav", decoded_audio.squeeze(0).cpu(), sample_rate=encodec.sample_rate)

    # TODO: KLD between the first 15s of the original song and the generated 15s -> should be a veeery small value


def test_musicgen_dac():
    dac = DACCompressionModel.get_pretrained("44khz")
    dac.set_num_codebooks(4)
    model = MusicGenTransformer(
        vocab_size=dac.vocab_size,
        pad_token_id=dac.pad_token_id,
        eos_token_id=dac.eos_token_id,
        bos_token_id=dac.bos_token_id,
        frame_rate=int(dac.frame_rate),
        audio_duration=TEST_SECs,
        model_size=MusicGenSize.TEST
    ).cuda()

    # Tokenize the audio
    input_tokens, target_tokens, meta = tokenize_audio(AUDIO_PATH, TEST_SECs, dac)

    # NOTICE: All hyperparams here are for test
    criterium = torch.nn.CrossEntropyLoss(ignore_index=model.pad_token_id) # Ignore the padding tokens in the loss calculation
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.0)

    # NOTICE: Cosine scheduler is used in musicgen but we wont use it for testing
    # scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100, eta_min=0.01)

    model.train()
    for update in range(1, UPDATES):
        optim.zero_grad()

        logits = model(src=None, tgt=input_tokens)

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
    with torch.no_grad():
        audio_tokens = model.generate(
            src=None,
            max_new_tokens=model.max_seq_len,
            temperature=1e-4, # < 1 -> eliminate randomness | = 1 -> the distribution learned | > 1 -> aproximate a uniform distribution
            top_k=1
        )

    print(f"Generate audio codes shape: {audio_tokens.shape}\n")

    # Decode back to audio
    with torch.no_grad():
        decoded_audio = dac.decode(audio_tokens.cpu(), meta)

    print(f"Decoded audio shape: {decoded_audio.shape}")
    total_generated_samples = audio_tokens.shape[-1] * dac.model.hop_length
    meta['original_length'] = min(meta['original_length'], total_generated_samples)
    decoded_audio.write('test_musicgen_dac.wav')

    # TODO: KLD between the first 15s of the original song and the generated 15s -> should be a veeery small value


def test_musicgen_t5_dac():
    # T5
    dec_size = MUSICGEN_SIZES["test"]
    t5 = T5Conditioner('t5-base', dec_size.d_model, device='cuda').cuda()

    nes_text = "NES"
    snes_text = "SUPER"
    jamendo_text = "JAMENDO"

    nes_t5_input = t5.tokenize([nes_text])
    snes_t5_input = t5.tokenize([snes_text])
    jamendo_t5_input = t5.tokenize([jamendo_text])

    with torch.no_grad():
        nes_t5_embeds, _ = t5(nes_t5_input)
        snes_t5_embeds, _ = t5(snes_t5_input)
        jamendo_t5_embeds, _ = t5(jamendo_t5_input)

    conditioners_txt = [nes_text, snes_text, jamendo_text]
    conditioners = [nes_t5_embeds.to(torch.bfloat16), snes_t5_embeds.to(torch.bfloat16), jamendo_t5_embeds.to(torch.bfloat16)]

    del t5

    # DAC
    dac = DACCompressionModel.get_pretrained("44khz")
    dac.set_num_codebooks(4)
    model = MusicGenTransformer(
        vocab_size=dac.vocab_size,
        pad_token_id=dac.pad_token_id,
        eos_token_id=dac.eos_token_id,
        bos_token_id=dac.bos_token_id,
        frame_rate=int(dac.frame_rate),
        audio_duration=TEST_SECs,
        model_size=MusicGenSize.TEST
    ).cuda()

    # Tokenize the audio
    nes_input_tokens, nes_target_tokens, nes_meta = tokenize_audio(AUDIO_PATH, TEST_SECs, dac) # type: ignore
    snes_input_tokens, snes_target_tokens, snes_meta = tokenize_audio(AUDIO_PATH_SNES, TEST_SECs, dac) # type: ignore
    jamendo_input_tokens, jamendo_target_tokens, jamendo_meta = tokenize_audio(AUDIO_PATH_JAMENDO, TEST_SECs, dac) # type: ignore

    metas = [nes_meta, snes_meta, jamendo_meta]
    model_inputs = [nes_input_tokens, snes_input_tokens, jamendo_input_tokens]
    target_tokens = [nes_target_tokens, snes_target_tokens, jamendo_target_tokens]

    # NOTICE: All hyperparams here are for test
    criterium = torch.nn.CrossEntropyLoss(ignore_index=model.pad_token_id) # Ignore the padding tokens in the loss calculation
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.0)

    # NOTICE: Cosine scheduler is used in musicgen but we wont use it for testing
    # scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100, eta_min=0.01)

    training_data = list(zip(conditioners, model_inputs, target_tokens))

    model.train()
    for update in range(1, UPDATES):
        for _ in range(len(training_data)):
            src, model_input, target = random.choice(training_data)

            optim.zero_grad()
            logits = model(src=src, tgt=model_input)

            # Reshape for CrossEntropyLoss
            flat_logits = logits.reshape(-1, model.vocab_size)
            flat_targets = target.reshape(-1)

            # Compute loss and step
            loss = criterium(flat_logits, flat_targets)
            loss.backward()
            optim.step()

            print(f"Update {update} | Loss: {loss.item():.4f}")

    # Run generation
    model.eval()
    with torch.no_grad():
        for name, src, meta in zip(conditioners_txt, conditioners, metas):
            print(f"Generating audio tokens for {name}...")
            audio_tokens = model.generate(
                src=src,
                max_new_tokens=model.max_seq_len,
                temperature=1e-4, # < 1 -> eliminate randomness | = 1 -> the distribution learned | > 1 -> aproximate a uniform distribution
                top_k=1
            )

            print(f"Generate audio codes shape: {audio_tokens.shape}\n")
            print(f"Generate audio final codes: {audio_tokens[:, :, -4:]}\n")

            # Decode back to audio
            with torch.no_grad():
                audio_tokens = torch.clamp(audio_tokens, min=0, max=dac.orig_vocab_size - 1)
                total_generated_samples = audio_tokens.shape[-1] * dac.model.hop_length
                meta['original_length'] = min(meta['original_length'], total_generated_samples)
                decoded_audio = dac.decode(audio_tokens.cpu(), meta)

            print(f"Decoded audio shape: {decoded_audio.shape}")
            decoded_audio.write(f'test_musicgen_t5_dac_{name}.wav')

            # TODO: KLD between the first 15s of the original song and the generated 15s -> should be a veeery small value