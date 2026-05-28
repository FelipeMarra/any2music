import torch
from any2music.audio.decoders.musicgen import DelayProvider, MusicGenTransformer, MusicGenSize

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

def test_musicgen_inference():
    model = MusicGenTransformer(model_size=MusicGenSize.TEST).cuda()

    # Setup dummy conditioning (src). 
    # If you are doing unconditional generation, you can pass src=None.
    # Assuming your encoder outputs shape (Batch, SeqLen, d_model)
    batch_size = 1
    src_seq_len = 10
    d_model = model.size_params.d_model
    dummy_src = torch.randn(batch_size, src_seq_len, d_model, dtype=model.dtype).cuda()

    # Run generation
    # max_new_tokens = 250 -> roughly 5 seconds of audio at 50 Hz frame rate
    print("Generating audio tokens...")
    audio_tokens = model.generate(
        model=model,
        src=dummy_src,
        max_new_tokens=250, 
        temperature=1.0,
        top_k=250
    )

    print(f"Generated aligned audio tokens shape: {audio_tokens.shape}")

    # we lose (K-1 = 3) frames during the realignment
    assert audio_tokens.shape == torch.Size([1, 4, 247])