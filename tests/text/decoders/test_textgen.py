import json
import random

import torch

from any2music.text.encoders import T5Conditioner
from any2music.text.decoders.textgen import TextGenSize, TEXTGEN_SIZES, TextGenTransformer

TEXT_TOKENS_PATH = "./samples/midi/348_TheLegendofZelda_01_02OverworldBGM_midi_tokens.json"
DESC_PATH = "./samples/description/348_TheLegendofZelda_01_02OverworldBGM_music_track_description.md"

ORIG_VOCAB_SIZE = 406
BOS = ORIG_VOCAB_SIZE +1
PAD = ORIG_VOCAB_SIZE +2
EOS = ORIG_VOCAB_SIZE +3
VOCAB_SIZE = EOS + 1

UPDATES = 100

def get_text_tokens(text_tokens_path) -> tuple[torch.Tensor, torch.Tensor]:
    with open(text_tokens_path, 'r') as f:
        text_tokens = json.load(f)

    text_tokens = torch.Tensor(text_tokens).unsqueeze(0).int()

    input_tokens = text_tokens[:, :-1]
    target_tokens = text_tokens[:, 1:]

    return input_tokens, target_tokens


def test_textgen_t5():
    # Get tokenized text (e.g., MIDI, NLM)
    input_tokens, target_tokens = get_text_tokens(TEXT_TOKENS_PATH)
    
    # Get tokenized description
    dec_size = TEXTGEN_SIZES["test"]
    t5 = T5Conditioner('t5-base', dec_size.d_model, device='cuda').cuda()

    with open(DESC_PATH, 'r') as f:
        desc = f.read()

    desc_tokens = t5.tokenize([desc])

    with torch.no_grad():
        desc_embs, desc_embs_mask = t5(desc_tokens)

    del t5

    desc_embs_mask = desc_embs_mask.bool()

    model = TextGenTransformer(
        vocab_size=VOCAB_SIZE,
        pad_token_id=PAD,
        eos_token_id=EOS,
        bos_token_id=BOS,
        max_seq_len = 1500,
        model_size=TextGenSize.TEST
    ).cuda()

    # NOTICE: All hyperparams here are for test
    criterium = torch.nn.CrossEntropyLoss(ignore_index=model.pad_token_id) # Ignore the padding tokens in the loss calculation
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.0)

    model.train()
    for update in range(UPDATES):
        optim.zero_grad()

        desc_embs = desc_embs.to(torch.bfloat16)
        input_tokens = input_tokens.cuda()
        desc_embs_mask = desc_embs_mask.cuda()

        logits = model(src=desc_embs, tgt=input_tokens, src_mask=desc_embs_mask)

        # Compute loss and step
        loss = criterium(logits.permute(0, 2, 1), target_tokens.long().cuda())
        loss.backward()
        optim.step()

        print(f"Update {update} | Loss: {loss.item():.4f}")

    # Run generation
    model.eval()
    with torch.no_grad():
        print(f"Generating audio tokens for {DESC_PATH}...")
        text_tokens = model.generate(
            src=desc_embs,
            src_mask=desc_embs_mask,
            max_new_tokens=model.max_seq_len,
            temperature=1e-4, # < 1 -> eliminate randomness | = 1 -> the distribution learned | > 1 -> aproximate a uniform distribution
            top_k=1
        )

        print(f"Generated text codes shape: {text_tokens.shape}\n")
        print(f"Generated text head: {text_tokens[:, :5]}\n")
        print(f"Generated text tail: {text_tokens[:, -5:]}\n")

        text_tokens = text_tokens.detach().cpu()
        with open('test_textgen_t5.json', 'w') as f:
            json.dump(text_tokens.tolist(), f)
