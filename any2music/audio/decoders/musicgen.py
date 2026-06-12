import math
import typing as tp
from enum import Enum
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from any2music.base import BaseDecoder

# To unsderstand how they model it on hf go to
# from transformers import MusicgenForConditionalGeneration
# Which uses 
# from transformers import MusicgenForCausalLM
# As the decoder
# Which uses
# from transformers import MusicgenModel
# As its base model
# Which uses MusicgenDecoder as its decoder
# Which is a ModuleList of MusicgenDecoderLayer

#########################################################
# Model Size Hyperparameters
#########################################################

class MusicGenSize(Enum):
    TEST = "test"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

@dataclass
class MusicGenSizeValues():
    d_model: int
    nhead: int
    num_decoder_layers: int

MUSICGEN_SIZES:tp.Dict[str, MusicGenSizeValues] = {
    "test": MusicGenSizeValues(d_model=1024, nhead=16, num_decoder_layers=12), # 3213MiB
    "small": MusicGenSizeValues(d_model=1024, nhead=16, num_decoder_layers=24)
}

#########################################################
# Transformer Layer Components
#########################################################

# Class obtained from https://github.com/huggingface/transformers/blob/10555512868d663ee1ff627e4f5c5c260114235b/src/transformers/models/musicgen/modeling_musicgen.py#L106
class MusicgenSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, dtype = torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.embedding_dim = embedding_dim
        self.num_positions = num_positions
        self.make_weights(num_positions, embedding_dim)

    def make_weights(self, num_embeddings: int, embedding_dim: int):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype and device of the param
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device) # type: ignore

        self.register_buffer("weights", emb_weights, persistent=False)

    def get_embedding(self, num_embeddings: int, embedding_dim: int):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb.to(self.dtype)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        _, _, seq_len = input_ids.size() # expects batch, codebooks, seq_len
        # Create the position ids from the input token ids.
        position_ids = (torch.arange(seq_len) + past_key_values_length).to(input_ids.device)
        # expand embeddings if needed
        if seq_len > self.weights.size(0): # type: ignore
            self.make_weights(seq_len, self.embedding_dim)
        return self.weights.index_select(0, position_ids.view(-1)).detach() # type: ignore

# Methods obtained from https://github.com/huggingface/transformers/blob/70257e9a3c2bfc7f7dda308836adc4ad561610b7/src/transformers/models/musicgen/modeling_musicgen.py#L813
class DelayProvider():
    @staticmethod
    def build_delay_pattern_mask(input_ids:torch.Tensor, pad_token_id:int, max_length:int, audio_channels:int=1):
        """
        Build a delayed pattern mask to the input_ids. Each codebook is offset by the previous codebook by
        one, giving a delayed pattern mask at the start of sequence and end of sequence. Take the example where there
        are 4 codebooks and a max sequence length of 8, we have the delayed pattern mask of shape `(codebooks,
        seq_len)`:
        - [P, -1, -1, -1, -1, P, P, P]
        - [P, P, -1, -1, -1, -1, P, P]
        - [P, P, P, -1, -1, -1, -1, P]
        - [P, P, P, P, -1, -1, -1, -1]
        where P is the special padding token id and -1 indicates that the token is valid for prediction. If we include
        a prompt (decoder input ids), the -1 positions indicate where new tokens should be predicted. Otherwise, the
        mask is set to the value in the prompt:
        - [P, a, b, -1, -1, P, P, P]
        - [P, P, c, d, -1, -1, P, P]
        - [P, P, P, e, f, -1, -1, P]
        - [P, P, P, P, g, h, -1, -1]
        where a-h indicate the input prompt (decoder input ids) that are offset by 1. Now, we only override the -1
        tokens in our prediction.
        """
        # (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
        #input_ids = input_ids.reshape(-1, num_codebooks, input_ids.shape[-1])
        B, K, S = input_ids.shape # batch, n_codebooks, seq_len

        input_ids_shifted = (
            torch.ones((B, K, max_length), dtype=torch.long, device=input_ids.device) * -1
        )

        channel_codebooks = K // 2 if audio_channels == 2 else K
        # we only apply the mask if we have a large enough seq len - otherwise we return as is
        if max_length < 2 * channel_codebooks - 1:
            # return input_ids.reshape(bsz * num_codebooks, -1), input_ids_shifted.reshape(bsz * num_codebooks, -1)
            return input_ids, input_ids_shifted

        # fill the shifted ids with the prompt entries, offset by the codebook idx
        for codebook in range(channel_codebooks):
            if audio_channels == 1:
                # mono channel - loop over the codebooks one-by-one
                input_ids_shifted[:, codebook, codebook : S + codebook] = input_ids[:, codebook]
            else:
                # left/right channels are interleaved in the generated codebooks, so handle one then the other
                input_ids_shifted[:, 2 * codebook, codebook : S + codebook] = input_ids[:, 2 * codebook]
                input_ids_shifted[:, 2 * codebook + 1, codebook : S + codebook] = input_ids[:, 2 * codebook + 1]

        # construct a pattern mask that indicates the positions of padding tokens for each codebook
        # first fill the upper triangular part (the EOS padding)
        delay_pattern = torch.triu(
            torch.ones((channel_codebooks, max_length), dtype=torch.bool), diagonal=max_length - channel_codebooks + 1
        )
        # then fill the lower triangular part (the BOS padding)
        delay_pattern = delay_pattern + torch.tril(torch.ones((channel_codebooks, max_length), dtype=torch.bool))

        if audio_channels == 2:
            # for left/right channel we need to duplicate every row of the pattern mask in an interleaved fashion
            delay_pattern = delay_pattern.repeat_interleave(2, dim=0)

        mask = ~delay_pattern.to(input_ids.device)
        input_ids = mask * input_ids_shifted + ~mask * pad_token_id

        # find the first position to start generating - this is the first place we have the -1 token
        # and will always be in the first codebook (since it has no codebook offset)
        first_codebook_ids = input_ids[:, 0, :]
        start_ids = (first_codebook_ids == -1).nonzero()[:, 1]
        if len(start_ids) > 0:
            first_start_id = min(start_ids)
        else:
            # we have no tokens that need to be filled - return entire matrix of input ids
            first_start_id = S

        # (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
        # pattern_mask = input_ids.reshape(bsz * num_codebooks, -1)
        # input_ids = input_ids[..., :first_start_id].reshape(bsz * num_codebooks, -1)
        pattern_mask = input_ids
        input_ids = input_ids[..., :first_start_id]
        return input_ids, pattern_mask

    @staticmethod
    def apply_delay_pattern_mask(input_ids:torch.Tensor, decoder_pad_token_mask:torch.Tensor):
        """
        Apply a delay pattern mask to the decoder input ids, only preserving predictions where
        the mask is set to -1, and otherwise setting to the value detailed in the mask.
        """
        seq_len = input_ids.shape[-1]
        decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]
        input_ids = torch.where(decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask)
        return input_ids

    @staticmethod
    def revert_delay_pattern(generated_ids: torch.Tensor):
        """
        Realigns the staggered codebooks back into synchronized audio frames.
        generated_ids shape: (Batch, Codebooks, SequenceLength)
        """
        B, K, S = generated_ids.shape

        # The valid sequence length is reduced by the maximum delay (K - 1)
        valid_length = S - (K - 1)

        if valid_length <= 0:
            raise ValueError("Generated sequence is too short to be aligned.")

        aligned_ids = torch.zeros((B, K, valid_length), dtype=generated_ids.dtype, device=generated_ids.device)

        for codebook_idx in range(K):
            # Shift each codebook back by its respective delay offset
            start_idx = codebook_idx
            end_idx = start_idx + valid_length
            aligned_ids[:, codebook_idx, :] = generated_ids[:, codebook_idx, start_idx:end_idx]

        return aligned_ids

#########################################################
# Transformer Layers
#########################################################

def get_musicgen_decoder(
        model_size:MusicGenSize=MusicGenSize.SMALL,
        dtype=torch.bfloat16
    ) -> nn.TransformerDecoderLayer:

    size_params = MUSICGEN_SIZES[model_size.value]

    return nn.TransformerDecoderLayer(
        d_model=size_params.d_model,
        nhead=size_params.nhead,
        activation=torch.nn.GELU(),
        dim_feedforward=size_params.d_model * 4,
        dropout=0.1,
        norm_first=True,
        batch_first=True,
        dtype=dtype
    )

#########################################################
# MusicGen Transformer
#########################################################

class MusicGenTransformer(BaseDecoder):
    def __init__(
            self, 
            vocab_size:int,
            frame_rate:int,
            audio_duration:int,
            encoder:tp.Optional[nn.TransformerEncoder] = None, 
            model_size:MusicGenSize=MusicGenSize.SMALL, 
            dtype:torch.dtype=torch.bfloat16
        ):
        super().__init__()
        self.size_params = MUSICGEN_SIZES[model_size.value]
        self.vocab_size = vocab_size + 1 # must match the codebook size used in the neural encoder + 1 for padding
        self.pad_token_id: int = vocab_size # starts counting from 0
        self.num_codebooks = 4
        self.max_seq_len = frame_rate*audio_duration+self.num_codebooks # frame_rate * audio_duration + embedding_dim (for the delay_pattern) -> 50 * 30 + 4 -> 1504
        self.dtype = dtype

        # Separate embedding layers for each codebook
        self.dec_embedding_layers = nn.ModuleList([
            nn.Embedding(self.vocab_size, self.size_params.d_model, dtype=self.dtype) for _ in range(self.num_codebooks)
        ])
        self.pos_embedding = MusicgenSinusoidalPositionalEmbedding(num_positions=self.max_seq_len, embedding_dim=self.size_params.d_model)

        # Explicit Encoder-Decoder Setup
        self.encoder = encoder

        dec_layer = get_musicgen_decoder(model_size=model_size, dtype=self.dtype)

        # Add a final LayerNorm to stabilize the output before the LM heads
        final_norm = nn.LayerNorm(self.size_params.d_model, dtype=self.dtype)

        self.decoder = nn.TransformerDecoder(
            dec_layer, 
            norm=final_norm,
            num_layers=self.size_params.num_decoder_layers
        )

        # One classification head for each codebook
        self.lm_heads = nn.ModuleList([
            nn.Linear(self.size_params.d_model, self.vocab_size, dtype=self.dtype) for _ in range(self.num_codebooks)
        ])

        # CFT learnable "null" context vector representing the absence of conditioning
        self.null_memory = nn.Parameter(torch.randn(1, 1, self.size_params.d_model, dtype=self.dtype)) # (1 batch, 1 seq_len, d_model)

    def forward(self, src, tgt, drop_conditioning=False):
        B, K, S = tgt.shape
        # print(f"\nTarget shape: {tgt.shape}\n")
        if src is not None: print(f"Src shape: {src.shape}\n")

        # CFG condition routing
        if self.encoder is None or drop_conditioning or src is None:
            # Unconditional path: Broadcast the learned null token across the batch.
            # This completely bypasses the encoder, saving compute!
            memory = self.null_memory.expand(B, 1, -1)
            # print(f"Memory is null, with shape: {memory.shape}\n")
        else:
            # Conditional path: Run standard encoder
            memory = self.encoder(src)
            # print(f"Memory came from encoder with shape: {memory.shape}\n")

        # TODO: Do we need a biderectional encoder mask?

        # Get embeddings per codebook
        dec_embs = torch.zeros(B, S, self.size_params.d_model, device=tgt.device, dtype=self.dtype)
        for i in range(K):
            dec_embs += self.dec_embedding_layers[i](tgt[:, i, :])

        # Scale and positional embedding
        dec_embs = dec_embs * math.sqrt(self.size_params.d_model) + self.pos_embedding(tgt).to(self.dtype)
        # print(f"Got decoder embedings with shape: {dec_embs.shape}\n")

        # Causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(S, device=tgt.device).to(self.dtype)
        # print(f"Got target mask with shape: {dec_embs.shape}")
        # print(f"Target mask:\n{tgt_mask}\n")

        out = self.decoder(tgt=dec_embs, memory=memory, tgt_mask=tgt_mask)
        #print(f"Got decoder output with shape:{out.shape}\n")
        # print(f"Got decoder output:\n{out}\n")

        # Inference on the codebook heads
        logits = torch.stack([head(out) for head in self.lm_heads], dim=1)
        # print(f"Got logits with shape:{logits.shape}\n")

        return logits

    def top_k_filtering(self, logits: torch.Tensor, top_k: int = 250, filter_value: float = -float("Inf")):
        """
        Filters logits to only keep the top k probabilities.
        """
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        return logits

    # TODO: KV cache
    @torch.no_grad()
    def generate(
        self,
        max_new_tokens: int,
        src: tp.Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 250,
    ):
        """
        Autoregressive generation loop for MusicGen.
        """
        self.eval()

        # Dynamically get the device the model is currently on
        device = next(self.parameters()).device

        # Determine batch size from the conditioning source, or default to 1
        B = src.shape[0] if src is not None else 1
        K = self.num_codebooks

        # Initialize the target tensor with padding tokens (acts as the BOS token setup)
        tgt = torch.full((B, K, 1), self.pad_token_id, dtype=torch.long, device=device)

        for step in range(max_new_tokens):
            # Forward pass
            # logits shape: (B, K, S, vocab_size)
            logits = self(src=src, tgt=tgt)

            # Extract logits from the last step in the sequence
            # next_token_logits shape: (B, K, vocab_size)
            next_token_logits = logits[:, :, -1, :]

            # Programatically setting the delay pattern for the padding tokens
            for k in range(K):
                if step < k:
                    # Force padding token
                    # 0;0 | 0;1 | 0;2 | 0;3  => a, P, P, P
                    # 1;0 | 1;1 | 1;2 | 1;3  => a, b, P, P ...
                    mask = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
                    mask[self.pad_token_id] = False
                    next_token_logits[:, k, mask] = -float("inf")
                else:
                    # Prevent padding token
                    next_token_logits[:, k, self.pad_token_id] = -float("inf")

            # TODO: ClassifierFreeGuidanceLogitsProcessor

            # Apply Temperature scaling
            next_token_logits = next_token_logits / temperature

            # Top-K Filtering
            next_token_logits = self.top_k_filtering(next_token_logits, top_k=top_k)

            # Convert to probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample from the distribution for each codebook
            # We reshape to (B * K, vocab_size) to use multinomial sampling efficiently
            probs_flat = probs.view(B * K, -1)
            next_tokens_flat = torch.multinomial(probs_flat, num_samples=1)

            # Reshape back to (B, K, 1)
            next_tokens = next_tokens_flat.view(B, K, 1)
            # print(f"----> Generate Next Tokens are: {next_tokens} ") #TODO REMOVE

            # Append the newly generated tokens to the sequence
            tgt = torch.cat([tgt, next_tokens], dim=-1)

        # Remove the initial padding token we used to kickstart the generation
        tgt = tgt[:, :, 1:]

        # Realign the codebooks to fix the delay pattern offset
        aligned_audio_tokens = DelayProvider.revert_delay_pattern(tgt)

        return aligned_audio_tokens