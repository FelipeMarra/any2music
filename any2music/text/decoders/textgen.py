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

class TextGenSize(Enum):
    TEST = "test"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

@dataclass
class TextGenSizeValues():
    d_model: int
    nhead: int
    num_decoder_layers: int

TEXTGEN_SIZES:tp.Dict[str, TextGenSizeValues] = {
    "test": TextGenSizeValues(d_model=1024, nhead=16, num_decoder_layers=12), # 3213MiB
    "small": TextGenSizeValues(d_model=1024, nhead=16, num_decoder_layers=24)
}

#########################################################
# Transformer Layer Components
#########################################################

# Class obtained from https://github.com/huggingface/transformers/blob/10555512868d663ee1ff627e4f5c5c260114235b/src/transformers/models/musicgen/modeling_musicgen.py#L106
class TextgenSinusoidalPositionalEmbedding(nn.Module):
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
        _, seq_len = input_ids.size() # expects batch, seq_len
        # Create the position ids from the input token ids.
        position_ids = (torch.arange(seq_len) + past_key_values_length).to(input_ids.device)
        # expand embeddings if needed
        if seq_len > self.weights.size(0): # type: ignore
            self.make_weights(seq_len, self.embedding_dim)
        return self.weights.index_select(0, position_ids.view(-1)).detach() # type: ignore

#########################################################
# Transformer Layers
#########################################################

def get_textgen_decoder(
        model_size:TextGenSize=TextGenSize.SMALL,
        dtype=torch.bfloat16
    ) -> nn.TransformerDecoderLayer:

    size_params = TEXTGEN_SIZES[model_size.value]

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

class TextGenTransformer(BaseDecoder):
    def __init__(
            self, 
            vocab_size:int,
            pad_token_id,
            eos_token_id,
            bos_token_id,
            max_seq_len,
            encoder:tp.Optional[nn.TransformerEncoder] = None,
            model_size:TextGenSize=TextGenSize.SMALL, 
            dtype:torch.dtype=torch.bfloat16
        ):
        super().__init__()
        self.size_params = TEXTGEN_SIZES[model_size.value]
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        # Embedding layers 
        self.dec_embedding_layer = nn.Embedding(self.vocab_size, self.size_params.d_model, dtype=self.dtype)
        self.pos_embedding = TextgenSinusoidalPositionalEmbedding(num_positions=self.max_seq_len, embedding_dim=self.size_params.d_model)

        # Explicit Encoder-Decoder Setup
        self.encoder = encoder

        # Add a final LayerNorm to stabilize the output before the LM head
        final_norm = nn.LayerNorm(self.size_params.d_model, dtype=self.dtype)
        dec_layer = get_textgen_decoder(model_size=model_size, dtype=self.dtype)
        self.decoder = nn.TransformerDecoder(
            dec_layer, 
            norm=final_norm,
            num_layers=self.size_params.num_decoder_layers
        )

        # Classification head
        self.lm_head = nn.Linear(self.size_params.d_model, self.vocab_size, dtype=self.dtype)

        # CFT learnable "null" context vector representing the absence of conditioning
        self.null_memory = nn.Parameter(torch.randn(1, 1, self.size_params.d_model, dtype=self.dtype)) # (1 batch, 1 seq_len, d_model)

    def forward(self, src, tgt, drop_conditioning=False, src_mask=None):
        B, S = tgt.shape
        # print(f"\nTarget shape: {tgt.shape}\n")
        #if src is not None: print(f"Src shape: {src.shape}\n")

        # CFG condition routing
        ## Conditional path: Run standard encoder
        if src is not None and self.encoder is not None and not drop_conditioning:
            memory, memory_mask = self.encoder(src)
            # print(f"Memory came from encoder with shape: {memory.shape}\n")

        ## Unconditional path: Broadcast the learned null token across the batch
        elif self.encoder is None and src is None or drop_conditioning:
            memory = self.null_memory.expand(B, 1, -1)
            memory_mask = None
            # print(f"Memory is null, with shape: {memory.shape}\n")

        ## Conditional path when the src comes already encoded 
        elif src is not None and self.encoder is None and not drop_conditioning:
            memory = src
            memory_mask = src_mask
            # print(f"Memory came ready w/o need to encode: {memory.shape}\n")

        # Get embeddings
        dec_embs = torch.zeros(B, S, self.size_params.d_model, device=tgt.device, dtype=self.dtype)
        dec_embs += self.dec_embedding_layer(tgt)

        # Scale and positional embedding
        dec_embs = dec_embs * math.sqrt(self.size_params.d_model) + self.pos_embedding(tgt).to(self.dtype)
        # print(f"Got decoder embedings with shape: {dec_embs.shape}\n")

        # Causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(S, device=tgt.device).to(self.dtype)
        # print(f"Got target mask with shape: {dec_embs.shape}")
        # print(f"Target mask:\n{tgt_mask}\n")

        out = self.decoder(tgt=dec_embs, memory=memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)
        #print(f"Got decoder output with shape:{out.shape}\n")
        # print(f"Got decoder output:\n{out}\n")

        # Inference on the codebook heads
        logits = self.lm_head(out)
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
        src_mask=None,
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

        # Initialize the target tensor with BOS token
        tgt = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # Forward pass
            # logits shape: (B, S, vocab_size)
            logits = self(src=src, tgt=tgt, src_mask=src_mask)

            # Extract logits from the last step in the sequence
            # next_token_logits shape: (B, S, vocab_size)
            next_token_logits = logits[:, -1, :]

            # TODO: ClassifierFreeGuidanceLogitsProcessor

            # Apply Temperature scaling
            next_token_logits = next_token_logits / temperature

            # Top-K Filtering
            next_token_logits = self.top_k_filtering(next_token_logits, top_k=top_k)

            # Convert to probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample
            next_tokens = torch.multinomial(probs, num_samples=1)

            if [self.eos_token_id] in next_tokens.tolist():
                break

            # Append the newly generated tokens to the sequence
            tgt = torch.cat([tgt, next_tokens], dim=-1)

        # Remove the initial bos token we used to kickstart the generation
        tgt = tgt[:, 1:]

        return tgt