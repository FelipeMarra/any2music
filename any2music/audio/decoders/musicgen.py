import math
import logging
import typing as tp
from enum import Enum
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from any2music.base import BaseDecoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    "test": MusicGenSizeValues(d_model=1024, nhead=16, num_decoder_layers=12),
    "small": MusicGenSizeValues(d_model=1024, nhead=16, num_decoder_layers=24)
}

#########################################################
# Delay Provider
#########################################################

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
        # delay_pattern = delay_pattern + torch.tril(torch.ones((channel_codebooks, max_length), dtype=torch.bool))
        delay_pattern = delay_pattern | torch.tril(torch.ones((channel_codebooks, max_length), dtype=torch.bool), diagonal=0)

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
        valid_length = S - (K - 1)

        if valid_length <= 0:
            raise ValueError("Generated sequence is too short to be aligned.")

        aligned_ids = torch.zeros((B, K, valid_length), dtype=generated_ids.dtype, device=generated_ids.device)

        for codebook_idx in range(K):
            start_idx = codebook_idx
            end_idx = start_idx + valid_length
            aligned_ids[:, codebook_idx, :] = generated_ids[:, codebook_idx, start_idx:end_idx]

        return aligned_ids


class MusicgenSinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int, dtype = torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.embedding_dim = embedding_dim
        self.num_positions = num_positions
        self.make_weights(num_positions, embedding_dim)

    def make_weights(self, num_embeddings: int, embedding_dim: int):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim)
        if hasattr(self, "weights"):
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)
        self.register_buffer("weights", emb_weights, persistent=False)

    def get_embedding(self, num_embeddings: int, embedding_dim: int):
        half_dim = embedding_dim // 2
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb.to(self.dtype)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        _, _, seq_len = input_ids.size() 
        position_ids = (torch.arange(seq_len) + past_key_values_length).to(input_ids.device)
        
        # TODO was: if seq_len > self.weights.size(0): # type: ignore
        target_len = seq_len + past_key_values_length
        if target_len > self.weights.size(0): # type: ignore
            self.make_weights(target_len, self.embedding_dim)
            
        return self.weights.index_select(0, position_ids.view(-1)).detach() # type: ignore


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dtype=torch.bfloat16):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self.w_k = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self.w_v = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self.w_out = nn.Linear(d_model, d_model, bias=False, dtype=dtype)


    def forward(self, x, tgt_mask, past_kv=None):
        B, S, D = x.shape

        # q, k and v will be (B, n_heads, S, head_size)
        q = self.w_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=tgt_mask)
        attn = attn.transpose(1, 2).reshape(B, S, D)

        return self.w_out(attn), present_kv


class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dtype=torch.bfloat16):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self.w_k = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self.w_v = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self.w_out = nn.Linear(d_model, d_model, bias=False, dtype=dtype)


    def forward(self, x, memory, memory_mask=None, past_kv=None):
        B, S, D = x.shape

        # B, self.n_heads, S, self.head_dim
        q = self.w_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if past_kv is None:
            B_m, S_m, _ = memory.shape

            # B_m, self.n_heads, S_m, self.head_dim
            k = self.w_k(memory).view(B_m, S_m, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.w_v(memory).view(B_m, S_m, self.n_heads, self.head_dim).transpose(1, 2)

            present_kv = (k, v)
        else:
            k, v = past_kv
            present_kv = past_kv

        attn_mask = None
        if memory_mask is not None:
            attn_mask = memory_mask.view(B, 1, 1, -1)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(B, S, D)
        return self.w_out(attn), present_kv


class MusicGenDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1, dtype=torch.bfloat16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0

        # Attention
        self.self_att = CausalSelfAttention(d_model, n_heads, dtype)
        self.cross_att = CrossAttention(d_model, n_heads, dtype)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward, dtype=dtype)
        self.linear2 = nn.Linear(dim_feedforward, d_model, dtype=dtype)
        self.activation = nn.GELU()

        # Norms & Dropout
        self.norm1 = nn.LayerNorm(d_model, dtype=dtype)
        self.norm2 = nn.LayerNorm(d_model, dtype=dtype)
        self.norm3 = nn.LayerNorm(d_model, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        tgt: torch.Tensor, 
        memory: torch.Tensor, 
        tgt_mask: tp.Optional[torch.Tensor] = None, 
        memory_mask: tp.Optional[torch.Tensor] = None,
        layer_past_kv: tp.Optional[dict] = None
    ):
        if layer_past_kv is None:
            layer_past_kv = {'self': None, 'cross': None}

        # Self-Attention 
        norm_tgt = self.norm1(tgt)
        out, present_self_kv = self.self_att(x=norm_tgt, tgt_mask=tgt_mask, past_kv=layer_past_kv['self'])
        tgt = tgt + self.dropout(out)

        # Cross-Attention
        norm_tgt = self.norm2(tgt)
        out, present_cross_kv = self.cross_att(x=norm_tgt, memory=memory, memory_mask=memory_mask, past_kv=layer_past_kv['cross'])
        tgt = tgt + self.dropout(out)

        # Feedforward
        norm_tgt = self.norm3(tgt)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(norm_tgt))))
        tgt = tgt + self.dropout(ff_out)

        new_layer_kv = {'self': present_self_kv, 'cross': present_cross_kv}
        return tgt, new_layer_kv


class MusicGenDecoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, norm: nn.Module):
        super().__init__()
        self.layers = layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, past_kv=None):
        """
            MusicGen Decoder's forward

            memory_mask: The same logic as HuggingFace T5, that is, 1 for sequence and 0 for padding 
        """
        if past_kv is None:
            past_kv = [None] * len(self.layers)

        new_kv = []
        for layer, layer_past in zip(self.layers, past_kv):
            tgt, new_layer_kv = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, layer_past_kv=layer_past)
            new_kv.append(new_layer_kv)

        if self.norm is not None:
            tgt = self.norm(tgt)

        return tgt, new_kv


#########################################################
# MusicGen Transformer
#########################################################

class MusicGenTransformer(BaseDecoder):
    def __init__(
            self, 
            vocab_size:int,
            pad_token_id,
            eos_token_id,
            bos_token_id,
            frame_rate:int,
            audio_duration:int,
            encoder:tp.Optional[nn.TransformerEncoder] = None,
            model_size:MusicGenSize=MusicGenSize.SMALL, 
            dtype:torch.dtype=torch.bfloat16,
            invert_src_mask=False
        ):
        """
            src/memory mask: True or 1 means valid token & False or 0 means a masked token
        """
        super().__init__()
        self.size_params = MUSICGEN_SIZES[model_size.value]
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.num_codebooks = 4
        self.max_seq_len = frame_rate * audio_duration + self.num_codebooks + 5
        self.dtype = dtype
        self.invert_src_mask = invert_src_mask

        self.dec_embedding_layers = nn.ModuleList([
            nn.Embedding(self.vocab_size, self.size_params.d_model, dtype=self.dtype) for _ in range(self.num_codebooks)
        ])
        self.pos_embedding = MusicgenSinusoidalPositionalEmbedding(num_positions=self.max_seq_len, embedding_dim=self.size_params.d_model)

        self.encoder = encoder

        dec_layers = nn.ModuleList([
            MusicGenDecoderLayer(
                d_model=self.size_params.d_model,
                n_heads=self.size_params.nhead,
                dim_feedforward=self.size_params.d_model * 4,
                dropout=0.1,
                dtype=self.dtype
            ) for _ in range(self.size_params.num_decoder_layers)
        ])
        
        final_norm = nn.LayerNorm(self.size_params.d_model, dtype=self.dtype)
        self.decoder = MusicGenDecoder(layers=dec_layers, norm=final_norm)

        self.lm_heads = nn.ModuleList([
            nn.Linear(self.size_params.d_model, self.vocab_size, dtype=self.dtype) for _ in range(self.num_codebooks)
        ])

        self.null_memory = nn.Parameter(torch.randn(1, 1, self.size_params.d_model, dtype=self.dtype))


    def forward(self, src, tgt, drop_conditioning=False, src_mask=None, past_kv=None, pos_offset:int=0, pre_comp_src=False):
        """
            src/memory mask: True or 1 means valid token & False or 0 means a masked token
        """
        B, K, S = tgt.shape

        if pre_comp_src:
            memory = src
            memory_mask = src_mask
        else:
            if (src is not None) and (self.encoder is not None) and (not drop_conditioning):
                memory, memory_mask = self.encoder(src)
            elif (self.encoder is None) and (src is None) or drop_conditioning:
                memory = self.null_memory.expand(B, 1, -1)
                memory_mask = None
            elif(src is not None) and (self.encoder is None) and (not drop_conditioning): # pre_comp_src is False but matches an equivalent condition
                memory = src
                memory_mask = src_mask

        # in the T5 memory_mask False means sequence, and for SDPA true means sequence and false means padding
        if memory_mask is not None:
            if memory_mask.dtype != torch.bool:
                memory_mask = memory_mask == 1

            if self.invert_src_mask:
                memory_mask = ~memory_mask

        # Get embeddings per codebook
        dec_embs = torch.zeros(B, S, self.size_params.d_model, device=tgt.device, dtype=self.dtype)
        for i in range(K):
            dec_embs += self.dec_embedding_layers[i](tgt[:, i, :])

        # Handle Positional Embeddings with offset for single-token generation
        end = pos_offset + S
        if end > self.pos_embedding.weights.size(0):
            self.pos_embedding.make_weights(end, self.pos_embedding.embedding_dim)

        pos_emb = self.pos_embedding.weights[pos_offset : end].to(self.dtype)  # (S, D)
        dec_embs = dec_embs * math.sqrt(self.size_params.d_model) + pos_emb

        tgt_mask = None
        if S > 1:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(S, device=tgt.device).to(self.dtype)

        out, new_kv = self.decoder(
            tgt=dec_embs, 
            memory=memory, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask,
            past_kv=past_kv
        )

        logits = torch.stack([head(out) for head in self.lm_heads], dim=1)
        return logits, new_kv


    def top_k_filtering(self, logits: torch.Tensor, top_k: int = 250, filter_value: float = -float("Inf")):
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            logits = logits.clone()
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        return logits


    def top_p_filtering(self, logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create a mask to remove tokens outside top_p
        # We shift the mask to ensure we keep the first token that crosses the threshold
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter mask back to original token ordering
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        
        # Filter logits, re-normalize probabilities, and sample
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = float('-inf')
        
        filtered_probs = F.softmax(filtered_logits, dim=-1)
        
        return filtered_probs


    @torch.no_grad()
    def generate(
        self,
        max_new_tokens: int,
        src: tp.Optional[torch.Tensor] = None,
        batch_size = 1,
        src_mask=None,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p=0.90,
        cfg_scale: float = 3.0
    ):

        assert max_new_tokens <= self.max_seq_len

        self.eval()
        device = next(self.parameters()).device
        B = src.shape[0] if src is not None else batch_size
        K = self.num_codebooks

        # Initial prompt tokens (shape B, K, 2)
        tgt = torch.full((B, K, 2), self.pad_token_id, dtype=torch.long, device=device)
        tgt[:, 0, 1] = self.bos_token_id

        # Init conditioning
        if src is not None:
            if self.encoder is not None:
                memory, memory_mask = self.encoder(src)
            else:
                memory, memory_mask = src, src_mask
        else:
            memory, memory_mask = None, None

        # Initial sequence (length 2) to initialize KV caches
        if src is not None:
            logits_cond, kv_cond = self(src=memory, tgt=tgt, src_mask=memory_mask, past_kv=None, pre_comp_src=True)
            logits_uncond, kv_uncond = self(src=None, tgt=tgt, drop_conditioning=True, past_kv=None)
            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
        else:
            logits, kv_uncond = self(src=None, tgt=tgt, drop_conditioning=True, past_kv=None)
            kv_cond = None

        next_token_logits = logits[:, :, -1, :]

        # Incremental Decoding Loop
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            for k in range(K):
                if step < k:
                    mask = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
                    if step == k - 1:
                        mask[self.bos_token_id] = False
                    else:
                        mask[self.pad_token_id] = False
                    next_token_logits[:, k, mask] = -float("inf")
                else:
                    next_token_logits[:, k, self.pad_token_id] = -float("inf")
                    next_token_logits[:, k, self.bos_token_id] = -float("inf")

            next_token_logits = next_token_logits / temperature
            next_token_logits = self.top_k_filtering(next_token_logits, top_k=top_k)
            probs = self.top_p_filtering(next_token_logits, p=top_p)

            probs_flat = probs.view(B * K, -1)
            next_tokens_flat = torch.multinomial(probs_flat, num_samples=1)
            next_tokens = next_tokens_flat.view(B, K, 1) # Shape: (B, K, 1)

            # Logic for sequences in the batch that already finisehed
            eos_hit = (next_tokens[:, K - 1, 0] == self.eos_token_id) # (B,)
            newly_finished = eos_hit & ~finished

            # Mask rows that were already finished with eos to trigger valid_mask later
            already_finished = finished.clone()
            if already_finished.any():
                next_tokens[already_finished] = self.eos_token_id

            # Update finished mask
            finished |= newly_finished

            tgt = torch.cat([tgt, next_tokens], dim=-1)

            if finished.all():
                break

            # Step forward with ONLY the single newest token (length 1)
            pos_offset = tgt.shape[-1] - 1
            if src is not None:
                logits_cond, kv_cond = self(src=memory, tgt=next_tokens, src_mask=memory_mask, past_kv=kv_cond, pos_offset=pos_offset, pre_comp_src=True)
                logits_uncond, kv_uncond = self(src=None, tgt=next_tokens, drop_conditioning=True, past_kv=kv_uncond, pos_offset=pos_offset)
                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
            else:
                logits, kv_uncond = self(src=None, tgt=next_tokens, drop_conditioning=True, past_kv=kv_uncond, pos_offset=pos_offset)

            next_token_logits = logits[:, :, -1, :]

        aligned_audio_tokens = DelayProvider.revert_delay_pattern(tgt)
        aligned_audio_tokens = aligned_audio_tokens[:, :, 2:]

        invalid_mask = (aligned_audio_tokens < 0) | (aligned_audio_tokens >= self.eos_token_id)

        valid_mask = (aligned_audio_tokens >= 0) & (aligned_audio_tokens < self.eos_token_id)
        valid_lengths = valid_mask.all(dim=1).sum(dim=-1) # Shape: (B,)

        aligned_audio_tokens[invalid_mask] = 0

        return aligned_audio_tokens, valid_lengths
