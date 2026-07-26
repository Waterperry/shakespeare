from functools import partial
from typing import Any, Callable

import numpy as np
import torch

from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torch.nested import nested_tensor

from shakespeare.components import transformer as custom_trf

def _padding_collate_fn(
    tokenizeds: list[dict[str, Tensor]],
    pad_token_id: int,
    max_sequence_length: int,
) -> dict[str, Tensor]:
    stack_keys = [
        "input_ids",
        # "attention_mask",
        # "token_type_ids",
    ]

    collated_tensors: dict[str, Any] = {k: list() for k in stack_keys}

    for tokenized in tokenizeds:
        for k in stack_keys:
            collated_tensors[k].append(tokenized[k][:max_sequence_length])

    # TODO: time this against the other method (torch.pad + torch.stack)
    return {
        "input_ids": nested_tensor(collated_tensors["input_ids"], layout=torch.jagged).to_padded_tensor(pad_token_id),
    }


class Model(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_decoder_layers: int,
        ffn_dim: int,
        max_seq_len: int,
        dropout: float,
        use_custom_pos_enc: bool,
        use_custom_transformer: bool,
    ) -> None:
        super().__init__()

        # copy init params
        self._vocab_size = vocab_size
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._num_decoder_layers = num_decoder_layers
        self._ffn_dim = ffn_dim
        self._max_seq_len = max_seq_len
        self._dropout = dropout
        self._use_custom_pos_enc = use_custom_pos_enc
        self._use_custom_transformer = use_custom_transformer

        # actual things
        self._embedding = nn.Embedding(vocab_size, embed_dim)

        if use_custom_pos_enc:
            self._pos_encoding = custom_trf.PositionalEncoding(max_seq_len, embed_dim)
        else:
            self._pos_encoding = nn.Embedding(max_seq_len, embed_dim)

        if use_custom_transformer:
            self._transformer = custom_trf.TransformerEncoder(
                n_blocks=num_decoder_layers,
                input_dim=embed_dim,
                n_heads=num_heads,
                qkv_dim=embed_dim,
                ffn_dim=ffn_dim,
                dropout_prob=dropout,
            )
        else:
            self._transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=ffn_dim,
                    dropout=dropout,
                    batch_first=True,
                ),
                num_layers=num_decoder_layers,
            )

        self._output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: Tensor, pad_token_id: int | None = None) -> Tensor:
        seq_len = x.size(1)
        x_emb = self._embedding(x)
        if self._use_custom_pos_enc:
            x_emb = self._pos_encoding(x_emb)  # the custom PosEnc layer takes the actual token embeddings
        else:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x_emb = x_emb + self._pos_encoding(positions)  # the nn.Embedding layer takes token indices
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        src_key_padding_mask: Tensor | None = None
        if pad_token_id is not None:
            src_key_padding_mask = torch.where(x == pad_token_id, float('-inf'), 0.0)

        out = self._transformer(
            x_emb,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True,
        )
        return self._output_proj(out)

    def embed(self, x: Tensor, pad_token_id: int | None = None) -> tuple[Tensor, Tensor]:
        seq_len = x.size(1)
        x_emb = self._embedding(x)
        if self._use_custom_pos_enc:
            x_emb = self._pos_encoding(x_emb)  # the custom PosEnc layer takes the actual token embeddings
        else:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x_emb = x_emb + self._pos_encoding(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        src_key_padding_mask: Tensor | None = None
        if pad_token_id is not None:
            src_key_padding_mask = torch.where(x == pad_token_id, float('-inf'), 0.0)

        out = self._transformer(
            x_emb,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True,
        )
        return self._output_proj(out), out

    def get_collate_function(
        self,
        pad_token_id: int,
    ) -> Callable[[list[dict[str, Tensor]]], dict[str, Tensor]]: 
        return partial(
            _padding_collate_fn,
            pad_token_id=pad_token_id,
            max_sequence_length=self._max_seq_len,
        )
