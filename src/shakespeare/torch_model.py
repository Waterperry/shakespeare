from functools import partial
from typing import Any, Callable

import torch

from torch import Tensor, nn
from torch.nested import nested_tensor


def _padding_collate_fn(
    tokenizeds: list[dict[str, Tensor]],
    pad_token_id: int,
    max_sequence_length: int,
) -> dict[str, Tensor]:
    stack_keys = [
        "input_ids",
        "attention_mask",
        # "token_type_ids",
    ]

    collated_tensors: dict[str, Any] = {k: list() for k in stack_keys}

    for tokenized in tokenizeds:
        for k in stack_keys:
            collated_tensors[k].append(tokenized[k][:max_sequence_length])

    # TODO: time this against the other method (torch.pad + torch.stack)
    return {
        k: nested_tensor(vs, layout=torch.jagged).to_padded_tensor(pad_token_id)
        for k, vs in collated_tensors.items()
    }



class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_decoder_layers: int,
        ffn_dim: int,
        max_seq_len: int,
        dropout: float,
        use_rope_emb: bool,
        use_custom_rope_emb: bool,
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
        self._use_rope_emb = use_rope_emb
        self._use_custom_rope_emb = use_custom_rope_emb
        self._use_custom_transformer = use_custom_transformer

        # actual things
        self._embedding = nn.Embedding(vocab_size, embed_dim)

        if use_rope_emb:
            if use_custom_rope_emb:
                raise NotImplementedError  # TODO: implement
            else:
                raise NotImplementedError  # TODO: implement
        else:
            self._pos_encoding = nn.Embedding(max_seq_len, embed_dim)

        if use_custom_transformer:
            raise NotImplementedError  # TODO: implement
        else:
            self._transformer = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
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
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x_emb = self._embedding(x) + self._pos_encoding(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        # ignore padding in tgt sequence
        tgt_key_padding_mask: Tensor | None = None
        if pad_token_id is not None:
            # construct the boolean mask first, but change it to float mask (used additively)
            # since pytorch emits warning about bool <=> float comparisons
            tgt_key_padding_mask = (x == pad_token_id).float()
            tgt_key_padding_mask[tgt_key_padding_mask == 0.] = -torch.inf
            tgt_key_padding_mask[tgt_key_padding_mask == 1.] = 0.0

        memory = torch.zeros_like(x_emb, device=x_emb.device, requires_grad=False)
        out = self._transformer(
            x_emb,
            memory,
            tgt_is_causal=True,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self._output_proj(out)


    def get_collate_function(
        self,
        pad_token_id: int,
    ) -> Callable[[list[dict[str, Tensor]]], dict[str, Tensor]]: 
        return partial(
            _padding_collate_fn,
            pad_token_id=pad_token_id,
            max_sequence_length=self._max_seq_len,
        )