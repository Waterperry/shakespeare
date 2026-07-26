from typing import Any

import numpy as np

import torch

from torch import nn


def fit_mask_to_dims(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim < 2:
        raise RuntimeError(f"Mask had invalid number of dims ({mask.ndim = }). Must be at least 2.")

    match mask.ndim:
        case 4:
            return mask  # 4D is expected for [batch, n_heads, seq, seq]
        case 3:
            return mask.unsqueeze(1)  # broadcast over heads since we have been given batch dims
        case 2:
            return mask.unsqueeze(0).unsqueeze(0)  # broadcast over heads and batch
        case _:
            raise RuntimeError(f"mask dims were too large ({mask.ndim = }). Expected between 2 and 4 dimensions.")


def scaled_dot_product_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    src_key_padding_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
        tuple: (values, attention)
    """
    qk_dim: int = q.shape[-1]
    assert k.shape[-1] == qk_dim

    raw_attn_values: torch.Tensor = q @ k.transpose(-2, -1)
    scaled_attn_values: torch.Tensor = raw_attn_values / np.sqrt(qk_dim)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scaled_attn_values.masked_fill_(attn_mask, float("-inf"))
        else:
            scaled_attn_values = scaled_attn_values + attn_mask

    if src_key_padding_mask is not None:
        src_key_padding_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1 (src_attn), tgt_attn]
        if src_key_padding_mask.dtype == torch.bool:
            scaled_attn_values.masked_fill_(src_key_padding_mask, float("-inf"))
        else:
            scaled_attn_values += src_key_padding_mask
    
    attn = scaled_attn_values.softmax(dim=-1)
    values = attn @ v
    return values, attn

def _test_sdpa() -> None:
    """A small hand-worked example. batch_size=1, seq_len=2, d_k=2, d_v=3"""
    q = torch.tensor([[[1, 2], [3, 4]]]).to(torch.float32)
    k = torch.tensor([[[2, 3], [4, 5]]]).to(torch.float32)
    v = torch.tensor([[[0, 1, 2], [3, 4, 5]]]).to(torch.float32)

    values, attn = scaled_dot_product_attn(q, k, v)

    expected_attn = torch.tensor([[
        [0.0142, 0.986],
        [5.02e-5, 0.9999],
    ]])
    expected_values = torch.tensor([[
        [2.958, 3.9582, 4.9584],
        [2.9997, 3.9997, 4.9996],
    ]])

    assert torch.isclose(expected_attn, attn, atol=1e-3).all()
    assert torch.isclose(expected_values, values, atol=1e-3).all()
    print("SDPA: ok")


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        q_dim: int,
        v_dim: int,
    ) -> None:
        super().__init__()
        self._q_dim: int = q_dim
        self._v_dim: int = v_dim
        self._n_heads: int = n_heads
        self._input_dim: int = input_dim

        self.w_qkv: nn.Linear = nn.Linear(input_dim, 3*q_dim, bias=False)  # NOTE: output size is 3*input_dim as per original paper but not necessary
        assert v_dim == q_dim, "Different v/q dims are currently not supported"
        assert v_dim % n_heads == 0
        self.w_o: nn.Linear = nn.Linear(v_dim, input_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attn_values: bool = False,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            mask = fit_mask_to_dims(mask)

        batch_size, seq_len, _ = x.shape
        qkv = self.w_qkv(x)
        qkv_heads: torch.Tensor = qkv.reshape(batch_size, seq_len, self._n_heads, -1)  # [batch, seq, head, qkv]
        q, k, v = torch.split(qkv_heads.permute(0, 2, 1, 3), self._q_dim // self._n_heads, dim=-1)  # permute to [batch, head, seq, qkv]

        values, attn = scaled_dot_product_attn(q, k, v, attn_mask=mask, src_key_padding_mask=src_key_padding_mask)
        values_cat = values.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self._q_dim)  # permute back to [batch, seq, head, qkv]

        if return_attn_values:
            return self.w_o(values_cat), attn
        
        return self.w_o(values_cat)

    @staticmethod
    def _test_mha() -> None:
        # small hand-worked example
        mha = MultiHeadAttention(2, 2, 2, 2)
        mha.w_qkv.weight = nn.Parameter(torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]).float().T)
        mha.w_o.weight = nn.Parameter(torch.tensor([[1, 2], [3, 4]]).float().T)
        x = torch.tensor([[[1, 2], [3, 4]]]).to(torch.float32)
        out = mha(x)
        expected_out = torch.tensor([[[215., 312]]]).float()
        assert torch.allclose(out, expected_out, atol=1e-3)

        # test src_key_padding_mask
        BATCH_SIZE = 11
        SEQ_LEN = 16
        HIDDEN_DIM = 128

        mha = MultiHeadAttention(HIDDEN_DIM, 4, 16, 16)
        ins = torch.rand((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM))
        causal_mask = nn.Transformer.generate_square_subsequent_mask(SEQ_LEN)
        for seq_len in range(3, SEQ_LEN):

            truncated_causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

            outs_padded = mha(ins, mask=causal_mask)
            outs_unpadded = mha(ins[:, :seq_len], mask=truncated_causal_mask)
            assert torch.allclose(outs_padded[:, :seq_len], outs_unpadded, atol=1e-5), f"MHSA pad test failed on {seq_len=}"


        # use src_key_padding_mask to ignore last two tokens
        src_key_padding_mask = torch.zeros((BATCH_SIZE, SEQ_LEN)).bool()
        src_key_padding_mask[:, -2:] = float("-inf")
        outs_padded = mha(ins, src_key_padding_mask=src_key_padding_mask)
        outs_unpadded = mha(ins[:, :-2])
        assert torch.allclose(outs_padded[:, :-2], outs_unpadded, atol=1e-5), f"MHSA src key pad test failed"

        print("MHSA: ok")



class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        qkv_dim: int,
        ffn_dim: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()

        self.mha = MultiHeadAttention(input_dim, n_heads, qkv_dim, qkv_dim)
        self.n1: nn.LayerNorm = nn.LayerNorm(input_dim)
        self.n2: nn.LayerNorm = nn.LayerNorm(input_dim)

        self.ff: nn.Module = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(ffn_dim, input_dim),
        )

        self._training: bool = True

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attn_values: bool = False,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        mha_out = self.mha(x, mask=mask, return_attn_values=return_attn_values, src_key_padding_mask=src_key_padding_mask)
        if return_attn_values:
            mha_out, attn_values = mha_out  # split tuple[Tensor, Tensor]

        attn_out = self.n1(mha_out + x)
        h1 = self.ff(attn_out)
        h2 = self.n2(attn_out + h1)

        if return_attn_values:
            return h2, attn_values
        return h2

    @staticmethod
    def _test_trf() -> None:
        BATCH_SIZE = 16
        SEQ_LEN = 7
        N_HEADS = 2
        INPUT_DIM = 128
        QKV_DIM = 8
        ffn_DIM = 16

        enc = TransformerEncoderBlock(INPUT_DIM, N_HEADS, QKV_DIM, ffn_DIM, dropout_prob=0.1)
        ins = torch.rand((BATCH_SIZE, SEQ_LEN, INPUT_DIM))

        out = enc(ins)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, INPUT_DIM)
        print("TrEB: ok")


class TransformerEncoder(nn.Module):
    def __init__(self, n_blocks: int, **encoder_layer_args: Any) -> None:
        super().__init__()

        self._blocks: nn.ParameterList = nn.ParameterList(
            TransformerEncoderBlock(**encoder_layer_args)
            for _ in range(n_blocks)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attn_values: bool = False,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = True,  # unused, but for compatibility
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        attns: list[torch.Tensor] = []

        for block in self._blocks:
            x = block(x, mask=mask, return_attn_values=return_attn_values, src_key_padding_mask=src_key_padding_mask)
            if return_attn_values:
                attns.append(x[1])
                x = x[0]

        if return_attn_values:
            return x, attns
        return x

    @staticmethod
    def _test_trf() -> None:
        INPUT_DIMS = 16
        BATCH_SIZE = 9
        SEQ_LEN = 7
        trf = TransformerEncoder(6, input_dim=INPUT_DIMS, n_heads=4, qkv_dim=24, ffn_dim=128, dropout_prob=0.2)
        ins = torch.rand((BATCH_SIZE, SEQ_LEN, INPUT_DIMS))

        out = trf(ins)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, INPUT_DIMS)
        print("TrEn: ok")


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, input_dims: int) -> None:
        super().__init__()
        self._input_dims: int = input_dims
        self._max_seq_len: int = max_seq_len

        positions = torch.arange(0, self._max_seq_len, dtype=torch.float32).unsqueeze(-1)
        denom_dims = torch.pow(10_000, torch.arange(0, self._input_dims, 2) / self._input_dims)

        sins = torch.sin(positions / denom_dims)
        coss = torch.cos(positions / denom_dims)

        self._pos_encs = nn.Buffer(
            torch.stack((sins, coss), dim=-1).view(self._max_seq_len, self._input_dims).unsqueeze(0)
        )

    def forward(self, x: torch.Tensor, pad_token_id: int | None = None) -> torch.Tensor:
        # broadcast along batch dim, limit to actual sequence length
        return x + self._pos_encs[:, :x.shape[1]]

    @staticmethod
    def get_pos_enc_naive(input_dims: int, i: int, pos: int) -> float:
        from math import sin, cos, pow

        is_even = i % 2 == 0

        denom = pow(10_000, (i if is_even else i - 1)/input_dims)
        if i % 2 == 0:
            return sin(pos / denom)
        return cos(pos / denom)

    @staticmethod
    def _test_penc(atol: float = 1e-6) -> None:
        input_dims = 4
        max_seq_len = 3

        pe = PositionalEncoding(max_seq_len, input_dims)
        ins = torch.rand((1, max_seq_len, input_dims))
        outs = pe(ins)

        # check sane output shape
        assert outs.shape == (1, max_seq_len, input_dims)

        # check pos_encs tensor lines up with expectation
        for pos in range(max_seq_len):
            for i in range(input_dims):
                naive = pe.get_pos_enc_naive(input_dims, i, pos)
                assert torch.abs(pe._pos_encs[0, pos, i] - naive) < atol

        print("PEnc: ok")


def main() -> None:
    _test_sdpa()
    MultiHeadAttention._test_mha()
    TransformerEncoderBlock._test_trf()
    TransformerEncoder._test_trf()
    PositionalEncoding._test_penc()


if __name__ == "__main__":
    main()