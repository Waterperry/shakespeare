from copy import deepcopy

import streamlit as st
import torch

from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

from shakespeare.torch_model import Model


@torch.no_grad
def decode_from_seed(
    tokens: list[int],
    model: Model,
    tokenizer: PreTrainedTokenizerFast,
    top_k: int,
    temperature: float,
    max_len: int = 150,
    device: str = "cpu",
) -> str:
    tokens = deepcopy(tokens)
    while len(tokens) < max_len:
        in_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
        outs = model(in_tensor)
        warmed_logits = outs[-1, -1] / temperature
        topk_probs, topk_indices = torch.softmax(warmed_logits, dim=-1).topk(top_k)
        out_id = topk_indices[torch.multinomial(topk_probs, 1)].item()
        tokens.append(out_id)
        if out_id == tokenizer.eos_token_id:
            break
    return "".join(tokenizer.convert_ids_to_tokens(tokens[1:-1]))


@st.cache_resource
def load_model_and_tokenizer() -> tuple[Model, PreTrainedTokenizerFast]:
    return (
        Model.from_pretrained("outs/model"),
        AutoTokenizer.from_pretrained("outs/tokenizer"),
    )


@torch.no_grad
def main() -> None:
    st.title("Shakespeare Sentence Generator")
    with st.spinner("Loading model and tokenizer..."):
        model, tokenizer = load_model_and_tokenizer()

    temp = st.slider("Temperature", min_value=1e-3, max_value=100., value=1., step=1e-1)
    top_k = st.slider("Top-K Tokens", min_value=1, max_value=100, value=1, step=1)
    max_tokens = st.slider("Max Tokens", min_value=1, max_value=150, value=100, step=1)

    prompt = st.text_input("Prompt Start")

    if prompt:
        seed_tokens = tokenizer(prompt, truncate=True)["input_ids"][:-1]  # remove [EOS]
    else:
        seed_tokens = [tokenizer.bos_token_id]

    if st.button("Generate"):
        st.text(
            decode_from_seed(
                seed_tokens,
                model,
                tokenizer,
                top_k=top_k,
                temperature=temp,
                max_len=max_tokens,
            )
        )


if __name__ == "__main__":
    main()
