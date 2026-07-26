from functools import partial

import torch

from datasets import DatasetDict
from fire import Fire
from rich.console import Console

from shakespeare.torch_model import Model


def embed(d, model: Model) -> dict[str, list[torch.Tensor]]:
    all_embeddings: list[torch.Tensor] = []
    all_sequences: list[torch.Tensor] = []
    all_token_probs: list[torch.Tensor] = []

    input_ids = d["input_ids"][:model._max_seq_len].unsqueeze(0)
    token_probs, embs = model.embed(input_ids)
    for token_idx in range(1, len((d["input_ids"]))):
        all_embeddings.append(embs[:token_idx])
        all_sequences.append(input_ids[:token_idx])
        all_token_probs.append(token_probs[:token_idx])

    return {
        "token_probs": all_token_probs,
        "embedding": all_embeddings,
        "sequence": all_sequences,
    }
 

def main(
    model_path: str,
    input_path: str,
    output_path: str,
) -> None:
    console = Console()
    ds = DatasetDict.load_from_disk(input_path)

    model = Model.from_pretrained(model_path)
    embed_fn = partial(embed, model=model)
   
    with torch.no_grad():
        embedded_ds = ds.select_columns("input_ids").with_format("pt").map(embed_fn)
    console.print(f"Columns: {embedded_ds.column_names}")
    embedded_ds.save_to_disk(output_path)


if __name__ == "__main__":
    Fire(main)