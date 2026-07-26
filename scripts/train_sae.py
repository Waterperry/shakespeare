import os

from typing import Any, Callable

import torch

from datasets import DatasetDict, disable_caching
from dvc import api
from fire import Fire
from rich.console import Console
from rich.progress import track
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import AdamW

from shakespeare.sae import SaeModel


disable_caching()


def train_epoch(
    epoch: int,
    model: SaeModel,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optim: AdamW,
) -> float:
    _losses: list[float] = []

    for batch in track(dataloader, description=f"Training (epoch {epoch})...", transient=True):
        optim.zero_grad()
        non_padding_token_index = batch["input_ids"].squeeze().size(0)
        embeddings = batch["embedding"].squeeze()[:non_padding_token_index].to(device)

        projection = model(embeddings)
        reconstructed = model.decode(projection)
        loss = loss_fn(reconstructed, embeddings)
        loss.backward()
        optim.step()
        _losses.append(loss.item())
    
    return sum(_losses) / len(_losses)


def main(
    dataset_path: str,
    tokenizer_path: str,
    model_output_path: str,
) -> None:
    model_emb_size: int = api.params_show()["train"]["model_init_params"]["embed_dim"]
    params: dict[str, Any] = api.params_show()["train_sae"]
    device: torch.device = torch.device(os.getenv("TRAIN_DEVICE", "cpu"))

    torch.random.manual_seed(params["random_seed"])

    console = Console()
    console.print(params)

    model = SaeModel(model_emb_size, **params["model_init_params"])
    optim = AdamW(model.parameters(), lr=params["learning_rate"])
    loss_fn = MSELoss()

    dataset = DatasetDict.load_from_disk(dataset_path).with_format("torch").select_columns(["embedding", "input_ids"])
    train_ds, val_ds = dataset["train"], dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    for epoch in track(range(params["num_epochs"]), description="Training"):
        model.train()
        train_dataset = train_ds.shuffle(params["random_seed"] + epoch)
        train_dataloader = DataLoader(train_dataset, batch_size=1)  # pyright: ignore[reportArgumentType]
        loss = train_epoch(epoch, model, tokenizer, train_dataloader, device, loss_fn, optim)

        console.print(f"{epoch:0>5} | {loss:.3e}")


if __name__ == "__main__":
    Fire(main)