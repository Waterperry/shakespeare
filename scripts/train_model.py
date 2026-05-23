import json
import os

from pathlib import Path
from random import seed
from typing import Any, Callable

import numpy as np
import torch

from datasets import DatasetDict
from dvc import api
from fire import Fire
from rich.console import Console
from rich.progress import track
from torch import nn
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from shakespeare.constants import INFO_STYLE
from shakespeare.utils import batch_to
from shakespeare.torch_model import Model


def train_epoch(
    epoch_no: int,
    model: Model,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optim: Optimizer,
) -> float:
    _losses = []
    for batch in track(dataloader, description=f"Training (epoch {epoch_no})...", transient=True):
        optim.zero_grad()
        batch = batch_to(batch, device=device)
        out = model(batch["input_ids"], batch["input_ids"])

        targets = batch["input_ids"][:, 1:]  # using [SEP] as [EOS] which is naughty but should work
        sources = out[:, :-1].permute(0, 2, 1)  # need to switch vocab dist and seq_len for CELoss

        loss = loss_fn(sources, targets)
        loss.backward()
        optim.step()
        _losses.append(loss.item())

    avg_loss: float = float(np.array(_losses).mean())
    return avg_loss


def main(
    input_path: str,
    model_output_path: str,
    artefacts_path: str,
) -> None:
    console = Console()
    summary_writer = SummaryWriter(log_dir=Path(artefacts_path).joinpath("tensorboard"))

    _device_name: str = os.getenv("TRAIN_DEVICE", "cpu")
    device = torch.device(_device_name)
    console.print(f"Using device {device}", style=INFO_STYLE)

    params = api.params_show()["train"]
    console.print(params, style=INFO_STYLE)

    seed(params["random_seed"])
    torch.manual_seed(params["random_seed"])

    with open(f"{input_path}_metadata.json") as f:
        ds_metadata: dict[str, Any] = json.load(f)

    dataset = DatasetDict.load_from_disk(input_path).with_format("torch").select_columns("input_ids")
    tokenizer = AutoTokenizer.from_pretrained("./outs/tokenizer")
    model = Model(vocab_size=ds_metadata["vocab_size"], **params["model_init_params"]).to(device)
    collate_fn = model.get_collate_function(pad_token_id=tokenizer.pad_token_id)
    train_dataloader = DataLoader(dataset["train"], batch_size=8, collate_fn=collate_fn)  # pyright: ignore[reportArgumentType]
    loss_fn = nn.CrossEntropyLoss()
    optim = AdamW(model.parameters())

    for epoch in range(params["num_epochs"]):
        loss = train_epoch(epoch, model, train_dataloader, device, loss_fn, optim)

        with open(f"outs/model_{epoch}.pt", "wb") as f:
            torch.save(model, f)

        summary_writer.add_scalar("loss/item", loss, global_step=epoch)
        print(f"{epoch:0>5} | {loss:.3e}")

    with open(model_output_path, "wb") as f:
        torch.save(model, f)


if __name__ == "__main__":
    Fire(main)
