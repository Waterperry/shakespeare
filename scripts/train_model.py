import json
import os
import re

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

from shakespeare.constants import ATTENTION_STYLE, INFO_STYLE, WARNING_STYLE
from shakespeare.utils import batch_to, get_truthy
from shakespeare.torch_model import Model


def try_parse_epoch_from_path(path: str) -> int | None:
    maybe_match = re.search(r"model_(\d+)\.pt", path)
    if maybe_match:
        epoch = maybe_match.group(1)
        return int(epoch)

    return None


def train_epoch(
    epoch_no: int,
    model: Model,
    dataloader: DataLoader,  # pyright: ignore[reportMissingTypeArgument]
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
    # load params and config
    artefacts_dir: Path = Path(artefacts_path)

    console = Console()
    summary_writer = SummaryWriter(log_dir=Path(artefacts_path).joinpath("tensorboard"))
    params = api.params_show()["train"]

    _device_name: str = os.getenv("TRAIN_DEVICE", "cpu")
    resume_checkpoint: bool = get_truthy(os.getenv("TRAINING_RESUME_CHECKPOINT", "false"))
    checkpoint_path: str | None = os.getenv("TRAINING_CHECKPOINT_PATH")

    # seeding the RNG is less useful if we are restoring from checkpoint since results won't be reproducible, but this is a dev measure anyway...
    seed(params["random_seed"])
    torch.manual_seed(params["random_seed"])

    curr_epoch: int = 0
    model: Model | None = None
    if resume_checkpoint and checkpoint_path:
        if Path(checkpoint_path).exists():
            console.print(f"Resuming training from checkpoint at {checkpoint_path}", style=INFO_STYLE)
            with open(checkpoint_path, "rb") as f:
                model = torch.load(f, map_location=torch.device("cpu"), weights_only=False)  # NOTE: security vulnerability
            maybe_curr_epoch = try_parse_epoch_from_path(checkpoint_path)
            if maybe_curr_epoch is not None:
                curr_epoch = maybe_curr_epoch
        else:
            console.print("`TRAINING_RESUME_CHECKPOINT` was set but no file was found at `TRAINING_CHECKPOINT_PATH`. Ignoring...", style=WARNING_STYLE)
    elif checkpoint_path:
        console.print("`TRAINING_CHECKPOINT_PATH` was specified but `TRAINING_RESUME_CHECKPOINT` was not. Ignoring...", style=WARNING_STYLE)
    elif resume_checkpoint:
        console.print("`TRAINING_RESUME_CHECKPOINT` was specified but `TRAINING_CHECKPOINT_PATH ` was not. Ignoring...", style=WARNING_STYLE)

    device = torch.device(_device_name)
    console.print(f"Using device {device}", style=INFO_STYLE)
    console.print(params, style=INFO_STYLE)

    with open(f"{input_path}_metadata.json") as f:
        ds_metadata: dict[str, Any] = json.load(f)

    dataset = DatasetDict.load_from_disk(input_path).with_format("torch").select_columns("input_ids")
    tokenizer = AutoTokenizer.from_pretrained("./outs/tokenizer")
    if model is None:
        model = Model(vocab_size=ds_metadata["vocab_size"], **params["model_init_params"]).to(device)
    collate_fn = model.get_collate_function(pad_token_id=tokenizer.pad_token_id)
    train_dataloader = DataLoader(dataset["train"], batch_size=8, collate_fn=collate_fn)  # pyright: ignore[reportArgumentType]
    loss_fn = nn.CrossEntropyLoss()
    optim = AdamW(model.parameters())

    for epoch in range(curr_epoch, params["num_epochs"]):
        loss = train_epoch(epoch, model, train_dataloader, device, loss_fn, optim)

        with open(artefacts_dir.joinpath(f"model_{epoch}.pt"), "wb") as f:
            torch.save(model, f)

        summary_writer.add_scalar("loss/item", loss, global_step=epoch)
        print(f"{epoch:0>5} | {loss:.3e}")

    with open(model_output_path, "wb") as f:
        torch.save(model, f)


if __name__ == "__main__":
    Fire(main)
