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
from torch import nn, device as torch_device
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from shakespeare.constants import ATTENTION_STYLE, INFO_STYLE, WARNING_STYLE
from shakespeare.utils import batch_to, get_truthy
from shakespeare.torch_model import Model


@torch.no_grad
def generate_from_scratch(
    model: Model,
    tokenizer: PreTrainedTokenizerFast,
    max_len: int,
    device: torch_device,
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
    sen_tok_ids: list[list[int]] = [[tokenizer.bos_token_id]]  # pyright: ignore[reportAssignmentType]
    while len(sen_tok_ids[0]) < max_len:
        in_tensor = torch.LongTensor(sen_tok_ids).to(device)
        outs = model(in_tensor)
        logits = outs[-1, -1] / temperature
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_ids = probs.topk(top_k)
        out_id = top_k_ids[torch.multinomial(top_k_probs, 1)].item()
        sen_tok_ids[0].append(out_id)
        if out_id == tokenizer.eos_token_id:
            break

    return "".join(tokenizer.convert_ids_to_tokens(sen_tok_ids[0]))
    

def try_parse_epoch_from_path(path: str) -> int | None:
    maybe_match = re.search(r"model_(\d+)", path)
    if maybe_match:
        epoch = maybe_match.group(1)
        return int(epoch)

    return None


@torch.no_grad
def val_epoch(
    epoch_no: int,
    model: Model,
    tokenizer: PreTrainedTokenizerFast,
    dataloader: DataLoader,  # pyright: ignore[reportMissingTypeArgument]
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    _losses = []

    for batch in track(dataloader, description=f"Validation (epoch {epoch_no})...", transient=True):
        _batch = batch_to(batch, device=device)
        input_ids = _batch["input_ids"]
        out = model(input_ids[:, :-1], pad_token_id=tokenizer.pad_token_id)
        targets = input_ids[:, 1:]
        sources = out.permute(0, 2, 1)  # need to switch vocab dist and seq_len for CELoss

        loss = loss_fn(sources, targets)
        _losses.append(loss.item())

    avg_loss: float = float(np.array(_losses).mean())
    return avg_loss


def train_epoch(
    epoch_no: int,
    model: Model,
    tokenizer: PreTrainedTokenizerFast,
    dataloader: DataLoader,  # pyright: ignore[reportMissingTypeArgument]
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optim: Optimizer,
) -> float:
    _losses = []

    for batch in track(dataloader, description=f"Training (epoch {epoch_no})...", transient=True):
        optim.zero_grad()
        _batch = batch_to(batch, device=device)
        input_ids = _batch["input_ids"]
        out = model(input_ids[:, :-1], pad_token_id=tokenizer.pad_token_id)
        targets = input_ids[:, 1:]
        sources = out.permute(0, 2, 1)  # need to switch vocab dist and seq_len for CELoss

        loss = loss_fn(sources, targets)
        loss.backward()
        optim.step()
        _losses.append(loss.item())

    avg_loss: float = float(np.array(_losses).mean())
    optim.zero_grad()
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
    all_params = api.params_show()
    vocab_size = all_params["fit_tokenizer"]["vocab_size"]
    params = all_params["train"]

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
            if maybe_curr_epoch is None:
                console.print("Could not infer current epoch from model path. Will run full training loop.", style=WARNING_STYLE)
            else:
                curr_epoch = maybe_curr_epoch + 1  # +1 since existence of model_X means epoch X finished.
        else:
            console.print("`TRAINING_RESUME_CHECKPOINT` was set but no file was found at `TRAINING_CHECKPOINT_PATH`. Ignoring...", style=WARNING_STYLE)
    elif checkpoint_path:
        console.print("`TRAINING_CHECKPOINT_PATH` was specified but `TRAINING_RESUME_CHECKPOINT` was not. Ignoring...", style=WARNING_STYLE)
    elif resume_checkpoint:
        console.print("`TRAINING_RESUME_CHECKPOINT` was specified but `TRAINING_CHECKPOINT_PATH ` was not. Ignoring...", style=WARNING_STYLE)

    device = torch.device(_device_name)
    console.print(f"Using device {device}", style=INFO_STYLE)
    console.print(params, style=INFO_STYLE)

    dataset = DatasetDict.load_from_disk(input_path).with_format("torch").select_columns(["input_ids", "attention_mask"])
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
    tokenizer = AutoTokenizer.from_pretrained("./outs/tokenizer")
    if model is None:
        model = Model(vocab_size=vocab_size, **params["model_init_params"])
    model = model.to(device)
    collate_fn = model.get_collate_function(pad_token_id=tokenizer.pad_token_id)
    # drop last so we can use a fixed-size BOS-prefix tensor in the training loop
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optim = AdamW(model.parameters(), lr=float(params["learning_rate"]))

    for epoch in range(curr_epoch, params["num_epochs"]):
        model.train()
        train_dataset = train_dataset.shuffle(params["random_seed"] + epoch)
        train_dataloader = DataLoader(train_dataset, batch_size=params["batch_size"], collate_fn=collate_fn)  # pyright: ignore[reportArgumentType]
        loss = train_epoch(epoch, model, tokenizer, train_dataloader, device, loss_fn, optim)

        if epoch % 5 == 0:
            model.save_pretrained(artefacts_dir.joinpath(f"model_{epoch}"))
            model.eval()
            val_dataloader = DataLoader(val_dataset, batch_size=params["batch_size"], collate_fn=collate_fn)  # pyright: ignore[reportArgumentType]
            val_loss = val_epoch(epoch, model, tokenizer, val_dataloader, device, loss_fn)
            summary_writer.add_scalars("loss", {"val": val_loss, "train": loss}, global_step=epoch)
        else:
            summary_writer.add_scalars("loss", {"train": loss}, global_step=epoch)

        generated = generate_from_scratch(model, tokenizer, 100, device)
        console.print(f"{epoch:0>5} | {loss:.3e} | {generated}")

    model.save_pretrained(model_output_path)

if __name__ == "__main__":
    Fire(main)
