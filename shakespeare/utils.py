from torch import Tensor, device as torch_device


def batch_to(batch: dict[str, Tensor], device: torch_device) -> dict[str, Tensor]:
    """
    Move a dict[str, torch.Tensor] (like what comes from a collate function) to a given
        torch device.
    """
    for k, v in batch.items():
        batch[k] = v.to(device)

    return batch


def get_truthy(s: str) -> bool:
    return s.lower() in {'1', 'true', 'yes', 'y', 't'}
