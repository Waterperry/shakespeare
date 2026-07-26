
import torch

from torch import nn

class SaeModel(nn.Module):
    def __init__(self, in_feat: int, hidden_dim: int, top_k: int = 5) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros((in_feat, )))
        self.w = nn.Parameter(torch.randn((in_feat, hidden_dim)))
        self._top_k: int = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifted = x - self.bias
        embedded = shifted @ self.w

        top_k = torch.topk(embedded, k=self._top_k, dim=-1)
        sparse = torch.zeros_like(embedded)
        sparse = sparse.scatter_(-1, top_k.indices, top_k.values.tanh())
        return sparse

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.w.T) + self.bias