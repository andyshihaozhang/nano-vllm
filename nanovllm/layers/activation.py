# FFN激活层SwiGLU

import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    # Silu（xWgate）* （xWup)
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1) # dim = -1 按照最后一维拆分
        return F.silu(x) * y
