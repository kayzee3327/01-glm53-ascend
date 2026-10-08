import torch


def helloworld(x: torch.Tensor, scalar: float = 1.0) -> torch.Tensor:
    return torch.ops.sgl_glm53_ascend.helloworld(x, scalar)
