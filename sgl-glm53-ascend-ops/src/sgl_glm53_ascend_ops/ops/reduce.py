import torch


def reduce_rows_128(src: torch.Tensor, impl: int = 0) -> torch.Tensor:
    return torch.ops.sgl_glm53_ascend.reduce_rows_128(src, impl)
