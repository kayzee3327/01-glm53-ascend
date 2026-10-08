import torch


def prepared_recurrent(*args, **kwargs):
    return torch.ops.sgl_glm53_ascend.prepared_recurrent(
        *args,
        **kwargs,
    )