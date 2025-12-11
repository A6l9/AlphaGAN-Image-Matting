import torch as tch


class BaseLoss:
    def __call__(self, **kwargs) -> tch.Tensor:
        raise NotImplementedError
