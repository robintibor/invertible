from torch import nn


class Inverse(nn.Module):
    def __init__(self, module, invert_logdet_sign):
        super().__init__()
        self.module = module
        self.invert_logdet_sign = invert_logdet_sign

    def forward(self, *args, **kwargs):
        x, logdet = self.module.invert(*args, **kwargs)
        if self.invert_logdet_sign:
            logdet = -logdet
        return x, logdet

    def invert(self, *args, **kwargs):
        x, logdet = self.module.forward(*args, **kwargs)
        if self.invert_logdet_sign:
            logdet = -logdet
        return x, logdet
