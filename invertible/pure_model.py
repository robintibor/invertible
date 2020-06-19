from torch import nn


class NoLogDet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x, logdet = self.model(x)
        return x

    def invert(self, y):
        x, logdet = self.model.invert(y)
        return x
