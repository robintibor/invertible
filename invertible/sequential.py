from torch import nn
from inspect import signature


class InvertibleSequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.sequential = nn.Sequential(*modules)
        # just always true, in case any submodule,
        # including possibly later added ones accepts the condition
        self.accepts_condition = True

    def forward(self, x, condition=None, fixed=None):
        sum_logdet = 0
        for child in self.sequential.children():

            if condition is not None and hasattr(
                    child, 'accepts_condition') and child.accepts_condition:
                x, logdet = child(x, condition, fixed=fixed)
            else:
                x, logdet = child(x, fixed=fixed)

            if hasattr(logdet, 'shape') and hasattr(sum_logdet, 'shape') and (
                len(logdet.shape) > 0 and len(sum_logdet.shape) > 0):
                assert len(logdet.shape) < 3
                assert len(sum_logdet.shape) < 3
                if len(logdet.shape) > 1 and len(sum_logdet.shape) > 1:
                    # make dimension broadcastable if necessary
                    if logdet.shape[1] == 1 and sum_logdet.shape[1] > 1:
                        logdet = logdet.squeeze(1).unsqueeze(1)
                    if logdet.shape[1] > 1 and sum_logdet.shape[1] == 1:
                        sum_logdet = sum_logdet.squeeze(1).unsqueeze(1)
                while len(sum_logdet.shape) < len(logdet.shape):
                    sum_logdet = sum_logdet.unsqueeze(1)
                while len(logdet.shape) < len(sum_logdet.shape):
                    logdet = logdet.unsqueeze(1)
            sum_logdet = logdet + sum_logdet
        return x, sum_logdet

    def invert(self, y, condition=None, fixed=None):
        sum_logdet = 0
        for child in reversed(list(self.sequential.children())):
            assert hasattr(child, 'invert'), (
                "Class {:s} has no method invert".format(
                    child.__class__.__name__))
            if condition is not None and hasattr(
                child, 'accepts_condition') and child.accepts_condition:
                y, logdet = child.invert(y, condition,
                                         fixed=fixed)
            else:
                y, logdet = child.invert(y,
                                         fixed=fixed)
            if hasattr(logdet, "shape") and hasattr(sum_logdet, "shape"):
                if logdet.ndim < sum_logdet.ndim:
                    logdet = logdet.unsqueeze(-1)
                if logdet.ndim > sum_logdet.ndim:
                    sum_logdet = sum_logdet.unsqueeze(0)
            sum_logdet = logdet + sum_logdet
        return y, sum_logdet
