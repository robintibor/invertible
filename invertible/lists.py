import torch as th
from torch import nn


class ApplyToList(nn.Module):
    """Apply different modules to different inputs.
    First module will be applied to first input, etc.
    So this module expects to receive a list of inputs
    in the forward."""

    def __init__(self, *modules):
        super().__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, xs, fixed=None):
        assert len(xs) == len(self.module_list), (
            f"{len(xs)} xs and {len(self.module_list)} modules")

        ys_logdets = [m(x, fixed=fixed) for m, x in zip(self.module_list, xs)]
        ys, logdets = list(zip(*ys_logdets))
        logdet = sum(logdets)
        return ys, logdet

    def invert(self, ys, fixed=None):
        assert len(ys) == len(self.module_list), (
            f"{len(ys)} ys and {len(self.module_list)} modules")

        xs_logdets = [m.invert(y, fixed=fixed) for m, y in zip(self.module_list, ys)]
        xs, logdets = list(zip(*xs_logdets))
        logdet = sum(logdets)
        return xs, logdet


class ApplyToListNoLogdets(nn.Module):
    """Apply different modules to different inputs.
    First module will be applied to first input, etc.
    So this module expects to receive a list of inputs
    in the forward."""

    def __init__(self, *modules):
        super().__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, xs, fixed=None):
        assert len(xs) == len(self.module_list), (
            f"{len(xs)} xs and {len(self.module_list)} modules")

        outputs = [m(x) for m, x in zip(self.module_list, xs)]
        return outputs


class ApplySplitterToList(object):
    """Apply different splitters to different inputs.
    First splitter will be applied to first input, etc.
    So this splitters expects to receive a list of inputs
    in the split and merge."""

    def __init__(self, *splitters):
        super().__init__()
        self.splitters = splitters

    def split(self, xs):
        assert len(xs) == len(self.splitters), (
            f"{len(xs)} xs and {len(self.splitters)} splitters")
        x1_x2s = [m.split(x) for m, x in zip(self.splitters, xs)]
        x1s, x2s = list(zip(*x1_x2s))
        return x1s, x2s

    def merge(self, y1s, y2s):
        assert len(y1s) == len(y2s) == len(self.splitters), (
            f"{len(y1s)} xs and {len(self.splitters)} splitters")
        ys = [m.merge(y1, y2) for m, y1, y2 in zip(self.splitters, y1s, y2s)]
        return ys


class ModifyList(nn.Module):
    """Apply different modules to different inputs.
    First module will be applied to first input, etc.
    So this module expects to receive a list of inputs
    in the forward."""

    def __init__(self, *modifiers):
        super().__init__()
        self.modifiers = nn.ModuleList(modifiers)

    def forward(self, xs, coefs, fixed=None):
        assert len(xs) == len(coefs) == len(self.modifiers), (
            f"{len(xs)} xs and {len(self.modifiers)} modules")

        ys_logdets = [m(x, c) for m, x, c in zip(self.modifiers, xs, coefs)]
        ys, logdets = list(zip(*ys_logdets))
        logdet = sum(logdets)
        return ys, logdet

    def invert(self, ys, coefs, fixed=None):
        assert len(ys) == len(coefs) == len(self.modifiers), (
            f"{len(ys)} ys and {len(self.modifiers)} modules")
        xs_logdets = [m.invert(y, c) for m, y, c in
                      zip(self.modifiers, ys, coefs)]
        xs, logdets = list(zip(*xs_logdets))
        logdet = sum(logdets)
        return xs, logdet
