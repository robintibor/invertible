import torch as th
from torch import nn


class MultipleInputOutput(nn.Module):
    def __init__(self, in_to_out_modules):
        super().__init__()
        self.in_to_out_modules = nn.ModuleList(
            [nn.ModuleList(ms) for ms in in_to_out_modules])

    def forward(self, x):
        all_outs = []
        for in_to_out in self.in_to_out_modules:
            this_outs = [
                m(a_x) for m, a_x in zip(in_to_out, x)
                if m is not None]
            this_out = th.stack(this_outs, dim=0).sum(dim=0)
            all_outs.append(this_out)
        return all_outs