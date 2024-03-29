import torch as th
from torch import nn
import numpy as np

def inverse_elu(y):
    mask = y > 1
    x = th.zeros_like(y)
    x.data[mask] = y.data[mask] - 1
    x.data[1-mask] = th.log(y.data[1-mask])
    return x

def inverse_sigmoid(y):
    return th.log(y / (1-y))

class ActNorm(nn.Module):
    def __init__(self, in_channel, scale_fn, eps=1e-8, verbose_init=True,
                 init_eps=None):
        super().__init__()

        self.bias = nn.Parameter(th.zeros(in_channel))
        self.raw_scale = nn.Parameter(th.zeros(in_channel))

        self.initialize_this_forward = False
        self.initialized = False
        self.scale_fn = scale_fn
        self.eps = eps
        self.verbose_init = verbose_init
        if init_eps is None:
            if scale_fn == 'exp':
                self.init_eps = 1e-6
            elif scale_fn == 'twice_sigmoid':
                self.init_eps = 0.5 + 1e-6 # can max multiply with 2...
            else:
                assert scale_fn == 'elu'
                self.init_eps = 1e-1
        else:
            self.init_eps = init_eps

    def initialize(self, x):
        with th.no_grad():
            # dimensions to standardize over
            # (all dims except channel dim=1)
            other_dims = (0,) + tuple(range(2, len(x.shape)))
            mean = x.mean(dim=other_dims)
            std = x.std(dim=other_dims)

            self.bias.data.copy_(-mean)
            if self.scale_fn == 'exp':
                self.raw_scale.data.copy_(th.log(1 / th.clamp_min(std, self.init_eps)))
            elif self.scale_fn == 'twice_sigmoid':
                self.raw_scale.data.copy_(inverse_sigmoid(0.5 / th.clamp_min(std, self.init_eps)))
            elif self.scale_fn == 'elu':
                self.raw_scale.data.copy_(inverse_elu(1 / th.clamp_min(std, self.init_eps)))
            else:
                assert False

            if self.scale_fn == 'exp':
                multipliers = th.exp(self.raw_scale.squeeze())
            elif self.scale_fn == 'elu':
                multipliers = th.nn.functional.elu(self.raw_scale) + 1
            elif self.scale_fn == 'twice_sigmoid':
                multipliers = th.sigmoid(self.raw_scale) * 2
            if self.verbose_init:
                print(f"Multiplier init to (log10) "
                f"min: {np.log10(th.min(multipliers).item()):3.0f} "
                f"max: {np.log10(th.max(multipliers).item()):3.0f} "
                f"mean: {np.log10(th.mean(multipliers).item()):3.0f}")

    def forward(self, x, fixed=None):
        if not self.initialized:
            assert self.initialize_this_forward, (
                "Please first initialize by setting initialize_this_forward to True"
                " and forwarding appropriate data")
        if self.initialize_this_forward:
            self.initialize(x)
            self.initialized = True
            self.initialize_this_forward = False

        scale, bias, logdet = self.scale_bias_and_logdet_unsqueezed(x)
        y = scale * (x + bias)
        return y, logdet

    def invert(self, z, fixed=None):
        scale, bias, logdet = self.scale_bias_and_logdet_unsqueezed(z)
        x = z / scale - bias
        return x, logdet

    def scale_and_logdet_per_pixel(self):
        if self.scale_fn == 'exp':
            scale = th.exp(self.raw_scale) + self.eps
            if self.eps == 0:
                logdet = th.sum(self.raw_scale)
            else:
                logdet = th.sum(th.log(scale))
        elif self.scale_fn == 'twice_sigmoid':
                # make it centered around  1 to [eps, 2-eps]
            scale = th.sigmoid(self.raw_scale) * (2 - 2 * self.eps) + self.eps
            logdet = th.sum(th.log(scale))
        elif self.scale_fn == 'elu':
            scale = th.nn.functional.elu(self.raw_scale) + 1 + self.eps
            logdet = th.sum(th.log(scale))
        else:
            assert False

        return scale, logdet

    def scale_bias_and_logdet_unsqueezed(self, x):
        scale, log_det_px = self.scale_and_logdet_per_pixel()
        scale = scale.unsqueeze(0)
        bias = self.bias.unsqueeze(0)
        while scale.ndim < x.ndim:
            scale = scale.unsqueeze(-1)
            bias = bias.unsqueeze(-1)
        assert scale.ndim == x.ndim
        assert bias.ndim == x.ndim
        n_dims = np.prod(x.shape[2:])
        logdet = n_dims * log_det_px
        logdet = logdet.repeat(len(x))
        return scale, bias, logdet


def init_act_norm(net, trainloader, n_batches=10, uni_noise_factor=1/255.0):
    if trainloader is not None:
        all_x = []
        for i_batch, (x, y) in enumerate(trainloader):
            all_x.append(x)
            if i_batch >= n_batches:
                break

        init_x = th.cat(all_x, dim=0)
        init_x = init_x.cuda()
        init_x = init_x + th.rand_like(init_x) * uni_noise_factor

        for m in net.modules():
            if hasattr(m, 'initialize_this_forward'):
                m.initialize_this_forward = True

        _ = net(init_x)
    else:
        for m in net.modules():
            if hasattr(m, 'initialize_this_forward'):
                m.initialized = True


class PureActNorm(nn.Module):
    def __init__(self, in_channel, initialized=False):
        super().__init__()
        self.bias = nn.Parameter(th.zeros(in_channel))
        self.scale = nn.Parameter(th.ones(in_channel))
        self.initialize_this_forward = False
        self.initialized = initialized

    def forward(self, x):
        if not self.initialized:
            assert self.initialize_this_forward, (
                "Please first initialize by setting initialize_this_forward to True"
                " and forwarding appropriate data")
        if self.initialize_this_forward:
            self.initialize(x)
            self.initialized = True
            self.initialize_this_forward = False

        bias = self.bias.unsqueeze(0)
        scale = self.scale.unsqueeze(0)
        while scale.ndim < x.ndim:
            scale = scale.unsqueeze(-1)
            bias = bias.unsqueeze(-1)
        y = scale * (x + bias)
        return y

    def initialize(self, x):
        with th.no_grad():
            flatten = x.transpose(0,1).contiguous().view(x.shape[1], -1)
            mean = (
                flatten.mean(1)
            )
            std = (
                flatten.std(1)
            )
            self.bias.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-4))
        print("Multiplier initialized to \n", self.scale.squeeze())
